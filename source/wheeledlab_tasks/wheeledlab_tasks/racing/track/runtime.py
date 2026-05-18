"""
Runtime component to RacingTerrainImporterCfg.

Must live off cfg since torch buffers are non-serializable (OmegaConf and
Hydra requirement).

Notes:

- RacingTerrainImporterCfg.configure() runs during scene cfg post_init and
  only does cheap sizing math (map dims, usd_path). No USD is written.

- InteractiveScene instantiation calls cfg.class_type(cfg) == this class.
  We author the track USD here (writes file at cfg.usd_path) and then call
  the base TerrainImporter, which loads that file. The returned TrackCache
  is stored on self for reward/event funcs.

  This deferral keeps `import wheeledlab_rl` cheap — track authoring used
  to fire at module import time via __post_init__, which got expensive
  with cone PointInstancers across many envs.

Call sites (reward/event funcs) read env.scene.terrain.<method>
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

from .generator import TrackCache, create_track_geometry
from .projection import project_nearest_segment, sample_poses_along_polylines

# avoid ciruclar import
if TYPE_CHECKING:
    from ..mushr_racing_env_cfg import RacingTerrainImporterCfg, InitialPoseCfg


# I don't really like this. Track utils have to live here since the track
# cache state is exposed and can't be in other functions like rewards.
class RacingTerrainImporter(TerrainImporter):
    """TerrainImporter that carries per-tile track geometry for racing rewards."""

    # init called on InteractiveScene Env construction
    def __init__(self, cfg: "RacingTerrainImporterCfg"):
        # Author the USD on disk before the base class loads cfg.usd_path.
        # cfg.configure(num_envs) must have run earlier to set the sizing fields.
        track_cache = create_track_geometry(
            cfg.usd_path, cfg.map_size, cfg.spacing, cfg.env_size,
            cfg.color_sampling, cfg.tile_padding_cells,
        )
        super().__init__(cfg)  # base class loads cfg.usd_path
        self.track_cache: TrackCache = track_cache
        self._polylines_t: torch.Tensor | None = None
        self._tangents_t: torch.Tensor | None = None
        self._segment_valid_t: torch.Tensor | None = None
        self._track_widths_t: torch.Tensor | None = None
        self._tile_origins_t: torch.Tensor | None = None
        self._segment_lengths_t: torch.Tensor | None = None
        self._cumulative_arc_t: torch.Tensor | None = None
        self._total_lengths_t: torch.Tensor | None = None
        self._is_closed_t: torch.Tensor | None = None
        self.prev_dist_to_finish: torch.Tensor | None = None
        self.goal_arc_length: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Helper to pull CPU numpy cache to CUDA device for torch op
    # ------------------------------------------------------------------
    def _ensure_track_tensors(self, device: torch.device) -> None:
        """Lazily upload the numpy cache to `device`. Re-uploads if the device
        changes (rare — only on first query).
        
        CALL BEFORE METHODS ON TERRAIN IMPORTER"""
        if self._polylines_t is not None and self._polylines_t.device == device:
            return
        tc = self.track_cache
        self._polylines_t = torch.as_tensor(tc.polylines_w, dtype=torch.float32, device=device)
        self._tangents_t = torch.as_tensor(tc.tangents_w, dtype=torch.float32, device=device)
        self._segment_valid_t = torch.as_tensor(tc.segment_valid, dtype=torch.bool, device=device)
        self._track_widths_t = torch.as_tensor(tc.track_widths_m, dtype=torch.float32, device=device)
        self._tile_origins_t = torch.as_tensor(tc.tile_origins_w, dtype=torch.float32, device=device)
        self._segment_lengths_t = torch.as_tensor(tc.segment_lengths_m, dtype=torch.float32, device=device)
        self._cumulative_arc_t = torch.as_tensor(tc.cumulative_arc_lengths_m, dtype=torch.float32, device=device)
        self._total_lengths_t = torch.as_tensor(tc.total_lengths_m, dtype=torch.float32, device=device)
        self._is_closed_t = torch.as_tensor(tc.is_closed, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------
    # Helper utilities for interacting with geometry features of track/render
    # ------------------------------------------------------------------
    def project_to_centerline(
        self, poses_xy_w: torch.Tensor, env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centerline projection with arc-length along the polyline.

        Args:
        - poses_xy_w: (N, 2) world-meter car positions
        - env_ids: (N,) int tile indices (env_id == tile_id by invariant)

        Returns:
        - d_signed: (N,) signed perpendicular distance, pos = left of tangent
        - tangent_xy: (N, 2) unit tangent (approx) at the winning segment
        - track_width: (N,) world-meter track width per env
        - arc_length: (N,) world-meter arc-length along the polyline at the
          projected foot, in [0, total_length]
        - seg_idx: (N,) winning segment index
        """
        self._ensure_track_tensors(poses_xy_w.device) 

        # unpack
        polylines = self._polylines_t[env_ids]
        tangents = self._tangents_t[env_ids]
        valid = self._segment_valid_t[env_ids]
        track_widths = self._track_widths_t[env_ids]
        cumulative = self._cumulative_arc_t[env_ids]
        seg_lengths = self._segment_lengths_t[env_ids]

        d_signed, nearest_tangent, seg_idx, t_param = project_nearest_segment(
            polylines, tangents, valid, poses_xy_w,
        )
        seg_start_arc = cumulative.gather(1, seg_idx.unsqueeze(-1)).squeeze(-1)
        seg_len = seg_lengths.gather(1, seg_idx.unsqueeze(-1)).squeeze(-1)
        arc_length = seg_start_arc + t_param * seg_len
        return d_signed, nearest_tangent, track_widths, arc_length, seg_idx


    def update_progress(self, env_ids: torch.Tensor, root_pos_xy_w: torch.Tensor) -> torch.Tensor:
        """Recompute distance-to-finish, return per-env reward delta.

        delta = prev_dist - curr_dist where dist = (goal_arc - arc_pos) mod total_length.
        
        NOTE: Storing prev_dist within buffer is non-ideal. Resuming training
        it would be broken until the first init_progress call. ie, this would not 
        work for reward functions tuples offline -> reward is tied directly to 
        the full trajectory.

        Args:
        - env_ids: (N,) int tile indices to update.
        - root_pos_xy_w: (N, 2) world-meter root position for those envs.

        Returns:
        - delta: (N,) reward delta in meters; positive = moving toward finish.
        """
        self._ensure_track_tensors(root_pos_xy_w.device)
        if self.prev_dist_to_finish is None or self.goal_arc_length is None:
            raise RuntimeError(
                "RacingTerrainImporter: progress state not initialized. "
                "Make sure init_progress runs as a reset event before update_progress."
            )
        _, _, _, arc_length, _ = self.project_to_centerline(root_pos_xy_w, env_ids)
        total = self._total_lengths_t[env_ids].clamp_min(1e-6)
        goal_arc = self.goal_arc_length[env_ids]
        prev_dist = self.prev_dist_to_finish[env_ids]
        curr_dist = torch.remainder(goal_arc - arc_length, total)
        delta = prev_dist - curr_dist
        self.prev_dist_to_finish[env_ids] = curr_dist
        return delta

    def init_progress(
        self,
        env_ids: torch.Tensor,
        root_pos_xy_w: torch.Tensor,
        loop_dist_frac_range: tuple[float, float],
    ) -> None:
        """Resetter, seeds prev_dist_to_finish and goal arc length for envs
        
        Track Types:
        - Chains: goal is the polyline endpoint -> initial_dist = total - spawn_arc.
        - Loops: initial_dist = uniform(*loop_dist_frac_range) * total

        Args:
        - env_ids: (N,) int tile indices being reset.
        - root_pos_xy_w: (N, 2) world-meter spawn positions.
        - loop_dist_frac_range: (lo, hi) fraction of perimeter for closed loops.
        """
        self._ensure_track_tensors(root_pos_xy_w.device)
        device = root_pos_xy_w.device
        num_envs = self._total_lengths_t.shape[0]
        if self.prev_dist_to_finish is None: # buffer out 
            self.prev_dist_to_finish = torch.zeros(num_envs, device=device)
            self.goal_arc_length = torch.zeros(num_envs, device=device)

        _, _, _, spawn_arc, _ = self.project_to_centerline(root_pos_xy_w, env_ids)
        total = self._total_lengths_t[env_ids].clamp_min(1e-6)
        is_closed = self._is_closed_t[env_ids]

        # how far from goal for loops and chains
        lo, hi = float(loop_dist_frac_range[0]), float(loop_dist_frac_range[1])
        loop_frac = torch.empty_like(spawn_arc).uniform_(lo, hi)
        loop_initial = loop_frac * total
        chain_initial = (total - spawn_arc).clamp_min(0.0)
        initial_dist = torch.where(is_closed, loop_initial, chain_initial)

        # seeds initial progress (mod for loops)
        goal_arc = torch.remainder(spawn_arc + initial_dist, total)
        self.prev_dist_to_finish[env_ids] = initial_dist
        self.goal_arc_length[env_ids] = goal_arc

    def out_of_tile(self, poses_xy_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        """True when a car is outside its own tile's world-frame bounding box.

        Since env_spacing=0, every env's origin is world (0,0,0) but each env's
        assigned tile sits at a different world offset inside the shared plane
        (see track_cache.tile_origins_w). 

        Args:
        - poses_xy_w: (N, 2) world-meter car positions
        - env_ids: (N,) int64 tile indices (env_id == tile_id by invariant)

        Returns:
        - (N,) bool — True if the car is outside its tile.
        """
        self._ensure_track_tensors(poses_xy_w.device)
        origins = self._tile_origins_t[env_ids]   # (N, 2), world-meter tile centers
        wx, wy = self.track_cache.tile_extent_m   # (meters)
        local = poses_xy_w - origins
        x_out = local[..., 0].abs() > (0.5 * wx)
        y_out = local[..., 1].abs() > (0.5 * wy)
        return torch.logical_or(x_out, y_out)

    def generate_random_poses(
        self, 
        num_poses: int, 
        env_ids=None,
        yaw_offset_deg_range: tuple[float, float] = (-30.0, 30.0),
    ) -> list["InitialPoseCfg"]:
        """Sample spawn poses for a reset batch.

        Args:
        - num_poses: how many poses to sample
        - env_ids: tensor of env IDs (required — spawn is polyline-based and
          needs to know which tile's track to sample from)
        """
        # Local import to avoid circular: cfg module imports this module for class_type.
        from ..mushr_racing_env_cfg import InitialPoseCfg

        if env_ids is None:
            raise ValueError("generate_random_poses requires env_ids")
        init_poses = sample_poses_along_polylines(
            self.track_cache, env_ids,
            car_width_m=self.cfg.car_width_m,
            margin_m=0.0,
            yaw_offset_deg_range=yaw_offset_deg_range,
        )
        return [
            InitialPoseCfg(
                pos=(x, y, 0.1),
                rot_euler_xyz_deg=(0., 0., angle),
            ) for x, y, angle in init_poses
        ]
