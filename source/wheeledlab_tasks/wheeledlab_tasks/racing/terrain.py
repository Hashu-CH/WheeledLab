"""
Runtime counterpart to RacingTerrainImporterCfg.

Lives off the cfg so numpy/torch buffers (polylines, tangents, cached device
tensors) aren't visible to OmegaConf when the env cfg is wrapped for Hydra
registration. 

Notes - Handoff pattern:
- RacingTerrainImporterCfg.configure() runs during scene __post_init__,
  writes the USD, and deposits the populated TrackCache into _PENDING_CACHES
  keyed by id(cfg).
- When InteractiveScene instantiates the terrain via cfg.class_type(cfg),
  RacingTerrainImporter.__init__ pops that entry and attaches it.

Call sites (reward/event fns) read `env.scene.terrain.<method>` 
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

from .utils import TrackCache
from .utils.track_utils import project_nearest_segment, sample_poses_along_polylines

if TYPE_CHECKING:
    from .mushr_racing_env_cfg import RacingTerrainImporterCfg, InitialPoseCfg


# Keyed by id(cfg) so multiple concurrent configs (param sweeps, tests) don't
# collide. Populated by the cfg's configure(); drained by __init__ below.
_PENDING_CACHES: dict[int, TrackCache] = {}


def stash_track_cache(cfg, cache: TrackCache) -> None:
    _PENDING_CACHES[id(cfg)] = cache


class RacingTerrainImporter(TerrainImporter):
    """TerrainImporter that carries per-tile track geometry for racing rewards."""

    # init called on InteractiveScene Env construction
    def __init__(self, cfg: "RacingTerrainImporterCfg"):
        super().__init__(cfg)
        cache = _PENDING_CACHES.pop(id(cfg), None)
        if cache is None:
            raise RuntimeError(
                "RacingTerrainImporter: no track cache staged for this cfg. "
                "Make sure RacingTerrainImporterCfg.configure(num_envs) runs "
                "before the scene instantiates the terrain."
            )
        self.track_cache: TrackCache = cache
        self._polylines_t: torch.Tensor | None = None
        self._tangents_t: torch.Tensor | None = None
        self._segment_valid_t: torch.Tensor | None = None
        self._track_widths_t: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Device-side tensor cache
    # ------------------------------------------------------------------
    def _ensure_track_tensors(self, device: torch.device) -> None:
        """Lazily upload the numpy cache to `device`. Re-uploads if the device
        changes (rare — only on first query)."""
        if self._polylines_t is not None and self._polylines_t.device == device:
            return
        tc = self.track_cache
        self._polylines_t = torch.as_tensor(tc.polylines_w, dtype=torch.float32, device=device)
        self._tangents_t = torch.as_tensor(tc.tangents_w, dtype=torch.float32, device=device)
        self._segment_valid_t = torch.as_tensor(tc.segment_valid, dtype=torch.bool, device=device)
        self._track_widths_t = torch.as_tensor(tc.track_widths_m, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Reward / reset entry points (used by mdp.rewards and mdp.events)
    # ------------------------------------------------------------------
    def project_to_centerline(
        self, poses_xy_w: torch.Tensor, env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project each car's (x, y) onto its assigned tile's centerline polyline.

        Args:
        - poses_xy_w: (N, 2) world-meter car positions
        - env_ids: (N,) int64 tile indices (env_id == tile_id by invariant)

        Returns:
        - d_signed: (N,) signed perpendicular distance, pos = left of tangent
        - tangent_xy: (N, 2) unit tangent at the winning segment
        - track_width: (N,) world-meter track width (used by cross_track_penalty)
        """
        self._ensure_track_tensors(poses_xy_w.device)
        polylines = self._polylines_t[env_ids]
        tangents = self._tangents_t[env_ids]
        valid = self._segment_valid_t[env_ids]
        track_widths = self._track_widths_t[env_ids]

        d_signed, nearest_tangent, _ = project_nearest_segment(
            polylines, tangents, valid, poses_xy_w,
        )
        return d_signed, nearest_tangent, track_widths

    def generate_random_poses(self, num_poses: int, env_ids=None) -> list["InitialPoseCfg"]:
        """Sample spawn poses for a reset batch.

        Args:
        - num_poses: how many poses to sample
        - env_ids: tensor of env IDs (required — spawn is polyline-based and
          needs to know which tile's track to sample from)
        """
        # Local import to avoid circular: cfg module imports this module for class_type.
        from .mushr_racing_env_cfg import InitialPoseCfg

        if env_ids is None:
            raise ValueError("generate_random_poses requires env_ids")
        init_poses = sample_poses_along_polylines(
            self.track_cache, env_ids,
            car_width_m=self.cfg.car_width_m,
            margin_m=0.0,
        )
        return [
            InitialPoseCfg(
                pos=(x, y, 0.1),
                rot_euler_xyz_deg=(0., 0., angle),
            ) for x, y, angle in init_poses
        ]
