"""
Scene / env configuration for the Mushr Racing Task.

Notes:

- RacingTerrainImporterCfg extends Isaac's TerrainImporterCfg with the
  track cache so reward functions can pull from it via env.scene.terrain.cfg.
- env_spacing is 0 and env.scene.env_origins stays at (0, 0, 0) for every
  env. All state is world-frame: polylines, spawn poses, pose observations,
  reward projections. The env_id == tile_id invariant is what lets each env
  index into its own track's cache — it doesn't require separate origins.
- Lifecycle: MushrRacingSceneCfg.__post_init__ calls terrain.configure(num_envs),
  which generates the USD and populates track_cache before any reset fires.

Warning: 

- If you ever add an env-local observation term, like dist from position x-y,
  note that all position states are in world-frame. Meaning that each env's 
  origin is (0,0,0) and not tile local.
"""

import os
import time
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg

from .utils import create_track_geometry, compute_map_size, TrackCache
from .utils.track_utils import project_nearest_segment, sample_poses_along_polylines
from .mdp import (
    RacingEventsCfg,
    RacingEventsRandomCfg,
    RacingObsCfg,
    RacingRewardsCfg,
    RacingTerminationsCfg,
)


# ---------------------------------------------------------------------------
# Spawn pose dataclass (consumed by reset_root_state in events.py)
# ---------------------------------------------------------------------------
@configclass
class InitialPoseCfg:
    """One spawn pose per resetting env. reset_root_state stacks these into
    tensors and writes them to the sim. pos/lin_vel/ang_vel are WORLD-frame;
    rot is absolute yaw in degrees."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Terrain Importer
# ---------------------------------------------------------------------------
@configclass
class RacingTerrainImporterCfg(TerrainImporterCfg):
    # Map generation parameters to make it realistic for how we test in real
    row_spacing = 0.5
    col_spacing = 0.5
    spacing = (row_spacing, col_spacing)

    # Sub-environments size to make it realistic for how we test in real
    env_num_rows = 12
    env_num_cols = 12
    env_size = (env_num_rows, env_num_cols)

    # Whether to sample colors
    color_sampling = False

    # Sizing + USD generation deferred to `configure()` so they can depend on num_envs.
    num_rows = 0
    num_cols = 0
    map_size = (0, 0)
    width = 0.0
    height = 0.0
    file_name = ""

    # Track tile geometry (set in configure after init)
    track_cache: TrackCache = None
    _polylines_t: torch.Tensor = None
    _tangents_t: torch.Tensor = None  
    _segment_valid_t: torch.Tensor = None
    _track_widths_t: torch.Tensor = None 
    car_width_m: float = 0.28

    # usd setup
    prim_path = "/World/plane"
    terrain_type = "usd"
    usd_path = ""
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=2.0,
        dynamic_friction=2.0,
    )
    debug_vis = False

    def configure(self, num_envs: int):
        """Populate map-sizing fields and generate the track USD for `num_envs` sub-envs.

        Called from MushrRacingSceneCfg after num envs is initialized. 
        Writes the USD and populates the track cache.

        Args:
        - num_envs: total parallel envs (sets tile grid size + track count)
        """
        self.num_rows, self.num_cols = compute_map_size(num_envs, self.env_size)
        self.map_size = (self.num_rows, self.num_cols)
        self.width = self.num_rows * self.row_spacing
        self.height = self.num_cols * self.col_spacing
        self.file_name = os.path.join(
            WHEELEDLAB_ASSETS_DATA_DIR, 'rgb_maps', time.strftime("%Y%m%d_%H%M%S.usd"),
        )
        self.usd_path = self.file_name
        # lives on env.scene.terrain.cfg.track_cache
        self.track_cache = create_track_geometry(
            self.file_name, self.map_size, self.spacing, self.env_size, self.color_sampling,
        )


    # ---------------------------------------------------------------------------
    # Stateful Entry Points for Utility Functions
    # ---------------------------------------------------------------------------

    def _ensure_track_tensors(self, device):
        """Move cache to GPU 

        Track cache lives on numpy arrays when configured. We need it to be on device
        torch tensors. We don't know device until car pose is passed to gpu in 
        project_to_centerline (see below)

        Args:
        - device: torch device of the incoming query (usually env.device)
        """
        if self._polylines_t is not None and self._polylines_t.device == device:
            # if tensors are already on device then skip
            return
        tc = self.track_cache
        self._polylines_t = torch.as_tensor(tc.polylines_w, dtype=torch.float32, device=device)
        self._tangents_t = torch.as_tensor(tc.tangents_w, dtype=torch.float32, device=device)
        self._segment_valid_t = torch.as_tensor(tc.segment_valid, dtype=torch.bool, device=device)
        self._track_widths_t = torch.as_tensor(tc.track_widths_m, dtype=torch.float32, device=device)

    def project_to_centerline(self, poses_xy_w: torch.Tensor, env_ids: torch.Tensor):
        """Project each car's (x, y) onto its assigned tile's centerline polyline.

        Method primarily to separate state viewing and actual computation done in 
        track utils. Used for rewards.

        Args:
        - poses_xy_w: (N, 2) world-meter car positions
        - env_ids: (N,) int64 tile indices (env_id == tile_id by invariant)

        Returns:
        - d_signed: (N,) signed perpendicular distance, pos = left of tangent
        - tangent_xy: (N, 2) unit tangent at the winning segment
        - track_width: (N,) world-meter track width (used by cross_track_penalty)
        """
        self._ensure_track_tensors(poses_xy_w.device) # upload to device
        polylines = self._polylines_t[env_ids]
        tangents = self._tangents_t[env_ids]
        valid = self._segment_valid_t[env_ids]
        track_widths = self._track_widths_t[env_ids]

        d_signed, nearest_tangent, _ = project_nearest_segment(
            polylines, tangents, valid, poses_xy_w,
        )
        return d_signed, nearest_tangent, track_widths

    def generate_random_poses(self, num_poses, env_ids=None):
        """Sample spawn poses for a reset batch.

        Method primarily to separate state viewing and actual computation done in 
        util init. See track utils for details on how poses are sampled. Used for rewards

        Args:
        - num_poses: how many poses to sample
        - env_ids: tensor of env IDs (required — spawn is polyline-based
          and needs to know which tile's track to sample from)
        """
        if env_ids is None or self.track_cache is None:
            raise ValueError(
                "requires populated track cache and env ids. Configure() hasn't run yet"
            )
        init_poses = sample_poses_along_polylines(
            self.track_cache, env_ids,
            car_width_m=self.car_width_m,
            margin_m=0.0,
        )
        return [
            InitialPoseCfg(
                pos=(x, y, 0.1),
                rot_euler_xyz_deg=(0., 0., angle),
            ) for x, y, angle in init_poses
        ]


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------
@configclass
class MushrRacingSceneCfg(InteractiveSceneCfg):
    """Configuration for a Mushr car scene with a racetrack terrain and sensors."""
    terrain = RacingTerrainImporterCfg()
    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn=sim_utils.GroundPlaneCfg(
            size=(1.0, 1.0),  # overwritten in __post_init__ once terrain is configured
            color=(0, 0, 0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=2.0,
                dynamic_friction=2.0,
            ),
        ),
    )
    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground.init_state.pos = (0.0, 0.0, -1e-4)

    # TODO wire hyper parameters from file so that all hyper params are observable at once
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/camera_link/camera",
        update_period=0.1,
        height=60,
        width=80,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.9299999475479126,
            horizontal_aperture=3.8959999084472656,
            vertical_aperture=2.453000068664551,
            clipping_range=(0.01, 1e2),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.08, 0.0, 0.0),
            rot=tuple(R.from_euler('xyz', [-90.0, 0.0, -90.0], degrees=True).as_quat().tolist()),
            convention="ros",
        ),
        debug_vis=False,
    )

    # Called by Env config to initialize scene -> TerrainImporter configure called
    def __post_init__(self):
        super().__post_init__()
        self.terrain.configure(self.num_envs) # inits track and cache
        self.ground.spawn.size = (self.terrain.width, self.terrain.height)
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# RL Training Config
# ---------------------------------------------------------------------------
@configclass
class MushrRacingRLEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.

    events: RacingEventsCfg = RacingEventsRandomCfg()
    actions: Mushr4WDActionCfg = Mushr4WDActionCfg()

    observations: RacingObsCfg = RacingObsCfg()
    rewards: RacingRewardsCfg = RacingRewardsCfg()
    terminations: RacingTerminationsCfg = RacingTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = [40., 0.0, 45.0]
        self.viewer.lookat = [0.0, 0.0, -3.]
        self.sim.dt = 0.02 # physics sim timestep in seconds 50Hz = .02
        self.decimation = 10 # num .dt ticks per policy step 
        self.episode_length_s = 20 # max episode length in seconds
        self.scene = MushrRacingSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )


# ---------------------------------------------------------------------------
# Inference Only Config (No Terminations or Rewards)
# ---------------------------------------------------------------------------
@configclass
class MushrRacingPlayEnvCfg(MushrRacingRLEnvCfg):
    """No terminations, deterministic reset."""
    rewards: RacingRewardsCfg = None
    terminations: RacingTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()
