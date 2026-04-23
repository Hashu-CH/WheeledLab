"""
Scene / env configuration for the Mushr Racing Task.

Notes:

- RacingTerrainImporterCfg is a pure-data spec (OmegaConf-safe). Track
  geometry and runtime methods live on RacingTerrainImporter in terrain.py;
  reward/event fns reach them via env.scene.terrain.<method>.
- env_spacing is 0 and env.scene.env_origins stays at (0, 0, 0) for every
  env. All state is world-frame: polylines, spawn poses, pose observations,
  reward projections. The env_id == tile_id invariant is what lets each env
  index into its own track's cache — it doesn't require separate origins.
- Lifecycle: MushrRacingSceneCfg.__post_init__ calls terrain.configure(num_envs),
  which writes the USD and stages the TrackCache. Isaac Lab then instantiates
  cfg.class_type(cfg) == RacingTerrainImporter, which pops the cache off the
  stash and holds it for the env's lifetime.

Warning: 

- If you ever add an env-local observation term, like dist from position x-y,
  note that all position states are in world-frame. Meaning that each env's 
  origin is (0,0,0) and not tile local.
"""

import os
import time
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

from .config import CONFIG
from .terrain import RacingTerrainImporter, stash_track_cache
from .utils import create_track_geometry, compute_map_size
from .mdp import (
    RacingEventsCfg,
    RacingEventsRandomCfg,
    RacingObsCfg,
    RacingRewardsCfg,
    RacingTerminationsCfg,
)

_ENV = CONFIG["env"]
_TER = CONFIG["terrain"]


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
    # Pure-data spec. Runtime state (track cache, device tensors) lives on
    # RacingTerrainImporter — see terrain.py. Don't keep non serializable 
    # data on config.
    class_type: type = RacingTerrainImporter # runtime class to instantiate

    # Map generation parameters to make it realistic for how we test in real
    row_spacing = float(_TER["row_spacing"])
    col_spacing = float(_TER["col_spacing"])
    spacing = (row_spacing, col_spacing)

    # Sub-environments size to make it realistic for how we test in real
    env_num_rows = int(_TER["env_num_rows"])
    env_num_cols = int(_TER["env_num_cols"])
    env_size = (env_num_rows, env_num_cols)

    # Whether to sample colors
    color_sampling = bool(_TER["color_sampling"])

    # Sizing + USD generation deferred to `configure()` so they can depend on num_envs.
    num_rows = 0
    num_cols = 0
    map_size = (0, 0)
    width = 0.0
    height = 0.0
    file_name = ""

    car_width_m: float = float(_TER["car_width_m"])

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
        """Populate map-sizing fields, write the track USD, and stage the
        per-tile track cache for the paired RacingTerrainImporter runtime.

        Called from MushrRacingSceneCfg after num envs is initialized.

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
        track_cache = create_track_geometry(
            self.file_name, self.map_size, self.spacing, self.env_size, self.color_sampling,
        )
        # Hand off to RacingTerrainImporter.__init__, which pops it at scene-build time.
        stash_track_cache(self, track_cache)


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
    seed: int = _ENV["seed"]
    num_envs: int = 512 # default value, truth in rss config
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
        self.sim.dt = float(_ENV["sim_dt"])
        self.decimation = int(_ENV["decimation"])
        self.episode_length_s = float(_ENV["episode_length_s"])
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
