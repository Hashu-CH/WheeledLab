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

from .utils import generate_random_poses, create_track_geometry, compute_map_size
from .mdp import (
    RacingEventsCfg,
    RacingEventsRandomCfg,
    RacingObsCfg,
    RacingRewardsCfg,
    RacingTerminationsCfg,
)


@configclass
class InitialPoseCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)


#####################
######## SCENE ######
#####################

@configclass
class RacingTerrainImporterCfg(TerrainImporterCfg):
    # Map generation parameters
    row_spacing = 0.5
    col_spacing = 0.5
    spacing = (row_spacing, col_spacing)

    # Sub-environments sized to match the drifting track's longest side (~5.6m).
    # 12 cells * 0.5m spacing = 6m per sub-env side.
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
    traversability_hashmap = None

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
        """Populate map-sizing fields and generate the track USD for `num_envs` sub-envs."""
        self.num_rows, self.num_cols = compute_map_size(num_envs, self.env_size)
        self.map_size = (self.num_rows, self.num_cols)
        self.width = self.num_rows * self.row_spacing
        self.height = self.num_cols * self.col_spacing
        self.file_name = os.path.join(
            WHEELEDLAB_ASSETS_DATA_DIR, 'rgb_maps', time.strftime("%Y%m%d_%H%M%S.usd"),
        )
        self.usd_path = self.file_name
        self.traversability_hashmap = create_track_geometry(
            self.file_name, self.map_size, self.spacing, self.env_size, self.color_sampling,
        )

    def generate_random_poses(self, num_poses):
        init_poses = generate_random_poses(
            num_poses, self.row_spacing, self.col_spacing,
            self.traversability_hashmap, margin=0.1,
        )
        return [
            InitialPoseCfg(
                pos=(x, y, 0.1),
                rot_euler_xyz_deg=(0., 0., angle),
            ) for x, y, angle in init_poses
        ]

    def get_traversability(self, poses):
        xs, ys = poses[:, 0], poses[:, 1]
        x_idx, y_idx = self.get_map_id(xs, ys)
        return torch.tensor(self.traversability_hashmap).to(x_idx.device)[x_idx, y_idx]

    def get_map_id(self, x, y):
        x_idx = torch.floor((x + self.width / 2 - self.row_spacing / 2) / self.row_spacing).long()
        y_idx = torch.floor((y + self.height / 2 - self.col_spacing / 2) / self.col_spacing).long()
        x_idx = torch.clamp(x_idx, 0, self.num_rows - 1)
        y_idx = torch.clamp(y_idx, 0, self.num_cols - 1)
        return x_idx, y_idx


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

    def __post_init__(self):
        super().__post_init__()
        self.terrain.configure(self.num_envs)
        self.ground.spawn.size = (self.terrain.width, self.terrain.height)
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.0))


######################
####### ENV CFG ######
######################

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
        self.sim.dt = 0.02
        self.decimation = 10
        self.episode_length_s = 10
        self.scene = MushrRacingSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )


######################
###### PLAY ENV ######
######################

@configclass
class MushrRacingPlayEnvCfg(MushrRacingRLEnvCfg):
    """No terminations, deterministic reset."""
    rewards: RacingRewardsCfg = None
    terminations: RacingTerminationsCfg = None

    def __post_init__(self):
        super().__post_init__()
