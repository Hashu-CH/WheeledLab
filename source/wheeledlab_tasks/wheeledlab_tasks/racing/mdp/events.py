"""
Contains reset helpers and agent randomizations for the Racing Task.

Notes:

- Our 'envs' are parallel robot instances running on a shared terrain. We
  establish the invariant: env_id == tile_id, meaning each env always spawns
  on its assigned track on termination. This lets reward functions read per-env
  track geometry by simple env_id indexing into
  env.scene.terrain.track_cache (a RacingTerrainImporter runtime attr).
- All spatial state in this task is expressed in WORLD coordinates —
  polylines, spawn poses, pose observations, reward projections. Isaac Lab's
  env_spacing is 0 and env.scene.env_origins is left at its default of
  (0, 0, 0) for every env.
"""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import Articulation, RigidObject
from isaaclab.terrains import TerrainImporter

from ..config import CONFIG

_EV = CONFIG["events"]
_WF = _EV["wheel_friction"]
_GOALS = CONFIG.get("goals", {})


# ---------------------------------------------------------------------------
# Reset Helper Functions
# ---------------------------------------------------------------------------
def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_offset_deg_range: tuple[float, float] = (-30.0, 30.0),
):
    """Reset a set (env_ids) of environments to random poses on their
    respective tracks. Invoked by the event manager on env termination.

    Args:
    - env: the running environment
    - env_ids: id's of the envs that we are going to reset
    - asset_cfg: asset handle on our robot
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # env_ids is threaded through so each env spawns inside its own tile.
    valid_poses = terrain.generate_random_poses(
        len(env_ids), env_ids=env_ids,
        yaw_offset_deg_range=yaw_offset_deg_range,
    )

    # Unpack valid_poses (a list of InitialPoseCfgs) into sim-usable tensors.
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

    # default root state is 0 but allows for changes to be reflected accurately later
    positions = posns + asset.data.default_root_state[env_ids, :3]
    orientations = oris
    lin_vels = lin_vels + asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels + asset.data.default_root_state[env_ids, 10:13]

    # Write pose/vel into the sim, scoped to just the resetting envs.
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)


def init_progress_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    loop_dist_frac_range: tuple[float, float] = (0.3, 0.95),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Seed per-env distance-to-finish + goal arc-length on the terrain.

    Must run AFTER reset_root_state in the same reset batch so the spawn pose
    has been written and we can project it onto the centerline. 
    -- Execution order follows field order in RacingEventsCfg. -- 

    Args:
    - env: the running environment
    - env_ids: ids of the envs being reset
    - loop_dist_frac_range: (lo, hi) fraction of how much dist to project 
      finish line forward
    - asset_cfg: handle for the racing robot
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    spawn_xy = asset.data.root_pos_w[env_ids, :2]
    terrain.init_progress(env_ids, spawn_xy, loop_dist_frac_range)


# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingEventsCfg:
    """Event Config Specification"""
    reset_root_state = EventTerm(
        func=reset_root_state,
        mode="reset",
        params={
            "yaw_offset_deg_range": tuple(_EV.get("spawn_yaw_offset_deg_range", [-30.0, 30.0])),
        },
    )
    init_progress = EventTerm(
        func=init_progress_state,
        mode="reset",
        params={
            "loop_dist_frac_range": tuple(_GOALS.get("loop_dist_frac_range", [0.3, 0.95])),
        },
    )


@configclass
class RacingEventsRandomCfg(RacingEventsCfg):
    """Adds robot randomizations"""
    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": tuple(_WF["static_friction_range"]),
            "dynamic_friction_range": tuple(_WF["dynamic_friction_range"]),
            "restitution_range": tuple(_WF["restitution_range"]),
            "num_buckets": int(_WF["num_buckets"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "make_consistent": bool(_WF["make_consistent"]),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": tuple(_EV["base_mass_range"]),
            "operation": "abs",
        },
    )

    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "mass_distribution_params": tuple(_EV["wheel_mass_range"]),
            "operation": "abs",
        },
    )
