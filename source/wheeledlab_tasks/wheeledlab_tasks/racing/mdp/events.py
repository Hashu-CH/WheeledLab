"""
Contains reset helpers and agent randomizations for the Racing Task.

Notes:

- Our 'envs' are parallel robot instances running on a shared terrain. We
  establish the invariant: env_id == tile_id, meaning each env always spawns
  on its assigned track on termination. This lets reward functions read per-env
  track geometry by simple env_id indexing into RacingTerrainImporterCfg.
  track_cache.
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


# ---------------------------------------------------------------------------
# Reset Helper Functions
# ---------------------------------------------------------------------------
def reset_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset a set (env_ids) of environments to random poses on their
    respective tracks. Invoked by the event manager on env termination.

    generate_random_poses returns WORLD-frame coords sampled along each env's
    centerline polyline. The asset's default_root_state is (0, 0, 0) for all
    envs (env_spacing=0, init_state.pos=0), so the subsequent add is a no-op
    in practice — kept for Isaac Lab convention and in case a non-zero
    init_state is ever configured.

    Args:
    - env: the running environment
    - env_ids: id's of the envs that we are going to reset
    - asset_cfg: asset handle on our robot
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # env_ids is threaded through so each env spawns inside its own tile.
    valid_poses = terrain.cfg.generate_random_poses(len(env_ids), env_ids=env_ids)

    # Unpack valid_poses (a list of InitialPoseCfgs) into sim-usable tensors.
    posns = torch.stack(list(map(lambda x: torch.tensor(x.pos, device=env.device), valid_poses))).float()
    oris = list(map(lambda x: torch.deg2rad(torch.tensor(x.rot_euler_xyz_deg, device=env.device)), valid_poses))
    oris = torch.stack([math_utils.quat_from_euler_xyz(*ori) for ori in oris]).float()
    lin_vels = torch.stack(list(map(lambda x: torch.tensor(x.lin_vel, device=env.device), valid_poses))).float()
    ang_vels = torch.stack(list(map(lambda x: torch.tensor(x.ang_vel, device=env.device), valid_poses))).float()

    # posns is already world-frame; default_root_state[:3] is zero here so the
    # add is a no-op, but keeps the pattern compatible with any future
    # init_state offset.
    positions = posns + asset.data.default_root_state[env_ids, :3]
    orientations = oris
    lin_vels = lin_vels + asset.data.default_root_state[env_ids, 7:10]
    ang_vels = ang_vels + asset.data.default_root_state[env_ids, 10:13]

    # Write pose/vel into the sim, scoped to just the resetting envs.
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.cat([lin_vels, ang_vels], dim=-1), env_ids=env_ids)


# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingEventsCfg:
    """Event Config Specification"""
    reset_root_state = EventTerm(
        func=reset_root_state,
        mode="reset",
    )


@configclass
class RacingEventsRandomCfg(RacingEventsCfg):
    """Adds robot randomizations"""
    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.4, 0.6),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "make_consistent": False,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "abs",
        },
    )

    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
            "mass_distribution_params": (.01, 0.3),
            "operation": "abs",
        },
    )
