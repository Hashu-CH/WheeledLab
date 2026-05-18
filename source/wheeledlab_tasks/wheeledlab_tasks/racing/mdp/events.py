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
import numpy as np

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
_TER = CONFIG["terrain"]


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
# Ground texture randomization
# ---------------------------------------------------------------------------

def randomize_ground_texture(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    plane_prim_path: str = "/World/ground_plane",
    grid_resolution_m: float = 1.0,   # unused; kept so existing EventTerm params still parse
    noise_sigma: float = 1.0,
    brightness_range: tuple = (0.1, 0.5),
    update_interval_steps: int = 256,
    tex_resolution: int = 128,
):
    """Write a Gaussian-blurred noise PNG and update the ground plane's UV texture.

    Each call generates a fresh `tex_resolution × tex_resolution` noise image and
    writes it to one of two alternating file paths, then updates the USD texture
    attribute so Isaac Sim reloads the image.  The alternating-path trick forces
    USD to treat the new path as a genuinely changed attribute (cache-busting).

    noise_sigma is in texture pixels: 0 = random per-pixel noise ("pixelated"),
    ~3 = smooth blobs.
    """
    last = getattr(env, "_last_ground_tex_step", -update_interval_steps)
    if int(env.common_step_counter) - last < update_interval_steps:
        return
    env._last_ground_tex_step = int(env.common_step_counter)

    try:
        from PIL import Image
        from pxr import UsdShade, Sdf
        import omni.usd
        import scipy.ndimage
    except ImportError:
        return

    from ..track.generator import GROUND_TEX_PATH, GROUND_TEX_PATH_ALT, GROUND_TEX_READER_PATH

    # Toggle between two file paths to force USD texture cache invalidation.
    toggle = getattr(env, "_ground_tex_toggle", 0)
    write_path = GROUND_TEX_PATH if toggle == 0 else GROUND_TEX_PATH_ALT
    env._ground_tex_toggle = 1 - toggle

    lo, hi = float(brightness_range[0]), float(brightness_range[1])
    noise = np.random.uniform(lo, hi, size=(tex_resolution, tex_resolution)).astype(np.float32)
    if noise_sigma > 0:
        noise = scipy.ndimage.gaussian_filter(noise, sigma=float(noise_sigma))
        noise = np.clip(noise, lo, hi)
    img_arr = (noise * 255).astype(np.uint8)
    Image.fromarray(img_arr, mode="L").convert("RGB").save(write_path)

    stage = omni.usd.get_context().get_stage()
    tex_prim = stage.GetPrimAtPath(GROUND_TEX_READER_PATH)
    if tex_prim.IsValid():
        UsdShade.Shader(tex_prim).GetInput("file").Set(Sdf.AssetPath(write_path))


# Lighting randomization
# ---------------------------------------------------------------------------

def randomize_lighting(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    light_prim_path: str = "/World/light",
    intensity_range: tuple = (1500.0, 6000.0),
    elevation_deg_range: tuple = (10.0, 45.0),
    azimuth_deg_range: tuple = (0.0, 360.0),
    color_r_range: tuple = (0.9, 1.0),
    color_g_range: tuple = (0.7, 1.0),
    color_b_range: tuple = (0.5, 1.0),
):
    """Randomize the scene's distant light each episode to simulate variable
    outdoor morning/afternoon sun conditions (intensity, color temperature,
    and sun elevation/azimuth angle).

    Light is global so we randomize once per reset call regardless of how
    many envs are being reset.
    """
    from pxr import UsdLux, UsdGeom, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath(light_prim_path)
    if not light_prim.IsValid():
        return

    dist_light = UsdLux.DistantLight(light_prim)

    intensity = float(np.random.uniform(*intensity_range))
    dist_light.GetIntensityAttr().Set(intensity)

    r = float(np.random.uniform(*color_r_range))
    g = float(np.random.uniform(*color_g_range))
    b = float(np.random.uniform(*color_b_range))
    dist_light.GetColorAttr().Set(Gf.Vec3f(r, g, b))

    # Sun direction: DistantLight illuminates along its local -Z.
    # Euler XYZ = (elev - 90, 0, azim) tilts the downward default
    # to the correct elevation and rotates it to the desired azimuth.
    elev_deg = float(np.random.uniform(*elevation_deg_range))
    azim_deg = float(np.random.uniform(*azimuth_deg_range))

    xformable = UsdGeom.Xformable(light_prim)
    xformable.ClearXformOpOrder()
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(elev_deg - 90.0, 0.0, azim_deg))


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

    randomize_ground_texture = EventTerm(
        func=randomize_ground_texture,
        mode="reset",
        params={
            "noise_sigma":           float(_EV.get("ground_noise_sigma", 1.0)),
            "brightness_range":      tuple(_EV.get("ground_brightness_range", [0.1, 0.5])),
            "update_interval_steps": int(_EV.get("ground_texture_update_interval", 256)),
            "tex_resolution":        int(_EV.get("ground_texture_resolution", 128)),
        },
    )
