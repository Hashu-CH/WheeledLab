"""
Reward functions and RewardsCfg for the Racing Task.

Notes:

- The reward manager invokes each function per step withs and composes them as
  sum(weight_k * term_k). We rely on the env_id == tile_id invariant set up in
  events.py so each env's reward reads from its own cached polyline.
- All track-aware rewards (traversable, tangential_speed, cross_track_penalty)
  go through _project() so the nearest-segment projection runs once per step
  instead of N times. Add any new track-relative term through the same helper.
- The newer goal-oriented terms (progress_reward, time_step_penalty,
  goal_reached) consume compute_progress_step(env), a per-step memoized
  result shared with the goal_reached termination so the polyline projection
  + state advance happens exactly once per env step regardless of who reads
  first.
- Two notions of "on-track" coexist: the geometric centerline distance
  (|d_signed| < track_width / 2) used by the legacy terms, and the rendered
  rasterised traversability grid queried per-wheel by progress_reward. The
  raster grid is the source of truth for what the camera sees, so it gates
  progress so the agent can't shortcut through unreachable cells.
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..config import CONFIG

_RW = CONFIG["rewards"]
_GOALS = CONFIG.get("goals", {})

# Cached SceneEntityCfg for wheel bodies
_WHEEL_BODY_CFG: SceneEntityCfg | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _project(env):
    """Single nearest-segment projection shared by all track-aware reward terms.

    Defers to RacingTerrainImporter.project_to_centerline, which gathers each
    env's polyline + tangents via the env_id == tile_id invariant.

    Args:
    - env: the running environment

    Returns:
    - d_signed:    (num_envs,) signed perpendicular distance to the nearest
                   segment. Sign follows n_hat = [-t_y, t_x].
    - tangent_xy:  (num_envs, 2) unit tangent at the nearest segment.
    - track_width: (num_envs,) world-meter track width for each env's tile.
    """
    poses_xy = mdp.root_pos_w(env)[..., :2] # world frame
    env_ids = torch.arange(env.num_envs, device=poses_xy.device, dtype=torch.long)
    d_signed, tangent_xy, track_width, _, _ = env.scene.terrain.project_to_centerline(poses_xy, env_ids)
    return d_signed, tangent_xy, track_width


def _wheel_body_cfg(env) -> SceneEntityCfg:
    """Resolve and cache the SceneEntityCfg covering the four wheel bodies."""
    global _WHEEL_BODY_CFG
    if _WHEEL_BODY_CFG is None:
        # resolve a single time, scene cfg entity will never change
        cfg = SceneEntityCfg("robot", body_names=".*wheel_.*link")
        cfg.resolve(env.scene)
        _WHEEL_BODY_CFG = cfg
    return _WHEEL_BODY_CFG


def _wheel_xy_w(env) -> torch.Tensor:
    """(num_envs, num_wheels, 2) world-frame wheel positions."""
    cfg = _wheel_body_cfg(env)
    asset = env.scene[cfg.name]
    return asset.data.body_pos_w[:, cfg.body_ids, :2]


def compute_progress_step(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-env-step progress update.

    Progress_reward and goal_reached in terminations pull from here. For each
    env step, we cache result to avoid unnecessary computation. 

    Also advances state on terrain importer based on mdp attributes.

    Returns:
    - delta: (num_envs,) prev_dist - curr_dist, in meters.
    - curr_dist: (num_envs,) post-step distance to finish, in [0, total).
    - prev_dist: (num_envs,) pre-update distance — used by termination to
      detect a wrap-around (sudden large drop signaling overshoot).
    """
    counter = int(env.common_step_counter) #seq num
    cache = getattr(env, "_racing_progress_cache", None)
    if cache is not None and cache[0] == counter:
        return cache[1]

    terrain = env.scene.terrain
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pos = mdp.root_pos_w(env)[..., :2]
    prev_dist = terrain.prev_dist_to_finish[env_ids].clone()
    delta = terrain.update_progress(env_ids, root_pos)
    curr_dist = terrain.prev_dist_to_finish[env_ids]
    result = (delta, curr_dist, prev_dist)
    env._racing_progress_cache = (counter, result) # cache result
    return result


# ---------------------------------------------------------------------------
# Old Reward Terms Inherited from Visual Task
# ---------------------------------------------------------------------------
def traversable_reward(env):
    d_signed, _, track_width = _project(env)
    on_track = d_signed.abs() < track_width * 0.5
    return torch.where(on_track, 1., -1.)

def low_speed_penalty(env, low_speed_thresh: float = 0.3):
    lin_speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    return torch.where(lin_speed < low_speed_thresh, 1., 0.)


# ---------------------------------------------------------------------------
# Goal agnostic Reward Terms
# ---------------------------------------------------------------------------
def tangential_speed(env):
    """Track Tangent Velocity

    Uses |v| * cos(theta) where theta is the angle between world-frame
    velocity and the track tangent at the nearest segment. Cos gives 
    negative weight if car velocity is opposite direction.
    
    Note: this is not body alignment. Drifting in the correct direction would 
    be rewarded normally.

    Args:
    - env: the running environment
    """
    _, tangent, _ = _project(env)
    v_xy = mdp.root_lin_vel_w(env)[..., :2]
    return (v_xy * tangent).sum(dim=-1)


# TODO yet another hyper param
def cross_track_penalty(env, k_in: float = 1.0, k_out: float = 10.0):
    """Non-positive penalty shaped by distance from the centerline.

    Shallow quadratic (-k_in * d^2) everywhere, plus a steeper quadratic
    (-k_out * max(|d| - band, 0)^2) once the car leaves the tolerance band.

    band = (track_width - car_width) / 2. 

    Spiritually, in-band is light since racing curves aren't purely centerlines.
    Out of band would be our version of hitting a boundary

    Args:
    - env: the running environment
    - k_in:  in-band quadratic coefficient
    - k_out: out-of-band quadratic coefficient
    """
    d_signed, _, track_width = _project(env)
    car_width = env.scene.terrain.cfg.car_width_m
    d = d_signed.abs()
    band = ((track_width - car_width) * 0.5).clamp_min(0.0)
    extra = (d - band).clamp_min(0.0)
    return -(k_in * d * d + k_out * extra * extra)


# ---------------------------------------------------------------------------
# Goal-Oriented Reward Terms
# ---------------------------------------------------------------------------
def progress_reward(env, max_step_m: float = 2.0, off_track_wheel_threshold: int = 3):
    """Per-step closing of distance to the finish line.

    delta = prev_dist - curr_dist; positive when the car gets closer.
    Reward is gated if car wheels (with threshold) are off track. 
    Clamps [+/- max step] to bound the car overshooting the finish.

    Args:
    - env: the running environment
    - max_step_m: clamp magnitude on the per-step delta in meters
    - off_track_wheel_threshold: if this many wheels are off-track, gate to 0
    """
    delta, _, _ = compute_progress_step(env)
    delta = delta.clamp(-max_step_m, max_step_m)

    wheel_xy = _wheel_xy_w(env)
    off_mask = env.scene.terrain.wheels_off_track(wheel_xy)
    on_track_enough = off_mask.sum(dim=-1) < off_track_wheel_threshold
    return torch.where(on_track_enough, delta, torch.zeros_like(delta))


def time_step_penalty(env):
    """Encourage fast termination via goal-reaching"""
    return torch.ones(env.num_envs, device=env.device)


# ---------------------------------------------------------------------------
# Reward manager config
# ---------------------------------------------------------------------------
@configclass
class RacingRewardsCfg:
    # Goal oriented rewards and penalties
    progress_rew = RewTerm(
        func=progress_reward,
        weight=float(_RW["progress_rew_weight"]),
        params={
            "max_step_m": float(_GOALS.get("progress_clamp_m", 2.0)),
            "off_track_wheel_threshold": int(_GOALS.get("off_track_wheel_threshold", 3)),
        },
    )
    time_step_pen = RewTerm(
        func=time_step_penalty,
        weight=float(_RW["time_step_pen_weight"]),
    )

    # Termination rewards and penalties
    goal_reached_rew = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["goal_reached_rew_weight"]),
        params={"term_keys": "goal_reached"},
    )
    out_of_tile_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["out_of_tile_pen_weight"]),
        params={"term_keys": "out_range"}, 
    )
