"""
Reward functions and RewardsCfg for the Racing Task.

Notes:

- The reward manager invokes each function per step withs and composes them as 
  sum(weight_k * term_k). We rely on the env_id == tile_id invariant set up in 
  events.py so each env's reward reads from its own cached polyline.
- All track-aware rewards (traversable, tangential_speed, cross_track_penalty)
  go through _project() so the nearest-segment projection runs once per step
  instead of N times. Add any new track-relative term through the same helper.
- "On-track" is defined geometrically as |d_signed| < track_width / 2, not as
  a pixel lookup against a rasterised hashmap. The geometric form is smoother
  at the boundary and keeps all reward logic against a single source of truth
  (the polyline cache).
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..config import CONFIG

_RW = CONFIG["rewards"]


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
    return env.scene.terrain.project_to_centerline(poses_xy, env_ids)


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

def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]


# ---------------------------------------------------------------------------
# Reward Terms
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


def chassis_off_track_penalty(env, k: float = 5.0, car_half_length_m: float = 0.21):
    """Penalize any chassis corner leaving the track boundary.

    pen is sum of squared overshot distances off track.

    Corner positions are computed from root pose + yaw rotation applied to the
    four (±half_length, ±half_width) offsets in the chassis local frame.
    half_width is derived from terrain.cfg.car_width_m.
    """

    # this is kinda like _project for 4 points on the mushr
    N = env.num_envs
    device = env.device

    # extract yaw components from quaternion
    pos_xy = mdp.root_pos_w(env)[..., :2]
    quat = mdp.root_quat_w(env)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)   # (N,)


    hl = car_half_length_m
    hw = env.scene.terrain.cfg.car_width_m * 0.5

    # 4 corners in chassis-local frame
    lx = torch.tensor([ hl,  hl, -hl, -hl], device=device)
    ly = torch.tensor([ hw, -hw,  hw, -hw], device=device)

    # expand rotation matrix of yaw comp to convert chassic frame to world frame
    wx = cos_y.unsqueeze(0) * lx.unsqueeze(1) - sin_y.unsqueeze(0) * ly.unsqueeze(1)
    wy = sin_y.unsqueeze(0) * lx.unsqueeze(1) + cos_y.unsqueeze(0) * ly.unsqueeze(1)

    # convert chassic local - world axis coordinates to pure world coordinates
    corners = torch.stack([
        # (4, N), (4,N) =? (4,n,2) then rehsape to (4n, 2)
        pos_xy[:, 0].unsqueeze(0) + wx, 
        pos_xy[:, 1].unsqueeze(0) + wy,
    ], dim=-1).reshape(-1, 2)

    # project to centerline track cache 
    env_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(4, -1).reshape(-1)
    d_signed, _, track_width = env.scene.terrain.project_to_centerline(corners, env_ids)

    # diff calc - how far past track_half_width (4, N)
    overshoot = (d_signed.abs() - track_width * 0.5).clamp_min(0.0).reshape(4, N)

    return k * overshoot.pow(2).sum(dim=0)


# ---------------------------------------------------------------------------
# Reward manager config
# ---------------------------------------------------------------------------
@configclass
class RacingRewardsCfg:
    vel_rew = RewTerm(func=tangential_speed, weight=float(_RW["vel_rew_weight"]))
    chassis_off_track_pen = RewTerm(
        func=chassis_off_track_penalty,
        weight=float(_RW["chassis_off_track_pen_weight"]),
        params={
            "k": float(_RW["chassis_off_track_k"]),
            "car_half_length_m": float(_RW["car_half_length_m"]),
        },
    )
    low_speed_pen = RewTerm(
        func=low_speed_penalty,
        weight=float(_RW["low_speed_weight"]),
        params={
            "low_speed_thresh": float(_RW["low_speed_thresh"]),
        },
    )

    # Large termination penalty
    out_of_tile_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["out_of_tile_pen_weight"]),
        params={"term_keys": "out_range"},
    )
