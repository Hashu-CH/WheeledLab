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


# ---------------------------------------------------------------------------
# Reward manager config
# ---------------------------------------------------------------------------
@configclass
class RacingRewardsCfg:
    vel_rew = RewTerm(func=tangential_speed, weight=float(_RW["vel_rew_weight"]))
    cross_track_pen = RewTerm(
        func=cross_track_penalty,
        weight=float(_RW["cross_track_pen_weight"]),
        params={
            "k_in": float(_RW["cross_track_k_in"]),
            "k_out": float(_RW["cross_track_k_out"]),
        },
    )
    low_speed_pen = RewTerm(
        func=low_speed_penalty,
        weight=float(_RW["low_speed_weight"]),
        params={
            "low_speed_thresh": float(_RW["low_speed_thresh"]),
        },
    )

    out_of_tile_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["out_of_tile_pen_weight"]),
        params={"term_keys": "out_range"},
    )
