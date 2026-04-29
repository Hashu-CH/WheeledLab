"""
Contains termination helpers for the racing task.

Notes:

- Terminations fire the reset states in the events file.
- Env_spacing is 0: every env's origin is world (0,0,0), but each env's
  assigned tile sits at a different world-frame offset inside the shared
  plane. Checking against the global map width/height would overshoot —
  a car can leave its own tile while still inside the big rectangle. The
  tile-local check below uses track_cache.tile_origins_w + tile_extent_m
  via the terrain's out_of_tile method.
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..config import CONFIG
from .rewards import compute_progress_step, on_track_mask

_GOALS = CONFIG.get("goals", {})


# ---------------------------------------------------------------------------
# Termination Helpers
# ---------------------------------------------------------------------------
def out_of_tile(env):
    """True when a car leaves its own tile's world-frame bounding box."""
    poses_xy_w = mdp.root_pos_w(env)[..., :2]
    env_ids = torch.arange(env.num_envs, device=poses_xy_w.device, dtype=torch.long)
    return env.scene.terrain.out_of_tile(poses_xy_w, env_ids)


def goal_reached(env, eps_m: float = 0.5, off_track_wheel_threshold: int = 3):
    """Fires when the car has arrived at (or just crossed) the finish line.

    Arg:
    - eps_m: car reached goal if within eps_m meters of finish
    - off_track_wheel_threshold: gate fires only if fewer than this many
      wheels are off-track at the crossing step

    Notes
    - Normal arrival: curr_dist < eps_m.
    - Overshoot wrap: prev_dist - curr_dist > total_length / 2, if within
      one step and the agent dropped more than half the track distance, it
      must have passed finish line
    """
    _, curr_dist, prev_dist = compute_progress_step(env)
    terrain = env.scene.terrain
    total = terrain._total_lengths_t.clamp_min(1e-6)
    arrived = curr_dist < eps_m
    overshoot = (prev_dist - curr_dist) > 0.5 * total
    on_track = on_track_mask(env, off_track_wheel_threshold)
    return (arrived | overshoot) & on_track


# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingTerminationsCfg:
    # time fields live in racing_env_cfg in the post init
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_range = DoneTerm(func=out_of_tile)
    goal_reached = DoneTerm(
        func=goal_reached,
        params={
            "eps_m": float(_GOALS.get("goal_reached_eps_m", 0.5)),
            "off_track_wheel_threshold": int(_GOALS.get("off_track_wheel_threshold", 3)),
        },
    )
