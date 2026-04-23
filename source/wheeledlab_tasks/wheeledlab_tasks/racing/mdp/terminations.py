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


# ---------------------------------------------------------------------------
# Termination Helpers
# ---------------------------------------------------------------------------
def out_of_tile(env):
    """True when a car leaves its own tile's world-frame bounding box."""
    poses_xy_w = mdp.root_pos_w(env)[..., :2]
    env_ids = torch.arange(env.num_envs, device=poses_xy_w.device, dtype=torch.long)
    return env.scene.terrain.out_of_tile(poses_xy_w, env_ids)


# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingTerminationsCfg:
    # time fields live in racing_env_cfg in the post init
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_range = DoneTerm(func=out_of_tile)
