"""
Contains termination helpers for the racing task.

Notes:

- Terminations fire the reset states in the events file
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Termination Helpers
# ---------------------------------------------------------------------------
def out_of_map(env):
    """Defines if a set of poses is outside of their respective envs"""
    poses = mdp.root_pos_w(env)[..., :2]
    terrain = env.scene[SceneEntityCfg("terrain").name]
    width = terrain.cfg.width
    height = terrain.cfg.height
    
    # reminder, env are in local coordinates now, allows us to check width / 2 
    x_out_range = torch.logical_or(poses[..., 0] > width / 2, poses[..., 0] < -width / 2)
    y_out_range = torch.logical_or(poses[..., 1] > height / 2, poses[..., 1] < -height / 2)
    return torch.logical_or(x_out_range, y_out_range)



# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingTerminationsCfg:
    # time fields live in racing_env_cfg in the post init
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_range = DoneTerm(func=out_of_map)
