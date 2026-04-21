import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..utils import TraversabilityHashmapUtil


def traversable_reward(env):
    poses = mdp.root_pos_w(env)[..., :2]
    traversability = TraversabilityHashmapUtil().get_traversability(poses)
    return torch.where(traversability, 1., -1.)


def low_speed_penalty(env, low_speed_thresh: float = 0.3):
    lin_speed = torch.norm(mdp.base_lin_vel(env), dim=-1)
    return torch.where(lin_speed < low_speed_thresh, 1., 0.)


def forward_vel(env):
    return mdp.base_lin_vel(env)[:, 0]


@configclass
class RacingRewardsCfg:
    traversablility = RewTerm(func=traversable_reward, weight=5.)
    vel_rew = RewTerm(func=forward_vel, weight=7.)
    vel_pen = RewTerm(func=low_speed_penalty, weight=-4.)
