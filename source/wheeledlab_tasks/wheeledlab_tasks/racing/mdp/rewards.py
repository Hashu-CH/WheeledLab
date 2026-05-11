"""
Reward functions and RewardsCfg for the Racing Task.

Notes:

- We rely on the env_id == tile_id invariant set up in
  events.py so each env's reward reads from its own cached polyline.
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..config import CONFIG

_RW = CONFIG["rewards"]
_GOALS = CONFIG.get("goals", {})
_PPO_ALG = CONFIG.get("ppo", {}).get("algorithm", {})

# Cached SceneEntityCfg for wheel bodies
_WHEEL_BODY_CFG: SceneEntityCfg | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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

def on_track_mask(env, off_track_wheel_threshold):
    wheel_xy = _wheel_xy_w(env)
    off_mask = env.scene.terrain.wheels_off_track(wheel_xy)
    on_track_enough = off_mask.sum(dim=-1) < off_track_wheel_threshold
    return on_track_enough


# ---------------------------------------------------------------------------
# Reward Terms
# ---------------------------------------------------------------------------
def progress_reward(env, gamma: float = 0.99):
    """Potential-based shaping (PBRS) on track progress.

    Args:
    - env: the running environment
    - gamma: discount factor; should match PPO's gamma for the PBRS invariance
      guarantee to hold.
    """
    _, curr_dist, prev_dist = compute_progress_step(env)
    total = env.scene.terrain._total_lengths_t.clamp_min(1e-6)
    wrap = (prev_dist - curr_dist) > 0.5 * total
    pbrs = prev_dist - gamma * curr_dist
    return torch.where(wrap, prev_dist, pbrs)


def time_step_penalty(env):
    """Encourage fast termination via goal-reaching"""
    return torch.ones(env.num_envs, device=env.device)


def off_track_penalty(env, off_track_wheel_threshold: int = 1):
    """Per-step cost while too many wheels are off track.

    Returns a positive value (1 when off, 0 when on); sign comes from the
    (negative) weight in RacingRewardsCfg.
    """
    on_track = on_track_mask(env, off_track_wheel_threshold)
    return (~on_track).float()


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
            "gamma": float(_PPO_ALG.get("gamma", 0.99)),
        },
    )
    time_step_pen = RewTerm(
        func=time_step_penalty,
        weight=float(_RW["time_step_pen_weight"]),
    )

    off_track_pen = RewTerm(
        func=off_track_penalty,
        weight=float(_RW["off_track_pen_weight"]),
        params={
            "off_track_wheel_threshold": int(_GOALS.get("off_track_wheel_threshold", 3)),
        },
    )
  
    # Termination reward for reaching finish lap
    goal_reached_rew = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["goal_reached_rew_weight"]),
        params={"term_keys": "goal_reached"},
    )

    # Termination penalty for leaving tile
    out_of_tile_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        weight=float(_RW["out_of_tile_pen_weight"]),
        params={"term_keys": "out_range"}, 
    )
