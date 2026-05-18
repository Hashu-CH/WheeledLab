"""
Reward functions and RewardsCfg for the Racing Task.

Notes:

- We rely on the env_id == tile_id invariant set up in
  events.py so each env's reward reads from its own cached polyline.
"""

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..config import CONFIG

_RW = CONFIG["rewards"]
_PPO_ALG = CONFIG.get("ppo", {}).get("algorithm", {})


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

def within_cones_mask(env) -> torch.Tensor:
    """(num_envs,) bool — True when the car is within the cone-defined track boundary.

    Asymmetric on tight turns: on a CCW turn the left side's clamped half-width
    is smaller than the right's, so the same `d_signed` cutoff would punish the
    car for being on the outer side even when the outer cone is still far. The
    bounds here are the per-arc-length cone offsets stored in the TrackCache.
    """
    terrain = env.scene.terrain
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pos = mdp.root_pos_w(env)[..., :2]
    d_signed, _, (left_hw, right_hw), _, _ = terrain.project_to_centerline(root_pos, env_ids)
    return (d_signed <= left_hw) & (d_signed >= -right_hw)


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
    # Φ(s) = total - dist(s) (≥ 0, grows toward goal). Equivalent to Φ = -dist up
    # to a constant — PBRS-invariant — but makes per-step idling reward
    # -(1-γ)·(total - dist) ≤ 0 instead of +(1-γ)·dist ≥ 0, so the critic can't
    # learn a positive-value plateau over stationary states.
    _, curr_dist, prev_dist = compute_progress_step(env)
    total = env.scene.terrain._total_lengths_t[:env.num_envs].clamp_min(1e-6)
    wrap = (prev_dist - curr_dist) > 0.5 * total
    pbrs = prev_dist - gamma * curr_dist - (1.0 - gamma) * total
    return torch.where(wrap, prev_dist, pbrs)


def time_step_penalty(env):
    """Encourage fast termination via goal-reaching"""
    return torch.ones(env.num_envs, device=env.device)


def outside_cones_penalty(env):
    """Per-step cost when the car crosses outside the cone barriers.

    Returns 1.0 when outside the track boundary, 0.0 when inside.
    Sign comes from the (negative) weight in RacingRewardsCfg.
    """
    return (~within_cones_mask(env)).float()


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

    outside_cones_pen = RewTerm(
        func=outside_cones_penalty,
        weight=float(_RW["off_track_pen_weight"]),
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
