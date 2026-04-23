"""
Helper utilities that operate on a TrackCache: nearest-segment projection
and polyline arc-length spawn sampling.

Notes:

- Both helpers are used by RacingTerrainImporterCfg methods, which gather
  per-env tiles from the cache and then call through here. This file is the
  pure-math / pure-logic half; the cfg handles state ownership and batching.
- project_nearest_segment drives cross-track + tangential-speed rewards.
  sample_poses_along_polylines drives reset spawn placement.
- The tangent we return is an approximation — the direction of the polyline
  chord, not the analytical Bezier tangent. Sampling more points per bezier
  segment in CurriculumConfig is the cheap way to tighten it.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batched nearest-segment projection (used by reward terms at each step)
# ---------------------------------------------------------------------------
def project_nearest_segment(
    polylines: torch.Tensor,
    tangents: torch.Tensor,
    segment_valid: torch.Tensor,
    poses_xy: torch.Tensor,
):
    """Project each pose onto the nearest valid segment of its polyline.

    N = num_envs (batch). M is the padded polyline length; shorter polylines
    have NaN vertices past their real length, and segment_valid flags which
    of the M-1 segments are built from real (non-NaN) endpoints. Need padding
    for tensors to be same-size for batched argmin to find closest segment.

    Args:
    - polylines: per-env polyline vertices, NaN-padded to length M. (N, M, 2)
    - tangents: precomputed unit tangents per segment — reward doesn't need
      to renormalize at every step.
    - segment_valid: (N, M-1) bool. Which segments of polylines are not padded.
    - poses_xy: current car (x, y) in world meters.

    Returns:
    - d_signed: (N,) signed perpendicular distance. Positive -> left of tangent.
    - nearest_tangent: (N, 2), nearest polyline tangent per env.
    - seg_idx: (N,) index of the winning segment, in [0, M-2].
    """
    # Distance from poses to the start of each segment.
    p0 = polylines[:, :-1, :]
    seg = polylines[:, 1:, :] - p0
    pose = poses_xy.unsqueeze(1)
    to_start = pose - p0  # (N, M-1, 2)

    # foot = p0 + t * seg; (pose - foot) · seg = 0 solves for t.
    seg_len_sq = (seg * seg).sum(dim=-1).clamp_min(1e-12)  # avoid div by 0
    t_param = ((to_start * seg).sum(dim=-1) / seg_len_sq).clamp(0.0, 1.0)

    # Use solved t to find foot of perpendicular and the offset from the pose.
    foot = p0 + t_param.unsqueeze(-1) * seg
    offset = pose - foot
    dist_sq = (offset * offset).sum(dim=-1)

    # Can't do argmin on NaN padded-segment distances. Set to +inf to exclude.
    dist_sq = torch.where(segment_valid, dist_sq, torch.full_like(dist_sq, float("inf")))

    # Gather the winning segment's offset + tangent. One argmin per env.
    seg_idx = dist_sq.argmin(dim=-1)
    gather_idx = seg_idx.view(-1, 1, 1).expand(-1, 1, 2)  # shape matching

    # `tangents[seg_idx]` is the direction of the polyline chord, not an
    # analytical Bezier tangent. Approximation tightening options:
    #   1. Raise steps_per_segment in CurriculumConfig (cheapest).
    #   2. Lerp + renormalise between tangents[seg_idx - 1] and
    #      tangents[seg_idx] using t_param.
    #   3. Cache per-segment Bezier control points and evaluate the
    #      analytical derivative B'(t) at the projected point.
    nearest_tangent = tangents.gather(1, gather_idx).squeeze(1)
    nearest_offset = offset.gather(1, gather_idx).squeeze(1)

    # n_hat = [-t_y, t_x] means positive d_signed = car is on the LEFT of tangent.
    n_hat = torch.stack([-nearest_tangent[:, 1], nearest_tangent[:, 0]], dim=-1)  # perp via rotate
    d_signed = (nearest_offset * n_hat).sum(dim=-1)
    return d_signed, nearest_tangent, seg_idx


# ---------------------------------------------------------------------------
# Polyline arc-length spawn sampling (used by reset_root_state each reset)
# ---------------------------------------------------------------------------
def sample_poses_along_polylines(
    track_cache,
    env_ids,
    car_width_m: float,
    margin_m: float = 0.0,
):
    """Sample WORLD-frame (x, y, yaw) spawn poses along each env's centerline.

    For each id:
      1. Pick a random non-padded polyline segment from that env's tile.
      2. Pick a random t ∈ [0, 1] along that segment; compute the foot in
         world meters (the polyline cache is already world-frame).
      3. Add a random lateral offset perpendicular to the tangent, bounded
         by ±(track_width/2 - car_width/2 - margin_m) so the chassis stays
         on-track at spawn.
      4. Random yaw in [0, 360).

    reset_root_state takes these poses directly (default_root_state is zero
    here since env_spacing=0 and init_state.pos=0).

    Args:
    - track_cache: TrackCache (from wheeledlab_tasks.racing.utils).
    - env_ids: env indices to sample one spawn pose each.
    - car_width_m: chassis width (shrinks the lateral band).
    - margin_m: extra safety margin so car doesn't spawn at the band's edge.
    """
  
    if hasattr(env_ids, 'cpu'): # env ids is a cuda tensor -> convert to cpu for numpy
      env_ids = env_ids.cpu()
    env_ids_np = np.asarray(env_ids, dtype=np.int64)
    env_ids_np = np.asarray(env_ids, dtype=np.int64)
    num_poses = len(env_ids_np)

    polylines = track_cache.polylines_w
    tangents = track_cache.tangents_w
    segment_valid = track_cache.segment_valid
    track_widths = track_cache.track_widths_m

    poses = []
    for k in range(num_poses):
        tile_idx = int(env_ids_np[k])
        valid_indices = np.where(segment_valid[tile_idx])[0]

        if len(valid_indices) == 0:
            # Fall back if track generation produced no valid segments.
            poses.append((0.0, 0.0, float(np.random.uniform(0, 360))))
            continue

        seg_idx = int(np.random.choice(valid_indices))
        t = float(np.random.uniform(0, 1))

        p0 = polylines[tile_idx, seg_idx]
        p1 = polylines[tile_idx, seg_idx + 1]
        foot_w = p0 + t * (p1 - p0)

        # Lateral offset perpendicular to the tangent (n_hat = [-t_y, t_x]).
        tangent = tangents[tile_idx, seg_idx]
        n_hat = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        band = max(
            0.0,
            (float(track_widths[tile_idx]) - car_width_m) * 0.5 - margin_m,
        )
        lateral = float(np.random.uniform(-band, band))

        world_pos = foot_w + lateral * n_hat
        angle = float(np.random.uniform(0, 360))
        poses.append((float(world_pos[0]), float(world_pos[1]), angle))

    return poses
