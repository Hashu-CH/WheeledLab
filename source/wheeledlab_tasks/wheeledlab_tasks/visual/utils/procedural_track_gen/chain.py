"""
Chain builder — sampling, intersection checking, and feature chaining.

Draws random parameters, checks candidates against
the existing track polyline, and assembles the final Bezier chain.
"""

import numpy as np
from .bezier import build_polylines
from .features import FEATURES, FEATURE_ORDER


# Intersection checks
def edges_cross(a0, a1, b0, b1):
    """True if line segment a0->a1 properly crosses b0->b1."""
    d1 = a1 - a0
    d2 = b1 - b0

    # detemrinant of x y \\ x y = 0 iff parallel
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-10:
        return False
    db = b0 - a0
    t = (db[0] * d2[1] - db[1] * d2[0]) / denom
    u = (db[0] * d1[1] - db[1] * d1[0]) / denom
    return 0 < t < 1 and 0 < u < 1


def polylines_intersect(new_pts, existing_pts, skip_last=3):
    """
    Check whether polyline new_pts crosses existing_pts.

    skip_last : ignore the last N edges of the existing polyline
                to avoid false positives at the C1 junction.
    """
    n_exist = len(existing_pts) - 1 - skip_last
    if n_exist <= 0:
        return False
    
    # sad brute force but idk how else to do
    for i in range(len(new_pts) - 1):
        a0, a1 = new_pts[i], new_pts[i + 1]
        for j in range(n_exist):
            if edges_cross(a0, a1, existing_pts[j], existing_pts[j + 1]):
                return True
    return False

# TRACK HYPERPARAMETERS THAT MAY NEED HAND-TUNING FOR GOOD CHAINS
def draw_random(feat_name, intensity):
    """Pure random parameter draw — no validation."""
    if feat_name == "straight":
        return {}
    
    sign = np.random.choice([-1, 1])
    max_angle = intensity * np.pi * 0.6

    if feat_name in ("curve", "s_curve"):
        angle = sign * np.random.uniform(max_angle * 0.4, max_angle)
        return {"angle": angle}

    if feat_name == "varying_curve":
        angle = sign * np.random.uniform(max_angle * 0.4, max_angle)
        tightening = np.random.uniform(0.3, 0.9) * np.random.choice([-1, 1])
        return {"angle": angle, "tightening": tightening}

    # chicane, hairpin
    return {"sign": sign}

# basically straight (fall back for bad generations)
SAFE_PARAMS = {
    "straight": {}, # no extra params
    "curve":    {"angle": 0.05},
    "s_curve":  {"angle": 0.05},
    "varying_curve": {"angle": 0.05, "tightening": 0.0},
    "chicane":  {"sign": 1},
    "hairpin":  {"sign": 1},
}


def sample_params(feat_name, intensity, pos, tan, seg_len,
                   track_points, max_retries=8):
    """
    draws random samples and evaluates intersections to either reject 
    or generate near straight line for chaining

    Returns (params, segs, exit_pos, exit_tan).
    """
    feat_fn = FEATURES[feat_name]["fn"]

    for _ in range(max_retries):
        params = draw_random(feat_name, intensity)
        segs, new_pos, new_tan = feat_fn(pos, tan, seg_len, intensity, **params)
        new_pts = np.array(build_polylines(segs))
        if not polylines_intersect(new_pts, track_points):
            return params, segs, new_pos, new_tan

    # all failed so fall back on safety
    segs, new_pos, new_tan = feat_fn(pos, tan, seg_len, intensity, **SAFE_PARAMS[feat_name])
    return params, segs, new_pos, new_tan


def build_chain(
    n_segments: int,
    segment_length: float,
    difficulty: float,
    cols,
    rows,
    intensity_range: tuple[float, float],
    start_pos: np.ndarray = None,
    start_tan: np.ndarray = None,
) -> tuple[list[dict], list[dict]]:
    """
    concatenate random (unlocked) bezier features into tangent matched chain

    Returns
    all_segments : list[dict]  — feed directly into build_polylines
    feature_log  : list[dict]  — for curriculum tracking / reward shaping
    """
    available = [f for f in FEATURE_ORDER if FEATURES[f]["unlock"] <= difficulty]

    pos = start_pos if start_pos is not None else np.array([0.0, 0.0])
    tan = start_tan if start_tan is not None else np.array([1.0, 0.0]) * segment_length

    all_segments = []
    feature_log = []
    track_points = np.empty((0, 2))

    for _ in range(n_segments):
        feat_name = np.random.choice(available)
        intensity = np.random.uniform(*intensity_range)
        seg_len = segment_length * np.random.uniform(0.7, 1.3) # another thing to test

        _, segs, pos, tan = sample_params(
            feat_name, intensity, pos, tan, seg_len, track_points,
        )

        # STOP loop if new points would go outside of environment
        new_pts = np.array(build_polylines(segs))
        if not np.all((0 <= new_pts[:, 0]) & (new_pts[:, 0] < cols)) or \
           not np.all((0 <= new_pts[:, 1]) & (new_pts[:, 1] < rows)):
            break
        if len(track_points) > 0:
            new_pts = new_pts[1:]  # skip shared end/start point
        
        track_points = np.vstack([track_points, new_pts]) if len(track_points) > 0 else new_pts

        all_segments.extend(segs)
        feature_log.append({"feature": feat_name, "intensity": round(intensity, 3)})

    return all_segments, feature_log
