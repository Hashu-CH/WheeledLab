"""
Bezier chain feature templates

Each template takes the current state (position + tangent) and
returns segment dicts in the same format as bezier.build_spline,
plus the exit state for C1-continuous chaining.

Common signature:
    feat_*(p1, tan, length, intensity, **params) -> (segments, exit_pos, exit_tan)
"""

import numpy as np
from .bezier import make_segment

# Helper functions for geometry features
def rotate(v, angle):
    """Rotate a 2D vector by angle (radians)."""
    cos, sin = np.cos(angle), np.sin(angle)
    # never in my life did I think I'd use a rotation matrix again
    return np.array([cos*v[0] - sin*v[1], sin*v[0] + cos*v[1]])

def fwd_perp(tan):
    """Return unit forward and perpendicular (left) from a tangent."""
    fwd = tan / np.linalg.norm(tan) 
    return fwd, np.array([-fwd[1], fwd[0]])


# Feature templates
def feat_straight(p1, tan, length, intensity):
    """ A Straight line """
    fwd, _ = fwd_perp(tan)

    # tbh these don't matter, just to keep output format consistent
    p2 = p1 + fwd * length
    cp1 = p1 + fwd * length/3
    cp2 = p2 - fwd * length/3

    exit_tan = p2 - cp2
    return [make_segment(p1, cp1, cp2, p2)], p2, exit_tan


def feat_curve(p1, tan, length, intensity, angle):
    """
    Smooth arc.

    angle : float (radians, signed) — positive = left, negative = right.
    """
    fwd, _ = fwd_perp(tan)

    cp1 = p1 + fwd * length / 3
    exit_fwd = rotate(fwd, angle)
    mid_fwd = rotate(fwd, angle / 2)
    p2 = p1 + mid_fwd * length
    cp2 = p2 - exit_fwd * length / 3

    return [make_segment(p1, cp1, cp2, p2)], p2, exit_fwd


def feat_varying_curve(p1, tan, length, intensity, angle, tightening):
    """
    Curve with changing radius. Two chained arcs with different curvatures.

    tightening > 0: 
        radius decreases (gentle into sharp)
    tightening < 0: 
        radius increases (sharp into gentle)
    """
    angle_1 = angle * (1 - tightening) / 2
    angle_2 = angle * (1 + tightening) / 2
    segs1, mid, mid_tan = feat_curve(p1, tan, length / 2, intensity, angle_1)
    segs2, end, exit_tan = feat_curve(mid, mid_tan, length / 2, intensity, angle_2)
    return segs1 + segs2, end, exit_tan


def feat_s_curve(p1, tan, length, intensity, angle):
    """
    Two opposite arcs chained together.

    angle : float (radians, signed) — direction of the first arc;
            the second arc uses -angle so the track S-bends.
    """
    segs1, mid, mid_tan = feat_curve(p1, tan, length / 2, intensity, angle)
    segs2, end, exit_tan = feat_curve(mid, mid_tan, length / 2, intensity, -angle)
    return segs1 + segs2, end, exit_tan


def feat_chicane(p1, tan, length, intensity, sign):
    """
    Sharp S with a net lateral shift.  Three segments.

    sign : +1 or -1 — lateral direction of the chicane.
    """
    fwd, perp = fwd_perp(tan)
    offset = sign * intensity * length * 0.50
    third = length / 3

    waypoints = [
        p1,
        p1 + fwd * third           + perp * offset,
        p1 + fwd * 2 * third       + perp * offset * 0.3,
        p1 + fwd * length          + perp * offset * 0.6,
    ]
    segs = []
    prev_dir = fwd
    for i in range(3):
        p1, p2 = waypoints[i], waypoints[i+1]
        seg_dir = p2 - p1
        seg_len = np.linalg.norm(seg_dir)
        if seg_len < 1e-6:
            continue
        seg_fwd = seg_dir / seg_len
        h = seg_len / 3
        cp1 = p1 + prev_dir * h
        cp2 = p2 - seg_fwd * h
        segs.append(make_segment(p1, cp1, cp2, p2))
        prev_dir = seg_fwd

    exit_tan = prev_dir
    return segs, waypoints[-1], exit_tan


def feat_hairpin(p1, tan, length, intensity, sign):
    """
    Tight 180° U-turn from two quarter-circle Bézier arcs.
    k ≈ 0.5523 is the standard quarter-circle approximation constant.

    sign : +1 or -1 — which side the U-turn goes toward.
    """
    fwd, perp = fwd_perp(tan)
    radius = np.interp(intensity, [0, 1], [length * 0.45, length * 0.18])
    k = 0.5523 * radius

    # arc 1: entry → apex
    cp1 = p1 + fwd * k
    apex = p1 + fwd * radius + sign * perp * radius
    cp2 = apex - sign * perp * k

    # arc 2: apex → exit
    exit_pos = p1 + sign * perp * 2 * radius
    exit_fwd = -fwd
    cp1b = apex + sign * perp * k
    cp2b = exit_pos - exit_fwd * k

    segs = [make_segment(p1, cp1, cp2, apex),
            make_segment(apex, cp1b, cp2b, exit_pos)]
    exit_tan = exit_fwd
    return segs, exit_pos, exit_tan


# macros for use by curriculum, named features have assoc unlock and fn
FEATURES = {
    "straight":      {"fn": feat_straight,      "unlock": 0.00},
    "curve":         {"fn": feat_curve,          "unlock": 0.05},
    "s_curve":       {"fn": feat_s_curve,        "unlock": 0.20},
    "varying_curve": {"fn": feat_varying_curve,  "unlock": 0.35},
    "chicane":       {"fn": feat_chicane,         "unlock": 0.50},
    "hairpin":       {"fn": feat_hairpin,         "unlock": 0.65},
}
FEATURE_ORDER = list(FEATURES.keys())
