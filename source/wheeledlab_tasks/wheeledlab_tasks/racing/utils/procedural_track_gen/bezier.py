"""
All functions operate on the segment dict format:
    {"p1": coords, "cp1": coords, "cp2": coords, "p2": coords}

This is the shared representation consumed by both the closed-loop
spline builder and the open-chain feature templates.
"""

import numpy as np


def sample_bezier(p1, cp1, cp2, p2, t: float) -> tuple[float, float]:
    """Coordinate of cubic Bezier segment at parameter t (Bernstein form)."""
    mt = 1.0 - t
    x = mt**3*p1[0] + 3*mt**2*t*cp1[0] + 3*mt*t**2*cp2[0] + t**3*p2[0]
    y = mt**3*p1[1] + 3*mt**2*t*cp1[1] + 3*mt*t**2*cp2[1] + t**3*p2[1]
    return (x, y)

def to_bezier_centripetal(p0, p1, p2, p3) -> tuple:
    """
    Centripetal Catmull-Rom to cubic Bezier control points.

    Knot interval = sqrt(chord_length) keeps tangent magnitude proportional
    to distance between points.

    Args:
        p0, p1, p2, p3: four consecutive points (world space).
                         p1-p2 is the segment drawn; p0 and p3 give context.
    Returns:
        (cp1, cp2): outgoing handle from p1, incoming handle into p2.
    """
    pts = [np.array(p) for p in [p0, p1, p2, p3]]

    # knot intervals + avoid diffs by 0
    d1 = max(np.linalg.norm(pts[1] - pts[0])**0.5, 1e-5)
    d2 = max(np.linalg.norm(pts[2] - pts[1])**0.5, 1e-5)
    d3 = max(np.linalg.norm(pts[3] - pts[2])**0.5, 1e-5)

    # tangent calculations, weighted sum of incoming tangent, outgoing, and thru
    m1 = (pts[1]-pts[0])/d1 - (pts[2]-pts[0])/(d1+d2) + (pts[2]-pts[1])/d2
    m2 = (pts[2]-pts[1])/d2 - (pts[3]-pts[1])/(d2+d3) + (pts[3]-pts[2])/d3

    cp1 = pts[1] + (m1 * d2 / 3.0) # bezier control point conversion formula
    cp2 = pts[2] - (m2 * d2 / 3.0)
    return tuple(cp1), tuple(cp2)


def build_spline(sorted_pts) -> list[dict]:
    """
    Build a closed cubic Bezier spline from sorted control points.

    Returns list of segment dicts:
        {"p1", "cp1", "cp2", "p2"}
    """
    n = len(sorted_pts)
    segments = []
    for i in range(n):
        p0 = sorted_pts[(i - 1) % n]
        p1 = sorted_pts[i]
        p2 = sorted_pts[(i + 1) % n]
        p3 = sorted_pts[(i + 2) % n]
        cp1, cp2 = to_bezier_centripetal(p0, p1, p2, p3)
        segments.append({"p1": p1, "p2": p2, "cp1": cp1, "cp2": cp2})
    return segments


def build_polylines(segments, steps=40) -> list[tuple[float, float]]:
    """
    Sample each segment [steps] times to build polyline of (x, y) points.

    Appends the final endpoint so the last segment is fully drawn.
    """
    pts = []
    for seg in segments:
        for i in range(steps):
            t = i / steps
            pts.append(sample_bezier(
                seg["p1"], seg["cp1"], seg["cp2"], seg["p2"], t))
    if segments:
        pts.append(segments[-1]["p2"])
    return pts


def make_segment(p1, cp1, cp2, p2) -> dict:
    """Helper: build a segment dict from four points (tuple)."""
    return {
        "p1":  tuple(p1),
        "cp1": tuple(cp1),
        "cp2": tuple(cp2),
        "p2":  tuple(p2),
    }
