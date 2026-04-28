"""
Control-point generation for the closed-loop pipeline.

Points are derived from an ellipse with angular and radial jitter,
then sorted clockwise for consistent spline winding.
"""

import math
import numpy as np

def generate_points(
    n: int,
    width: float,
    height: float,
    min_radius: float = 0.35,
    margin: float = 20.0,
    max_jitter: float = 0.40,
) -> list[tuple[float, float]]:
    """
    Jitters rays along evenly spaced angles about an ellipse, then places points 
    within some bounding box and min radius along such rays.

    Args:
        n:          number of control points
        width:      bounding box width in world units
        height:     bounding box height in world units
        min_radius: min radial fraction (0 = centre, 1 = ellipse edge)
        margin:     edge margin so track doesn't touch the border
        max_jitter: angular jitter as fraction of inter-point angle
                    (> ~0.45 risks overlap)

    Returns:
        list of (x, y) world-space coordinates
    """
    cx, cy = width / 2, height / 2
    rx, ry = cx - margin, cy - margin
    jitter = 2 * math.pi / n * max_jitter

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles += np.random.uniform(-jitter, jitter, size=n)
    sampled_radii = np.random.uniform(min_radius, 1.0, size=n)

    # convert polar coordinates back to world axis
    x = cx + np.cos(angles) * rx * sampled_radii
    y = cy + np.sin(angles) * ry * sampled_radii
    return list(zip(x, y))


def sort_clockwise(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort points clockwise by angle from their centroid."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
