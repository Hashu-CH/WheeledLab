"""
Rasterise a polyline (open or closed) onto a boolean grid.
Uses symmetric binary dilation for uniform track width. 
Matches util function within WheeledLab
"""

import math
import numpy as np
from scipy.ndimage import binary_dilation

def rasterise_track(
    track: list[tuple[float, float]],
    rows,
    cols,
    track_width: int,
    closed: bool = True,
) -> np.ndarray:
    """
    Paint a polyline onto a boolean grid, then dilate to desired width with 
    asymmetric kernel.

    Args:
        track:       list of (x, y) world-space points (NORAMLIZED)
        grid_size:   resolution of the longer grid axis
        track_width: width in tiles (larger → easier)
        env_size:    widht height grid output size
        closed:      if True, connect last point back to first

    Returns:
        boolean numpy array — True = traversable
    """
    grid = np.zeros((rows, cols), dtype=bool)
    n = len(track)
    end = n if closed else n - 1

    for i in range(end):
        ax, ay = track[i][0] * cols, track[i][1] * rows
        bx, by = track[(i + 1) % n][0] * cols, track[(i + 1) % n][1] * rows
        steps = max(int(math.hypot(bx - ax, by - ay)) * 2, 1)
        for s in range(steps + 1):
            t = s / steps
            # paint tiles along discrete grid indices
            gx, gy = int(ax + (bx - ax) * t), int(ay + (by - ay) * t)
            if 0 <= gx < cols and 0 <= gy < rows:
                grid[gy, gx] = True

    # asymmetric cross kernel for uniform dilation
    struct = np.array([[0, 1, 0],
                       [0, 1, 1],
                       [0, 0, 0]], dtype=bool)
    iterations = max(1, track_width // 2)
    grid = binary_dilation(grid, structure=struct, iterations=iterations)
    return grid
