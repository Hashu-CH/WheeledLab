"""
Renders a matplotlib figure per tile showing the cached polyline and tangent
arrows. Call from a debug script after the track_cache has been built.
"""
import numpy as np


def visualize_track_cache(
    track_cache, spacing, env_size,
    tile_indices=(0, 1, 2, 3),
):
    """Render polyline + tangent arrows for a few tiles.

    The tile bounds from track_cache.tile_cell_bounds define the local axes
    so each tile renders in its own (env_num_cols x env_num_rows) cell-space
    frame. No rasterised background — reward correctness doesn't depend on
    it, and the hashmap is no longer persisted.

    Args:
    - track_cache:  TrackCache (must be populated).
    - spacing:      (row_spacing, col_spacing) from the terrain cfg.
    - env_size:     (env_num_rows, env_num_cols) cells per tile.
    - tile_indices: which tile indices to render.
    """
    import matplotlib.pyplot as plt  # lazy import: training runs shouldn't pay for matplotlib

    row_spacing, col_spacing = spacing
    env_num_rows, env_num_cols = env_size

    fig, axes = plt.subplots(1, len(tile_indices), figsize=(4 * len(tile_indices), 4))
    if len(tile_indices) == 1:
        axes = [axes]

    for ax, k in zip(axes, tile_indices):
        # Inverse of the world-coord transform in generated_colored_track_plane.
        # Polyline world coords -> tile-local cell coords for display.
        poly_w = track_cache.polylines_w[k]
        valid_pts = ~np.isnan(poly_w).any(axis=-1)
        poly_w = poly_w[valid_pts]
        # num_cols = total-map cells; we just need to map back to the tile frame.
        # cell_x = world_x / row_spacing + num_cols/2 - c0 -- but num_cols cancels
        # against the tile origin offset, so we can express this using only the
        # tile-local quantity: world_x - tile_origin_x in the same units.
        tile_origin = track_cache.tile_origins_w[k]
        poly_cell_x = (poly_w[:, 0] - tile_origin[0]) / row_spacing + env_num_cols / 2.0
        poly_cell_y = (poly_w[:, 1] - tile_origin[1]) / col_spacing + env_num_rows / 2.0
        ax.plot(poly_cell_x, poly_cell_y, color="red", linewidth=1.0)

        # Tangent arrows sampled every 10th vertex.
        if len(poly_cell_x) > 1:
            # Per-segment validity (both endpoints valid); poly_cell_x has V
            # valid points, so tang must have V - 1 valid segments.
            tang = track_cache.tangents_w[k][track_cache.segment_valid[k]]
            stride = max(1, len(poly_cell_x) // 10)
            ax.quiver(
                poly_cell_x[:-1:stride], poly_cell_y[:-1:stride],
                tang[::stride, 0], tang[::stride, 1],
                color="blue", scale=20, width=0.005,
            )

        ax.set_title(f"tile {k}  width={track_cache.track_widths_m[k]:.2f}m")
        ax.set_xlim(0, env_num_cols)
        ax.set_ylim(0, env_num_rows)
        ax.set_aspect("equal")

    fig.tight_layout()
    return fig
