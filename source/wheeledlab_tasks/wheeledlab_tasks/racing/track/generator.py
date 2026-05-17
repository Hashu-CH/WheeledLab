"""
Terrain/Plane generation and track-geometry cache for the Racing Task.

Notes:

- Track-aware rewards need a geometric view of the terrain — centerline
  polyline, tangents, per-tile widths — which live in the TrackCache
  dataclass below and are populated when the USD is authored.
- The ground is a simple flat grey plane. Track boundaries are defined
  purely by cone positions; the traversability grid is no longer used.
- Coord-frame convention: polylines from curriculum.generate_track are in
  grid-cell units within a tile (x in [0, env_num_cols], y in [0, env_num_rows]).
  We convert to WORLD METERS here so reward-time projection doesn't re-derive
  the transform. All TrackCache arrays are world-frame.
- tile_padding_cells adds a gap (in cells) around each tile so that cones
  from neighbouring tiles don't sit too close together.

Concern:

- Right now, each environment (robot) gets its own track and tile. Which 1, constrains
  compute since we cache track information and can't re-use. 2, will likely be high
  variance.
  We could seed tracks and duplicate them as an easy fix. Most optimally, we allow duplicate
  envs on the same tiles. TODO: It's easy to prevent collisions (collision groups) but
  much harder to make envs invisible to each other in the camera observation.
"""

import os
from dataclasses import dataclass
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from .procedural.curriculum import CurriculumConfig, generate_track
from ..config import CONFIG

_TER = CONFIG["terrain"]

# Absolute paths to cone USDA assets (4 levels up from this file → source/ → wheeledlab_assets)
_CONES_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                 "wheeledlab_assets", "data", "cones")
)
_ORANGE_CONE_USD = os.path.join(_CONES_DIR, "orange_cone.usda")
_BLUE_CONE_USD   = os.path.join(_CONES_DIR, "blue_cone.usda")


# ---------------------------------------------------------------------------
# Per-tile track geometry cache
# ---------------------------------------------------------------------------
@dataclass
class TrackCache:
    """Per-tile track geometry consumed by reward functions at runtime.

    All world-meter arrays already include the per-tile offset.
    Padding convention: see projection utils.

    Naming conventions:
    - num_tiles = num_env_rows * num_env_cols
    - M_max = max polyline length across all tiles (so cachce can be rectangle tensor)
    """
    polylines_w: np.ndarray # (num_tiles, M_max, 2), float32
    tangents_w: np.ndarray # (num_tiles, M_max - 1, 2), unit vectors
    segment_lengths_m: np.ndarray # (num_tiles, M_max - 1), float32, segment world lengths
    cumulative_arc_lengths_m: np.ndarray # (num_tiles, M_max), float32, prefix sum of seg lengths along polyline
    total_lengths_m: np.ndarray # (num_tiles,), float32, last valid cumulative length per polyline
    is_closed: np.ndarray # (num_tiles,), bool, True for closed-loop tracks
    segment_valid: np.ndarray # (num_tiles, M_max - 1), bool
    track_widths_m: np.ndarray # (num_tiles,), meters
    tile_origins_w: np.ndarray # (num_tiles, 2), world-meter tile centers
    tile_extent_m: tuple[float, float] # (x_extent, y_extent) per tile, meters (fixed here)
    tile_cell_bounds: np.ndarray # (num_tiles, 4): row_start, row_end, col_start, col_end


# ---------------------------------------------------------------------------
# Cone placement helpers
# ---------------------------------------------------------------------------

def _compute_cone_positions(
    polyline_w: np.ndarray,       # (M, 2) world-meter centerline
    track_width_m: float,
    cone_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_positions, right_positions) arrays of shape (N, 2).

    Left/right are defined relative to the direction of travel along the
    centerline. Left = 90° CCW from tangent, right = 90° CW.
    Positions are subsampled at approximately cone_spacing_m arc-length intervals.
    """
    if len(polyline_w) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    half = track_width_m / 2.0

    # Arc-length prefix sum along centerline
    diffs = np.diff(polyline_w, axis=0)                         # (M-1, 2)
    seg_lens = np.linalg.norm(diffs, axis=1)                    # (M-1,)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])       # (M,)
    total_len = cum_len[-1]
    if total_len < cone_spacing_m:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Sample positions at regular arc-length intervals
    sample_arcs = np.arange(0.0, total_len, cone_spacing_m)
    sample_pts  = np.stack([
        np.interp(sample_arcs, cum_len, polyline_w[:, 0]),
        np.interp(sample_arcs, cum_len, polyline_w[:, 1]),
    ], axis=1)                                                   # (N, 2)

    # Tangent at each sample via finite difference on the resampled points
    tan = np.diff(sample_pts, axis=0)                           # (N-1, 2)
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan_norm = np.where(tan_norm > 0, tan_norm, 1.0)
    tan = tan / tan_norm                                         # unit tangents
    # Duplicate last tangent so shapes align with sample_pts
    tan = np.concatenate([tan, tan[[-1]]], axis=0)              # (N, 2)

    # Left normal = CCW 90°: (tx, ty) -> (-ty, tx)
    left_n  = np.stack([-tan[:, 1],  tan[:, 0]], axis=1)
    right_n = np.stack([ tan[:, 1], -tan[:, 0]], axis=1)

    left_pos  = sample_pts + half * left_n
    right_pos = sample_pts + half * right_n

    return left_pos.astype(np.float32), right_pos.astype(np.float32)


def _spawn_cones_in_stage(
    stage: Usd.Stage,
    left_positions_per_tile:  list[np.ndarray],   # list of (N_i, 2) float32
    right_positions_per_tile: list[np.ndarray],
) -> None:
    """Add orange (left) and blue (right) cone USD references to the stage."""
    if not os.path.isfile(_ORANGE_CONE_USD):
        import warnings
        warnings.warn(
            f"Orange cone asset not found at {_ORANGE_CONE_USD}. "
            "Skipping cone spawning — run wheeledlab_assets/data/cones/generate_cones.py first."
        )
        return
    if not os.path.isfile(_BLUE_CONE_USD):
        import warnings
        warnings.warn(
            f"Blue cone asset not found at {_BLUE_CONE_USD}. "
            "Skipping cone spawning."
        )
        return

    UsdGeom.Xform.Define(stage, "/World/cones")

    for tile_idx, (left_pos, right_pos) in enumerate(
        zip(left_positions_per_tile, right_positions_per_tile)
    ):
        for ci, (px, py) in enumerate(left_pos):
            prim_path = f"/World/cones/orange_t{tile_idx}_c{ci}"
            xf = UsdGeom.Xform.Define(stage, prim_path)
            xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), 0.0))
            xf.GetPrim().GetReferences().AddReference(_ORANGE_CONE_USD)

        for ci, (px, py) in enumerate(right_pos):
            prim_path = f"/World/cones/blue_t{tile_idx}_c{ci}"
            xf = UsdGeom.Xform.Define(stage, prim_path)
            xf.AddTranslateOp().Set(Gf.Vec3d(float(px), float(py), 0.0))
            xf.GetPrim().GetReferences().AddReference(_BLUE_CONE_USD)


# ---------------------------------------------------------------------------
# Map sizing
# ---------------------------------------------------------------------------
def compute_map_size(num_envs, env_size, tile_padding_cells=0):
    """Map dimensions (in cells) to pack `num_envs` non-overlapping sub-envs in
    a square-ish grid.

    num_tiles == num_envs which lets env_id == tile_id hold.

    Args:
    - num_envs: total parallel envs (must be a power of 2)
    - env_size: (env_num_rows, env_num_cols) — cells per tile (inner track area)
    - tile_padding_cells: extra cells added around each tile to space cones apart
    """
    if num_envs < 1 or (num_envs & (num_envs - 1)) != 0:
        raise ValueError(f"num_envs must be a power of 2, got {num_envs}")
    env_num_rows, env_num_cols = env_size
    padded_rows = env_num_rows + tile_padding_cells
    padded_cols = env_num_cols + tile_padding_cells
    exponent = num_envs.bit_length() - 1
    grid_rows = 1 << (exponent // 2)
    grid_cols = 1 << (exponent - exponent // 2)
    return grid_rows * padded_rows, grid_cols * padded_cols


# ---------------------------------------------------------------------------
# Terrain + track-cache construction
# ---------------------------------------------------------------------------
def generated_colored_track_plane(map_size, spacing, env_size, color_sampling=False, tile_padding_cells=0):
    """Build a flat grey ground-plane mesh plus the per-tile track cache.

    Cones define the track boundaries; no traversability grid is rasterised.
    tile_padding_cells adds a border of empty cells around each tile so that
    cones from neighbouring tiles are separated by at least
    (tile_padding_cells * spacing) metres.

    Args:
    - map_size: (num_rows, num_cols) — total grid cells (already accounts for padding)
    - spacing: (row_spacing, col_spacing) — meters per cell
    - env_size: (env_num_rows, env_num_cols) — inner track area cells per tile
    - color_sampling: unused, kept for API compatibility
    - tile_padding_cells: gap cells added around each inner track tile

    Returns:
    - USD mesh fields (vertices, faces, face_counts, face_colors_triangle)
      and a TrackCache.
    """
    num_rows, num_cols = map_size
    row_spacing, col_spacing = spacing
    env_num_rows, env_num_cols = env_size

    # Convention: col -> x, row -> y.
    width = num_cols * col_spacing   # x-extent
    height = num_rows * row_spacing  # y-extent

    padded_num_rows = env_num_rows + tile_padding_cells
    padded_num_cols = env_num_cols + tile_padding_cells
    half_pad = tile_padding_cells // 2

    if num_rows % padded_num_rows != 0 or num_cols % padded_num_cols != 0:
        raise ValueError("Map size must be a multiple of the padded tile size.")

    num_env_rows = num_rows // padded_num_rows
    num_env_cols = num_cols // padded_num_cols

    # Simple 4-vertex flat grey ground plane (no traversability coloring).
    vertices = [
        (-width / 2, -height / 2, 0),
        ( width / 2, -height / 2, 0),
        (-width / 2,  height / 2, 0),
        ( width / 2,  height / 2, 0),
    ]
    faces = [0, 1, 2, 2, 1, 3]
    face_counts = [3, 3]

    # init structs for track data cache
    num_tiles = num_env_rows * num_env_cols
    tile_polylines_w: list[np.ndarray] = [] # world-meter polylines per tile
    tile_track_widths_m: list[float] = [] # world-meter track width per tile
    tile_is_closed: list[bool] = [] # True for closed-loop tracks (phase 2)
    tile_origins_w = np.zeros((num_tiles, 2), dtype=np.float32)
    tile_cell_bounds = np.zeros((num_tiles, 4), dtype=np.int32)
    cone_spacing_m = float(_TER.get("cone_spacing_m", 1.5))
    left_boundary_positions: list[np.ndarray] = []
    right_boundary_positions: list[np.ndarray] = []

    # Generate per-env tracks and build track data cache.
    # Each tile occupies a padded_num_rows × padded_num_cols slot in the global
    # grid; the actual track lives in the inner env_num_rows × env_num_cols region
    # offset by half_pad on each side.
    for i in range(num_env_rows):
        for j in range(num_env_cols):
            tile_idx = i * num_env_cols + j
            slot_start_row = i * padded_num_rows
            slot_start_col = j * padded_num_cols
            track_start_row = slot_start_row + half_pad
            track_start_col = slot_start_col + half_pad

            _diff_lo, _diff_hi = _TER["difficulty_range"]
            config = CurriculumConfig(
                difficulty=np.random.uniform(float(_diff_lo), float(_diff_hi)),
                phase_boundary=float(_TER["phase_boundary"]),
                steps_per_segment=int(_TER["steps_per_segment"]),
                track_width=int(_TER["track_width_cells"]),
                margin_frac=float(_TER["margin_frac"]),
            )
            config.resolve()  # resolve upfront so we can capture config.track_width per-tile
            _, polyline = generate_track(env_size=env_size, config=config)

            # Convert tile-local cell-coord polyline -> world meters.
            # poly_cells[:, 0] is the col-direction (x), poly_cells[:, 1] is the row-direction (y).
            poly_cells = np.asarray(polyline, dtype=np.float32)  # (M_tile, 2)
            world_x = (poly_cells[:, 0] + track_start_col - num_cols / 2.0) * col_spacing
            world_y = (poly_cells[:, 1] + track_start_row - num_rows / 2.0) * row_spacing
            tile_polylines_w.append(np.stack([world_x, world_y], axis=-1))

            # Track width: cells -> meters (approximate - asymmetric dilation in rasterise
            # means the effective corridor is ~= track_width * spacing).
            tile_w_m = float(config.track_width) * row_spacing
            tile_track_widths_m.append(tile_w_m)
            tile_is_closed.append(not config.is_chain)

            left_pos, right_pos = _compute_cone_positions(
                tile_polylines_w[-1], tile_w_m, cone_spacing_m
            )
            left_boundary_positions.append(left_pos)
            right_boundary_positions.append(right_pos)

            # Tile origin: world-meter center of the inner track area.
            # Used by out_of_tile() with tile_extent_m = inner track size.
            tile_center_col = track_start_col + env_num_cols / 2.0
            tile_center_row = track_start_row + env_num_rows / 2.0
            tile_origins_w[tile_idx, 0] = (tile_center_col - num_cols / 2.0) * col_spacing
            tile_origins_w[tile_idx, 1] = (tile_center_row - num_rows / 2.0) * row_spacing

            # Cell bounds: inner track area (used by per-env spawn restriction).
            tile_cell_bounds[tile_idx] = (
                track_start_row, track_start_row + env_num_rows,
                track_start_col, track_start_col + env_num_cols,
            )

    # Pad polylines to uniform shape with NaN; build tangents + segment-valid mask.
    M_max = max(len(p) for p in tile_polylines_w)
    polylines_w = np.full((num_tiles, M_max, 2), np.nan, dtype=np.float32)
    for k, pl in enumerate(tile_polylines_w):
        polylines_w[k, :len(pl)] = pl

    # Tangent approximations
    diffs = polylines_w[:, 1:] - polylines_w[:, :-1] # (num_tiles, M_max-1, 2)
    valid_endpoints = ~np.isnan(polylines_w).any(axis=-1) # (num_tiles, M_max)
    segment_valid = valid_endpoints[:, :-1] & valid_endpoints[:, 1:] # (num_tiles, M_max-1)
    lengths = np.linalg.norm(diffs, axis=-1, keepdims=True)
    safe_lengths = np.where(lengths > 0, lengths, 1.0)
    tangents_w = (diffs / safe_lengths).astype(np.float32)
    tangents_w[~segment_valid] = 0.0

    # Per-segment lengths (zeroed past the polyline so cumsum doesn't accumulate
    # padding) and prefix-sum cumulative arc-length per polyline vertex.
    segment_lengths_m = lengths.squeeze(-1).astype(np.float32) # (num_tiles, M_max-1)
    segment_lengths_m = np.where(segment_valid, segment_lengths_m, 0.0).astype(np.float32)
    cumulative_arc_lengths_m = np.concatenate(
        [
            np.zeros((num_tiles, 1), dtype=np.float32),
            np.cumsum(segment_lengths_m, axis=1, dtype=np.float32),
        ],
        axis=1,
    ) # (num_tiles, M_max)
    total_lengths_m = cumulative_arc_lengths_m[:, -1].astype(np.float32) # (num_tiles,)

    track_cache = TrackCache(
        polylines_w=polylines_w,
        tangents_w=tangents_w,
        segment_lengths_m=segment_lengths_m,
        cumulative_arc_lengths_m=cumulative_arc_lengths_m,
        total_lengths_m=total_lengths_m,
        is_closed=np.asarray(tile_is_closed, dtype=bool),
        segment_valid=segment_valid,
        track_widths_m=np.asarray(tile_track_widths_m, dtype=np.float32),
        tile_origins_w=tile_origins_w,
        tile_extent_m=(env_num_cols * col_spacing, env_num_rows * row_spacing),
        tile_cell_bounds=tile_cell_bounds,
    )

    grey = Gf.Vec3f(0.15, 0.15, 0.15)
    face_colors_triangle = [grey, grey]

    return vertices, faces, face_counts, face_colors_triangle, track_cache, \
        left_boundary_positions, right_boundary_positions


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------
def create_track_geometry(file_path, map_size, spacing, env_size, color_sampling=False, tile_padding_cells=0):
    """Create a USD file with a flat grey ground plane and cone markers.

    Returns the TrackCache for storage on RacingTerrainImporterCfg.
    Track boundaries are defined by cones; no traversability grid is used.

    Args:
    - file_path: output USD path (written to disk)
    - map_size: (num_rows, num_cols) — total grid cells (padding already included)
    - spacing: (row_spacing, col_spacing) in meters
    - env_size: inner track cells per tile
    - color_sampling: unused, kept for API compatibility
    - tile_padding_cells: gap cells around each tile (passed to plane generator)
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())

    plane = UsdGeom.Mesh.Define(stage, '/World/ground_plane')

    vertices, faces, face_counts, face_colors, track_cache, left_pos, right_pos = \
        generated_colored_track_plane(map_size, spacing, env_size, color_sampling, tile_padding_cells)

    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors)

    UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")

    _spawn_cones_in_stage(stage, left_pos, right_pos)

    stage.GetRootLayer().Save()

    return track_cache
