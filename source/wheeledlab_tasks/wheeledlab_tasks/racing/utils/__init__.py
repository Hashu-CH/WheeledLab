"""
Terrain/Plane generation and track-geometry cache for the Racing Task.

Notes:

- Track-aware rewards need a geometric view of the terrain — centerline
  polyline, tangents, per-tile widths — which live in the TrackCache
  dataclass below and are populated when the USD is authored.
- A rasterised traversability grid is still computed inside
  generated_colored_track_plane to color the ground-plane mesh faces
  (black = off-track, white = on-track). It gets discarded after the USD is
  written; reward logic and spawn sampling run off the polyline cache.
- Coord-frame convention: polylines from curriculum.generate_track are in
  grid-cell units within a tile (x in [0, env_num_cols], y in [0, env_num_rows]).
  We convert to WORLD METERS here so reward-time projection doesn't re-derive
  the transform. All TrackCache arrays are world-frame.

Concern:

- Right now, each environment (robot) gets its own track and tile. Which 1, constraints 
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

from .procedural_track_gen.curriculum import CurriculumConfig, generate_track


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
    segment_valid: np.ndarray # (num_tiles, M_max - 1), bool
    track_widths_m: np.ndarray # (num_tiles,), meters
    tile_origins_w: np.ndarray # (num_tiles, 2), world-meter tile centers
    tile_extent_m: tuple[float, float] # (x_extent, y_extent) per tile, meters (fixed here)
    tile_cell_bounds: np.ndarray # (num_tiles, 4): row_start, row_end, col_start, col_end


# ---------------------------------------------------------------------------
# Map sizing
# ---------------------------------------------------------------------------
def compute_map_size(num_envs, env_size):
    """Map dimensions (in cells) to pack `num_envs` non-overlapping sub-envs in
    a square-ish grid.

    num_tiles == num_envs which lets env_id == tile_id hold.

    Args:
    - num_envs: total parallel envs (must be a power of 2)
    - env_size: (env_num_rows, env_num_cols) — cells per tile
    """
    if num_envs < 1 or (num_envs & (num_envs - 1)) != 0:
        raise ValueError(f"num_envs must be a power of 2, got {num_envs}")
    env_num_rows, env_num_cols = env_size
    exponent = num_envs.bit_length() - 1
    grid_rows = 1 << (exponent // 2)
    grid_cols = 1 << (exponent - exponent // 2)
    return grid_rows * env_num_rows, grid_cols * env_num_cols


# ---------------------------------------------------------------------------
# Terrain + track-cache construction
# ---------------------------------------------------------------------------
def generated_colored_track_plane(map_size, spacing, env_size, color_sampling):
    """Build the colored ground-plane mesh plus the per-tile track cache.

    For each tile we generate a procedural track, rasterise it into the global
    traversability hashmap, and capture the centerline polyline (converted to
    world meters) into TrackCache. The cache is the geometric counterpart to
    the pixel hashmap — rewards use it for centerline projection while the
    hashmap is what is rendered and 'seen'.

    Args:
    - map_size: (num_rows, num_cols) — total grid cells across the whole map
    - spacing: (row_spacing, col_spacing) — meters per cell
    - env_size: (env_num_rows, env_num_cols) — cells per tile
    - color_sampling: whether to jitter tile colors (inherited from visual task)

    Returns:
    - USD mesh fields (vertices, faces, face_counts, face_colors_triangle)
      and a TrackCache. The rasterised grid is used only for the mesh faces
    """
    num_rows, num_cols = map_size
    row_spacing, col_spacing = spacing
    env_num_rows, env_num_cols = env_size

    width = num_rows * row_spacing
    height = num_cols * col_spacing

    if num_rows % env_num_rows != 0 or num_cols % env_num_cols != 0:
        raise ValueError("Map size must be a multiple of the sub environment size.")

    num_env_rows = num_rows // env_num_rows
    num_env_cols = num_cols // env_num_cols

    xs = np.linspace(-width / 2, width / 2, num_rows) - row_spacing / 2
    ys = np.linspace(-height / 2, height / 2, num_cols) - col_spacing / 2
    xx, yy = np.meshgrid(xs, ys)

    vertices = [(x, y, 0) for x, y in zip(xx.ravel(), yy.ravel())]

    def color_sampler(r, g, b, range):
        r = np.random.uniform(r - range // 2, r + range // 2) / 255.
        g = np.random.uniform(g - range // 2, g + range // 2) / 255.
        b = np.random.uniform(b - range // 2, b + range // 2) / 255.
        return Gf.Vec3f(r, g, b)

    if color_sampling:
        colors = [
            color_sampler(30, 30, 30, 30),
            color_sampler(220.0, 220.0, 220.0, 30),
        ]
    else:
        colors = [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 1.0, 1.0),
        ]

    # USD mesh takes triangles. 2 per tile
    faces = []
    face_counts = []
    traversability_hashmap = np.zeros((num_rows, num_cols)).astype(bool)
    for row_index in range(num_rows - 1):
        for col_index in range(num_cols - 1):
            v0 = row_index * num_cols + col_index
            v1 = v0 + 1
            v2 = v0 + num_cols
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    # init structs for track data cache
    num_tiles = num_env_rows * num_env_cols
    tile_polylines_w: list[np.ndarray] = [] # world-meter polylines per tile
    tile_track_widths_m: list[float] = [] # world-meter track width per tile
    tile_origins_w = np.zeros((num_tiles, 2), dtype=np.float32)
    tile_cell_bounds = np.zeros((num_tiles, 4), dtype=np.int32)

    # Generate per env tracks and fills track data cache
    for i in range(num_env_rows):
        for j in range(num_env_cols):
            tile_idx = i * num_env_cols + j
            start_row = i * env_num_rows
            start_col = j * env_num_cols

            config = CurriculumConfig(
                difficulty=np.random.uniform(.5, .90), 
                steps_per_segment=30,
                track_width=1,
            )
            config.resolve()  # resolve upfront so we can capture config.track_width per-tile
            grid, polyline = generate_track(env_size=env_size, config=config)
            traversability_hashmap[
                start_row:start_row + env_num_rows,
                start_col:start_col + env_num_cols,
            ] = grid

            # TODO if polyline is rotated, swap poly cells
            # Convert tile-local cell-coord polyline -> world meters.
            poly_cells = np.asarray(polyline, dtype=np.float32)  # (M_tile, 2)
            world_x = (poly_cells[:, 0] + start_col - num_cols / 2.0) * row_spacing
            world_y = (poly_cells[:, 1] + start_row - num_rows / 2.0) * col_spacing
            tile_polylines_w.append(np.stack([world_x, world_y], axis=-1))

            # Track width: cells -> meters (approximate - asymmetric dilation in rasterise
            # means the effective corridor is ~= track_width * spacing).
            tile_track_widths_m.append(float(config.track_width) * row_spacing)

            # Tile origin: world-meter center of the tile. Stored for viz /
            # debug; not currently used at runtime — see TODO in
            # mushr_racing_env_cfg.py about env-local observations.
            tile_center_col = start_col + env_num_cols / 2.0
            tile_center_row = start_row + env_num_rows / 2.0
            tile_origins_w[tile_idx, 0] = (tile_center_col - num_cols / 2.0) * row_spacing
            tile_origins_w[tile_idx, 1] = (tile_center_row - num_rows / 2.0) * col_spacing

            # Cell bounds: used by per-env spawn restriction.
            tile_cell_bounds[tile_idx] = (
                start_row, start_row + env_num_rows,
                start_col, start_col + env_num_cols,
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

    # Save track cache for generated world plane
    track_cache = TrackCache(
        polylines_w=polylines_w,
        tangents_w=tangents_w,
        segment_valid=segment_valid,
        track_widths_m=np.asarray(tile_track_widths_m, dtype=np.float32),
        tile_origins_w=tile_origins_w,
        tile_extent_m=(env_num_rows * row_spacing, env_num_cols * col_spacing),
        tile_cell_bounds=tile_cell_bounds,
    )

    # paint triangle colors for USD mesh init
    face_colors = [
        colors[int(traversability_hashmap[row_index, col_index])]
        for row_index in range(num_rows - 1)
        for col_index in range(num_cols - 1)
    ]
    face_colors_triangle = []
    for color in face_colors:
        face_colors_triangle += [color, color]

    return vertices, faces, face_counts, face_colors_triangle, track_cache


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------
def create_track_geometry(file_path, map_size, spacing, env_size, color_sampling=False):
    """Create a USD file with a rasterised track plane.

    Returns the TrackCache for storage on RacingTerrainImporterCfg. 
    
    Change from visual task: travers_hashmap is used just to color mesh.
    All rewards follow the pure geometry from the track cache

    Args:
    - file_path: output USD path (written to disk)
    - map_size: (num_rows, num_cols)
    - spacing: (row_spacing, col_spacing) in meters
    - env_size: cells per tile
    - color_sampling: whether to jitter tile colors
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(file_path)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.LinearUnits.meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    xform = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(xform.GetPrim())

    plane = UsdGeom.Mesh.Define(stage, '/World/colored_plane')

    vertices, faces, face_counts, face_colors, track_cache = \
        generated_colored_track_plane(map_size, spacing, env_size, color_sampling)

    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)
    plane.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(face_colors)

    UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")

    stage.GetRootLayer().Save()

    return track_cache

# Spawn-pose sampling lives in track_utils.sample_poses_along_polylines —
# kept next to project_nearest_segment since both operate on the same cache.
