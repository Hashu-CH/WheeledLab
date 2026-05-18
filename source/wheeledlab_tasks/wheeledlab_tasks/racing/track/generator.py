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
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

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

# Two alternating texture files so events.py can force USD to reload by toggling paths.
GROUND_TEX_PATH     = "/tmp/wheeledlab_ground_tex_a.png"
GROUND_TEX_PATH_ALT = "/tmp/wheeledlab_ground_tex_b.png"
# USD prim path for the texture reader shader (read by events.py at runtime).
GROUND_TEX_READER_PATH = "/World/ground_material/TexReader"


def _write_grey_texture(path: str, resolution: int = 128, value: int = 77) -> None:
    """Write a solid grey 8-bit RGB PNG using only stdlib.

    RGB (color type 2), not greyscale, because Omniverse's gpu.foundation
    texture loader treats 1-channel PNGs as "empty file" and refuses them.
    Stdlib-only so a missing PIL install can't silently leave a 0-byte file.
    Errors are NOT swallowed; if the write fails the caller should know.
    """
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    # IHDR: 8-bit RGB (color type 2).
    ihdr = struct.pack(">IIBBBBB", resolution, resolution, 8, 2, 0, 0, 0)
    # IDAT: each scanline = filter byte (0=None) + 3 bytes (R,G,B) per pixel.
    pixel = bytes([value, value, value])
    row = b"\x00" + pixel * resolution
    idat = zlib.compress(row * resolution, 9)

    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", idat)
           + _chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(png)


def _create_ground_material(stage, mat_path: str, tex_file_path: str):
    """Create a UsdPreviewSurface material with a UV-mapped texture and return it."""
    mat = UsdShade.Material.Define(stage, mat_path)

    st_reader = UsdShade.Shader.Define(stage, mat_path + "/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, mat_path + "/TexReader")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_file_path))
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    surf_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    mat.CreateSurfaceOutput().ConnectToSource(surf_out)
    return mat


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
    # Per-vertex clamped half-widths after inner-curve clamp; off-track reward
    # checks d_signed against these (left = car on CCW-normal side of tangent).
    # Padded with 0 past valid polyline length; never queried there.
    left_half_widths_m: np.ndarray  # (num_tiles, M_max), float32
    right_half_widths_m: np.ndarray # (num_tiles, M_max), float32
    tile_origins_w: np.ndarray # (num_tiles, 2), world-meter tile centers
    tile_extent_m: tuple[float, float] # (x_extent, y_extent) per tile, meters (fixed here)
    tile_cell_bounds: np.ndarray # (num_tiles, 4): row_start, row_end, col_start, col_end


# ---------------------------------------------------------------------------
# Cone placement helpers
# ---------------------------------------------------------------------------

# Safety margin (m) between inner cone and the arc's center of curvature.
# Also the floor on inner half-width: prevents inner cones from collapsing
# onto the centerline on extreme hairpins (radius < INNER_SAFETY_M).
# Shared by cone placement and the per-vertex width cache so the cones the
# car sees match the corridor the reward enforces.
INNER_SAFETY_M = 0.05


def _per_vertex_half_widths(
    polyline_w: np.ndarray,    # (M, 2)
    half_width_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-vertex (kappa_signed, left_half_w, right_half_w) after inner-curve clamp.

    On a turn with local radius R = 1/|kappa| < half_width_m, the inner cone
    would cross the centerline if placed at the nominal half_width_m. We
    clamp the inner half-width to (R - INNER_SAFETY_M), floored at
    INNER_SAFETY_M. The outer side keeps half_width_m. On straights/gentle
    turns both sides equal half_width_m. Sign convention: kappa > 0 = CCW.
    """
    M = len(polyline_w)
    if M < 3:
        return (np.zeros(M, dtype=np.float32),
                np.full(M, half_width_m, dtype=np.float32),
                np.full(M, half_width_m, dtype=np.float32))
    diffs = np.diff(polyline_w, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    safe_lens = np.where(seg_lens > 0, seg_lens, 1.0)
    tans = diffs / safe_lens[:, None]
    cross_signed = tans[:-1, 0] * tans[1:, 1] - tans[:-1, 1] * tans[1:, 0]
    avg_len = 0.5 * (seg_lens[:-1] + seg_lens[1:])
    kappa_v = np.zeros(M, dtype=np.float64)
    kappa_v[1:-1] = cross_signed / np.maximum(avg_len, 1e-6)

    abs_k = np.abs(kappa_v)
    radius = np.where(abs_k > 1e-6, 1.0 / np.maximum(abs_k, 1e-6), np.inf)
    inner = np.minimum(half_width_m, np.maximum(radius - INNER_SAFETY_M, INNER_SAFETY_M))
    left  = np.where(kappa_v > 0, inner, half_width_m).astype(np.float32)
    right = np.where(kappa_v < 0, inner, half_width_m).astype(np.float32)
    return kappa_v.astype(np.float32), left, right


def _compute_cone_positions(
    polyline_w: np.ndarray,       # (M, 2) world-meter centerline
    half_width_m: float,
    cone_spacing_m: float,
    curvature_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_positions, right_positions) arrays of shape (N, 2).

    Left/right are defined relative to the direction of travel along the
    centerline. Left = 90° CCW from tangent, right = 90° CW.

    When curvature_scale > 0, local spacing shrinks on tighter bends:
        local_spacing = cone_spacing_m / (1 + curvature_scale * kappa)
    clamped to cone_spacing_m / 4 as a minimum, giving ~4× peak density on
    the tightest corners without arbitrarily small steps on near-zero-radius
    numerical artifacts.

    Inner-curve clamping (see _per_vertex_half_widths): on hairpins the inner
    side's half-width shrinks toward the centerline so cones never cross to
    the opposite side. The same per-vertex widths are stored on the TrackCache,
    so the reward boundary matches the visible cone gate exactly.
    """
    if len(polyline_w) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Arc-length prefix sum along centerline
    diffs = np.diff(polyline_w, axis=0)                         # (M-1, 2)
    seg_lens = np.linalg.norm(diffs, axis=1)                    # (M-1,)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])       # (M,)
    total_len = cum_len[-1]
    if total_len < cone_spacing_m:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # Per-vertex curvature + clamped widths; reused by the reward via TrackCache.
    kappa_v_signed, left_v, right_v = _per_vertex_half_widths(polyline_w, half_width_m)
    kappa_v_abs = np.abs(kappa_v_signed)

    # --- Sample arc positions ------------------------------------------------
    if curvature_scale > 0.0 and len(polyline_w) >= 3:
        min_spacing = cone_spacing_m / 4.0
        arc_positions = [0.0]
        pos = 0.0
        while True:
            kappa = float(np.interp(pos, cum_len, kappa_v_abs))
            step = max(min_spacing, cone_spacing_m / (1.0 + curvature_scale * kappa))
            pos += step
            if pos >= total_len - 1e-6:
                break
            arc_positions.append(pos)
        sample_arcs = np.asarray(arc_positions, dtype=np.float64)
    else:
        sample_arcs = np.arange(0.0, total_len, cone_spacing_m)

    sample_pts = np.stack([
        np.interp(sample_arcs, cum_len, polyline_w[:, 0]),
        np.interp(sample_arcs, cum_len, polyline_w[:, 1]),
    ], axis=1)                                                   # (N, 2)

    # Tangent at each sample via finite difference on the resampled points
    tan = np.diff(sample_pts, axis=0)                           # (N-1, 2)
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True)
    tan_norm = np.where(tan_norm > 0, tan_norm, 1.0)
    tan = tan / tan_norm                                         # unit tangents
    tan = np.concatenate([tan, tan[[-1]]], axis=0)              # (N, 2)

    # Left normal = CCW 90°: (tx, ty) -> (-ty, tx)
    left_n  = np.stack([-tan[:, 1],  tan[:, 0]], axis=1)
    right_n = np.stack([ tan[:, 1], -tan[:, 0]], axis=1)

    # Interpolate per-vertex widths onto sample arcs so cones sit at the same
    # offsets the runtime reward will recover from the TrackCache.
    left_half_w  = np.interp(sample_arcs, cum_len, left_v)
    right_half_w = np.interp(sample_arcs, cum_len, right_v)

    left_pos  = sample_pts + left_half_w[:, None]  * left_n
    right_pos = sample_pts + right_half_w[:, None] * right_n

    return left_pos.astype(np.float32), right_pos.astype(np.float32)


def _spawn_cones_in_stage(
    stage: Usd.Stage,
    left_positions_per_tile:  list[np.ndarray],   # list of (N_i, 2) float32
    right_positions_per_tile: list[np.ndarray],
) -> None:
    """Author orange (left) and blue (right) cones as a single PointInstancer.

    One prim per color is referenced as a prototype; every cone instance is just
    an entry in the positions/protoIndices arrays. Keeps USD composition O(2)
    references instead of O(num_cones), which is the dominant cost on large
    tracks (tens of thousands of cones across all envs).
    """
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
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/cones/instancer")

    # Prototypes live under the instancer; PointInstancer hides them from
    # ordinary imageable traversal, so they only render via instances.
    orange_proto_path = Sdf.Path("/World/cones/instancer/Prototypes/orange")
    blue_proto_path   = Sdf.Path("/World/cones/instancer/Prototypes/blue")
    UsdGeom.Scope.Define(stage, "/World/cones/instancer/Prototypes")
    orange_proto = UsdGeom.Xform.Define(stage, orange_proto_path)
    orange_proto.GetPrim().GetReferences().AddReference(_ORANGE_CONE_USD)
    blue_proto = UsdGeom.Xform.Define(stage, blue_proto_path)
    blue_proto.GetPrim().GetReferences().AddReference(_BLUE_CONE_USD)

    total = sum(len(l) + len(r) for l, r in
                zip(left_positions_per_tile, right_positions_per_tile))
    positions = np.empty((total, 3), dtype=np.float32)
    proto_indices = np.empty(total, dtype=np.int32)
    cursor = 0
    for left_pos, right_pos in zip(left_positions_per_tile, right_positions_per_tile):
        n_l = len(left_pos)
        positions[cursor:cursor + n_l, 0:2] = left_pos
        positions[cursor:cursor + n_l, 2]   = 0.0
        proto_indices[cursor:cursor + n_l]  = 0
        cursor += n_l
        n_r = len(right_pos)
        positions[cursor:cursor + n_r, 0:2] = right_pos
        positions[cursor:cursor + n_r, 2]   = 0.0
        proto_indices[cursor:cursor + n_r]  = 1
        cursor += n_r

    instancer.CreatePrototypesRel().SetTargets([orange_proto_path, blue_proto_path])
    instancer.CreatePositionsAttr().Set(
        [Gf.Vec3f(*p) for p in positions.tolist()]
    )
    instancer.CreateProtoIndicesAttr().Set(proto_indices.tolist())


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

    # Ground plane as a regular grid for per-episode texture DR.
    # Resolution is coarser than the track cell grid — one vertex per
    # ground_resolution_m rather than per row/col_spacing.
    grid_res = float(_TER.get("ground_resolution_m", 1.0))
    ncols_v = max(2, int(np.ceil(width  / grid_res)) + 1)
    nrows_v = max(2, int(np.ceil(height / grid_res)) + 1)
    xs_g = np.linspace(-width  / 2, width  / 2, ncols_v)
    ys_g = np.linspace(-height / 2, height / 2, nrows_v)
    xx_g, yy_g = np.meshgrid(xs_g, ys_g)
    vertices = [(float(x), float(y), 0.0) for x, y in zip(xx_g.ravel(), yy_g.ravel())]

    faces = []
    face_counts = []
    for r in range(nrows_v - 1):
        for c in range(ncols_v - 1):
            v0 = r * ncols_v + c
            v1 = v0 + 1
            v2 = v0 + ncols_v
            v3 = v2 + 1
            faces += [v0, v1, v2, v2, v1, v3]
            face_counts += [3, 3]

    # init structs for track data cache
    num_tiles = num_env_rows * num_env_cols
    tile_polylines_w: list[np.ndarray] = [] # world-meter polylines per tile
    tile_left_half_widths_m: list[np.ndarray] = []  # per-vertex clamped left half-width
    tile_right_half_widths_m: list[np.ndarray] = [] # per-vertex clamped right half-width
    tile_is_closed: list[bool] = [] # True for closed-loop tracks (phase 2)
    tile_origins_w = np.zeros((num_tiles, 2), dtype=np.float32)
    tile_cell_bounds = np.zeros((num_tiles, 4), dtype=np.int32)
    cone_spacing_m    = float(_TER.get("cone_spacing_m", 1.5))
    curvature_scale   = float(_TER.get("cone_curvature_scale", 0.0))
    # track_width_m is the single source of truth for the full L-cone to R-cone
    # corridor width. Cones, reward, and the routing planner all derive from it.
    # Planner gets a discrete cell count; ceil keeps the planner corridor >= the
    # cone corridor so routing never thinks it has less room than the gate allows.
    _track_width_m = float(_TER["track_width_m"])
    _planner_track_cells = max(1, int(np.ceil(_track_width_m / row_spacing)))
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
                track_width=_planner_track_cells,
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

            # Cones sit at ±cone_half_w from the centerline on straights; on
            # tight turns the inner side is clamped (see _per_vertex_half_widths).
            # Per-vertex widths feed both cone authoring and the off-track reward.
            cone_half_w = _track_width_m / 2.0
            tile_is_closed.append(not config.is_chain)

            _, left_v, right_v = _per_vertex_half_widths(tile_polylines_w[-1], cone_half_w)
            tile_left_half_widths_m.append(left_v)
            tile_right_half_widths_m.append(right_v)

            left_pos, right_pos = _compute_cone_positions(
                tile_polylines_w[-1], cone_half_w, cone_spacing_m, curvature_scale
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
    # Half-widths are 0-padded past the valid length — never queried there because
    # seg_idx comes from argmin over segment_valid.
    M_max = max(len(p) for p in tile_polylines_w)
    polylines_w = np.full((num_tiles, M_max, 2), np.nan, dtype=np.float32)
    left_half_widths_m = np.zeros((num_tiles, M_max), dtype=np.float32)
    right_half_widths_m = np.zeros((num_tiles, M_max), dtype=np.float32)
    for k, pl in enumerate(tile_polylines_w):
        polylines_w[k, :len(pl)] = pl
        left_half_widths_m[k, :len(pl)] = tile_left_half_widths_m[k]
        right_half_widths_m[k, :len(pl)] = tile_right_half_widths_m[k]

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
        left_half_widths_m=left_half_widths_m,
        right_half_widths_m=right_half_widths_m,
        tile_origins_w=tile_origins_w,
        tile_extent_m=(env_num_cols * col_spacing, env_num_rows * row_spacing),
        tile_cell_bounds=tile_cell_bounds,
    )

    # UV coordinates: vertex interpolation, u=col→[0,1], v=row→[0,1].
    # Covers the full ground plane in a single tile so each episode's
    # texture update is visible everywhere.
    uvs = [
        (float(c) / max(ncols_v - 1, 1), float(r) / max(nrows_v - 1, 1))
        for r in range(nrows_v)
        for c in range(ncols_v)
    ]

    return vertices, faces, face_counts, uvs, track_cache, \
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

    vertices, faces, face_counts, uvs, track_cache, left_pos, right_pos = \
        generated_colored_track_plane(map_size, spacing, env_size, color_sampling, tile_padding_cells)

    plane.GetPointsAttr().Set(vertices)
    plane.GetFaceVertexCountsAttr().Set(face_counts)
    plane.GetFaceVertexIndicesAttr().Set(faces)

    # UV primvar for texture mapping — vertex interpolation, one (u,v) per vertex.
    # Newer USD removed UsdGeom.Mesh.CreatePrimvar; go through PrimvarsAPI.
    st_primvar = UsdGeom.PrimvarsAPI(plane).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st_primvar.Set(uvs)

    # Write initial grey placeholder PNGs and bind a UV-mapped material.
    _write_grey_texture(GROUND_TEX_PATH)
    _write_grey_texture(GROUND_TEX_PATH_ALT)
    mat = _create_ground_material(stage, "/World/ground_material", GROUND_TEX_PATH)
    UsdShade.MaterialBindingAPI(plane.GetPrim()).Bind(mat)

    UsdPhysics.MeshCollisionAPI.Apply(xform.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(plane.GetPrim())
    UsdPhysics.CollisionGroup.Define(stage, "/World/colored_plane/collision_group")

    _spawn_cones_in_stage(stage, left_pos, right_pos)

    stage.GetRootLayer().Save()

    return track_cache
