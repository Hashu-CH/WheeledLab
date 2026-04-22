"""
Main entry point for track generation and curriculum config 

Phase 1 (difficulty < phase_boundary):  open Bezier chains
    - this phase is riddled with hardcoded hyper parameters 
    - in both the chain file and the resolve fn in this file 
    - if tracks look poor, hand tune till your fingers are sore :)
Phase 2 (difficulty geq phase_boundary):  full closed loop tracks

Notes: 

- I made this unwisely. Splines are computed in world coordinates 
  which then have to be converted to grid coordinates for the hashmap.
  I normalize the spline so I can round based on the grid sizing, 
  then in the utility. When finding track tangents you have to use 
  the unnormalized versions of the spline and find the closest 
  track point. I'm sorry. 
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bezier import build_spline, build_polylines
from .rasterise import rasterise_track
from .points import generate_points, sort_clockwise
from .features import FEATURES, FEATURE_ORDER
from .chain import build_chain


@dataclass
class CurriculumConfig:
    """
    Central knob: ``difficulty`` in [0, 1].

    Phase 1 (d < phase_boundary): open Bézier chains
        agent isolates individual features
    Phase 2 (d >= phase_boundary): closed loops
        agent handles continuous lapping

    None fields resolved via resolve()
    """
    difficulty:      float = 0.0
    phase_boundary:  float = 0.50

    # phase 1 params
    n_segments:      Optional[int]   = None
    segment_length:  Optional[float] = None
    intensity_range: Optional[tuple] = None

    # shared between
    track_width:       Optional[int] = None
    steps_per_segment: int           = 40

    # phase 2 params
    n_control_points: Optional[int]   = None
    max_jitter:       Optional[float] = None
    min_radius:       Optional[float] = None

    def resolve(self):
        """Resolve empty fields via linear interpolation"""
        d = np.clip(self.difficulty, 0.0, 1.0)

        if self.track_width is None:
            self.track_width = int(np.interp(d, [0, 1], [8, 3]))

        if d < self.phase_boundary:
            dp = d / self.phase_boundary  # normalise within Phase 1
            if self.n_segments is None:
                self.n_segments = int(np.interp(dp, [0, 1], [4, 12]))
            # segment_length resolved in generate_chain (depends on env_size)
            if self.intensity_range is None:
                lo = float(np.interp(dp, [0, 1], [0.15, 0.50]))
                hi = float(np.interp(dp, [0, 1], [0.40, 1.00]))
                self.intensity_range = (lo, hi)
        else:
            dp = (d - self.phase_boundary) / (1.0 - self.phase_boundary)
            if self.n_control_points is None:
                self.n_control_points = int(np.interp(dp, [0, 1], [6, 14]))
            if self.max_jitter is None:
                self.max_jitter = float(np.interp(dp, [0, 1], [0.15, 0.40]))
            if self.min_radius is None:
                self.min_radius = float(np.interp(dp, [0, 1], [0.50, 0.30]))
        return self

    @property
    def is_chain(self) -> bool:
        return self.difficulty < self.phase_boundary

    def available_features(self) -> list[str]:
        return [f for f in FEATURE_ORDER if FEATURES[f]["unlock"] <= self.difficulty]


def generate_track(
    env_size = (100, 100),
    config:    CurriculumConfig    = None,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Single entry point for the full curriculum.

    Returns
        grid : np.ndarray[bool] rasterised traversability map
        polyline : list of (x, y) points in tile-local cell coords
                   (x in [0, cols], y in [0, rows])
    """
    if config is None:
        config = CurriculumConfig()
    config.resolve()

    rows, cols = env_size

    if config.is_chain:
        return generate_chain(config, rows, cols)
    else:
        return generate_loop(config, rows, cols)


def generate_chain(config, rows, cols):
    """Phase 1: chain builder"""
    if config.segment_length is None:
        config.segment_length = min(rows, cols) * 0.15

    start_pos = np.array([cols * 0.2, rows * 0.5])
    start_tan = np.array([1.0, 0.0]) * config.segment_length

    segments, feat_log = build_chain(
        config.n_segments, config.segment_length,
        config.difficulty, cols, rows,
        config.intensity_range,
        start_pos, start_tan,
    )
    polyline = build_polylines(segments, config.steps_per_segment)
    if len(polyline) == 0:
        polyline = [(cols * 0.5, rows * 0.15), (cols * 0.5, rows * 0.85)]
        feat_log = [{"feature": "straight", "intensity": 0.0}]
    temp = np.array(polyline)
    line_normalized = list(map(tuple, temp / np.array([cols, rows])))

    grid = rasterise_track(
        line_normalized, rows, cols, config.track_width, closed=False)

    return grid, polyline

def generate_loop(config, rows, cols):
    """Phase 2: closed loop full track."""
    raw = generate_points(
        config.n_control_points, cols, rows,
        config.min_radius, margin=cols * 0.05,
        max_jitter=config.max_jitter,
    )
    pts = sort_clockwise(raw)
    segments = build_spline(pts)
    polyline = build_polylines(segments, config.steps_per_segment)
    temp = np.array(polyline) 
    line_normalized = list(map(tuple, temp / np.array([cols, rows])))
    grid = rasterise_track(
        line_normalized, rows, cols, config.track_width, closed=True)

    return grid, polyline
