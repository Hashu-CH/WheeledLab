"""
track_gen — Curriculum track generation created for use by WheeledLab's visual task

Usage:
    from track_gen import CurriculumConfig, generate_track

    cfg = CurriculumConfig(difficulty=agent.current_level)
    grid, meta = generate_track(config=cfg)

Package layout:
    bezier.py      — sample_bezier, build_spline, build_polylines, make_segment
    rasterise.py   — rasterise_track
    points.py      — generate_points, sort_clockwise
    features.py    — chain templates (straight→hairpin), pure geometry
    chain.py       — sampling, intersection rejection, build_chain
    curriculum.py  — CurriculumConfig, generate_track
    viz.py         — plot_curriculum_progression, plot_chain_features
"""

# Core track pipeline for WheeledLab Formatting
from .bezier import (
    sample_bezier,
    to_bezier_centripetal,
    build_spline,
    build_polylines,
    make_segment,
)
from .rasterise import rasterise_track
from .points import generate_points, sort_clockwise

# Pipeline to build phase 1 track features
from .features import (
    FEATURES,
    FEATURE_ORDER,
    feat_straight,
    feat_curve,
    feat_varying_curve,
    feat_s_curve,
    feat_chicane,
    feat_hairpin,
)
from .chain import build_chain

# Curriculum entry points
from .curriculum import CurriculumConfig, generate_track

# For testing and visualizations
from .viz import plot_curriculum_progression

# For * import calls
__all__ = [
    # bezier
    "sample_bezier", "to_bezier_centripetal", "build_spline",
    "build_polylines", "make_segment",
    # rasterise
    "rasterise_track",
    # points
    "generate_points", "sort_clockwise",
    # features
    "FEATURES", "FEATURE_ORDER", "build_chain",
    "feat_straight", "feat_curve", "feat_varying_curve",
    "feat_s_curve", "feat_chicane", "feat_hairpin",
    # curriculum
    "CurriculumConfig", "generate_track",
    # viz
    "plot_curriculum_progression",
]
