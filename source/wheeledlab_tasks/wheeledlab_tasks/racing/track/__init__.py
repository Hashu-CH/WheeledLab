"""
Track subpackage for racing task terrain.

Very briefly:
- runtime.py: defines RacingTerrainImporter as non-serializable terrain component
- generator.py: generates track plane and initializes the track cache
- projection.py: centerline projection and sampling helpers
- procedural/: per tile track generation (includes chains and loops)

Usage:
- generator.create_track_geometry runs inside RacingTerrainImporter.__init__
  (i.e. when InteractiveScene builds the terrain), not at config-parse time.
- runtime.py is held by the scene at env.scene.terrain. Handles utility
  functions like traversability lookup, progress state, and other updates
  used primarily for mdp functions.
"""

from .runtime import RacingTerrainImporter
from .generator import (
    TrackCache,
    compute_map_size,
    create_track_geometry,
    generated_colored_track_plane,
)