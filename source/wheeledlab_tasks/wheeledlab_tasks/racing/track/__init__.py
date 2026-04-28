"""
Track subpackage for racing task terrain.

Very briefly:
- runtime.py: defines RacingTerrainImporter as non-serializable terrain component
- generator.py: generates track plane and initializes the track cache
- projectinon.py: centerline projection and sampling helpers
- procedural/: per tile track generation (includes chains and loops)

Usage: 
- generator.py runs during scene post_init -- builds USD and track cache.
- runtime.py is held by the scene at env.scene.terrain. Handles utility 
  functions like traversability lookup, progress state, and other updates
  used primarily for mdp functions.
"""

from .runtime import RacingTerrainImporter, stash_track_cache
from .generator import (
    TrackCache,
    compute_map_size,
    create_track_geometry,
    generated_colored_track_plane,
)