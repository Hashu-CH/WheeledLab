"""
Visual test for the per-tile polyline tangent cache built in RacingTerrainImporterCfg.

python source/wheeledlab_rl/scripts/viz_track_cache.py --num-envs 16 --out ./imgs/track_viz.png
"""

import argparse
import pathlib
import sys


_RACING_DIR = pathlib.Path(__file__).resolve().parents[2] / "wheeledlab_tasks/wheeledlab_tasks/racing"
sys.path.insert(0, str(_RACING_DIR))

import matplotlib
matplotlib.use("Agg")  # headless

from utils import compute_map_size, generated_colored_track_plane
from utils.verify_track_cache import visualize_track_cache

ROW_SPACING = 0.5
COL_SPACING = 0.5
ENV_NUM_ROWS = 12
ENV_NUM_COLS = 12
COLOR_SAMPLING = False

parser = argparse.ArgumentParser(description="Render track cache tiles.")
parser.add_argument("--num-envs", type=int, default=16,
                    help="Must be a power of 2.")
parser.add_argument("--tiles", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="Tile indices to render.")
parser.add_argument("--out", type=str, default="track_viz.png",
                    help="Output PNG path.")
args = parser.parse_args()

spacing = (ROW_SPACING, COL_SPACING)
env_size = (ENV_NUM_ROWS, ENV_NUM_COLS)
map_size = compute_map_size(args.num_envs, env_size)

*_, track_cache = generated_colored_track_plane(
    map_size, spacing, env_size, COLOR_SAMPLING,
)

fig = visualize_track_cache(
    track_cache, spacing, env_size, tile_indices=tuple(args.tiles),
)
fig.savefig(args.out, dpi=120)
print(f"wrote {args.out}  (tiles={args.tiles}, num_envs={args.num_envs})")
