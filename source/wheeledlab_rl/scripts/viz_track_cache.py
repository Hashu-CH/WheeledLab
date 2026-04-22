"""
Visual test for the per-tile polyline tangent cache built in RacingTerrainImporterCfg

python source/wheeledlab_rl/scripts/viz_track_cache.py --num-envs 16 --out /tmp/track_viz.png

Scene cfg's post_init should fire -> terrain configure -> track_cache population 
"""

from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="Render track cache tiles.")
parser.add_argument("--num-envs", type=int, default=16,
                    help="Must be a power of 2.")
parser.add_argument("--tiles", type=int, nargs="*", default=[0, 1, 2, 3],
                    help="Tile indices to render.")
parser.add_argument("--out", type=str, default="track_viz.png",
                    help="Output PNG path.")
simulation_app, args_cli = startup(parser=parser)

import matplotlib
matplotlib.use("Agg") # headless

from wheeledlab_tasks.racing.mushr_racing_env_cfg import MushrRacingRLEnvCfg
from wheeledlab_tasks.racing.utils.verify_track_cache import visualize_track_cache

cfg = MushrRacingRLEnvCfg(num_envs=args_cli.num_envs)

terrain = cfg.scene.terrain
assert terrain.track_cache is not None, "track_cache was not populated"

fig = visualize_track_cache(
    terrain.track_cache,
    terrain.spacing,
    terrain.env_size,
    tile_indices=tuple(args_cli.tiles),
)
fig.savefig(args_cli.out, dpi=120)
print(f"wrote {args_cli.out}  (tiles={args_cli.tiles}, num_envs={args_cli.num_envs})")

simulation_app.close()
