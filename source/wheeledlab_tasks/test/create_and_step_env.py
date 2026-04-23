"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser()
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# Racing + Visual tasks spawn a TiledCamera; Isaac Lab requires this flag to
# initialize the RTX camera pipeline. Force it on so this script works for
# camera-bearing tasks without the caller having to remember the flag.
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs.utils.spaces import sample_space
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import torch
import gymnasium as gym

import wheeledlab_tasks


def _dump_camera_frame(env, path: str = "cam_sample.png") -> None:
    """Save env 0's first RGB sensor frame to disk so we can eyeball whether the
    viewport's path-traced noise is also leaking into the policy's camera input.
    Skip silently if the scene has no `camera` sensor."""
    scene = env.unwrapped.scene
    if "camera" not in scene.sensors:
        return
    rgb = scene["camera"].data.output["rgb"][0]  # (H, W, 3), uint8 or float
    if rgb.dtype != torch.uint8:
        rgb = (rgb.clamp(0.0, 1.0) * 255).to(torch.uint8)
    from PIL import Image
    Image.fromarray(rgb.cpu().numpy()).save(path)
    print(f"[cam dump] wrote {path} shape={tuple(rgb.shape)}")


def main(task_name: str = "Isaac-MushrRacingRL-v0", num_envs: int = 4, num_steps: int = 20):
    env_cfg = parse_env_cfg(task_name, num_envs=num_envs)
    # parse_env_cfg's num_envs kwarg can be swallowed by __post_init__ that
    # builds the scene from its own default num_envs. Force both paths and
    # re-run terrain.configure so the USD + track cache match the override.
    env_cfg.num_envs = num_envs
    if hasattr(env_cfg, "scene") and env_cfg.scene is not None:
        env_cfg.scene.num_envs = num_envs
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "configure"):
            terrain.configure(num_envs)
            # Re-sync ground plane size to the rebuilt terrain dims.
            ground = getattr(env_cfg.scene, "ground", None)
            if ground is not None and hasattr(ground.spawn, "size"):
                ground.spawn.size = (terrain.width, terrain.height)
    env = gym.make(task_name, cfg=env_cfg)

    # reset environment
    obs, _ = env.reset()

    # Step a few times before dumping so the RTX renderer has produced a stable
    # frame (first post-reset frame can be blank / pre-render).
    with torch.inference_mode():
        for _ in range(5):
            actions = sample_space(
                env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
            )
            env.step(actions)
        _dump_camera_frame(env)

        for _ in range(num_steps):
            actions = sample_space(
                env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
            )
            env.step(actions)


if __name__ == "__main__":
    main()