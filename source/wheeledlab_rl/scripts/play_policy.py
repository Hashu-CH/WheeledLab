"""
Play a policy in an environment and record the data.

Usage:

python play_policy.py -p <path-to-run> -sd --video

This command will save data and record a video of the playback using an existing run folder.

"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Play a policy in WheeledLab.")
# These arguments assume that a run folder can be found
parser.add_argument('-p', "--run-path", type=str, default=None, help="Path to run folder")
parser.add_argument("--checkpoint", type=int, default=None, help="Checkpoint to load")
# If no run folder, the task and policy model must be provided
parser.add_argument("--task", type=str, default=None, help="Task name. Overrides run config env if provided")
parser.add_argument("--policy-path", type=str, default=None, help="Path to policy file.")
# Playback
parser.add_argument("--steps", type=int, default=200, help="Length of recorded video in steps")
# Logging
parser.add_argument('-sd', "--save-data", action="store_true", help="Save episode data")
parser.add_argument("--video", action="store_true", help="Record video of the playback")
parser.add_argument("--log-dir", type=str, default="playback/",
                    help="Directory to save logs. If run path is provided, this is ignored.")
parser.add_argument("--play-name", type=str, default="play-name", help="Name of the playback")
parser.add_argument("--play-cfg", action="store_true",
                    help="Use the registered play_env_cfg_entry_point (no rewards / no terminations) "
                         "instead of the train env cfg. Useful for visual inspection.")
parser.add_argument("--dump-camera", action="store_true",
                    help="Dump the per-env tiled camera frames the policy consumes to mp4. "
                         "Writes <play-name>-camera-rgb.mp4 (raw RGB) and "
                         "<play-name>-camera-policy.mp4 (cropped + grayscale, the deterministic "
                         "part of the policy obs — augmentations are skipped so frames are comparable).")
parser.add_argument("--dump-camera-env-id", type=int, default=0,
                    help="Which env's camera to dump when --dump-camera is set (default: 0).")

simulation_app, args_cli = startup(parser=parser)
### Extract task_name and agent_cfg from run_config.pkl ###

# Validate arguments
if args_cli.run_path is None:
    if args_cli.task is None and args_cli.policy_path is None:
        raise ValueError("Either path to run directory or task/policy must be provided.")

import os
import gymnasium as gym
import time
import torch
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import load_pickle
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction


# Resolve paths
FROM_RUN = args_cli.run_path is not None
if FROM_RUN: # Load paths for run folder

    # Load run config
    path_to_run_cfg_pkl = os.path.join(args_cli.run_path, "run_config.pkl")
    run_cfg: RunConfig = load_pickle(path_to_run_cfg_pkl) # load_yaml does not work on slices
    run_agent_cfg = run_cfg.agent
    task = run_cfg.env_setup.task_name if args_cli.task is None else args_cli.task
    agent_entry_point = None

    # Get policy path
    chkpt = args_cli.checkpoint if args_cli.checkpoint is not None else ".*"
    fp = os.path.abspath(args_cli.run_path)
    run_dirname = os.path.dirname(fp)
    run_folder = os.path.basename(fp)
    policy_resume_path = get_checkpoint_path(log_path=run_dirname, run_dir=run_folder,
                                        other_dirs=["models"], checkpoint=chkpt)

    # Set playback directory to be in run folder
    playback_dir = os.path.join(args_cli.run_path, "playback")

else:

    task = args_cli.task
    agent_entry_point = "rsl_rl_cfg_entry_point" # rsl is the only supported library for now
    playback_dir = args_cli.log_dir
    policy_resume_path = args_cli.policy_path


# Optionally swap to the play env cfg before Hydra resolves it.
# Hydra reads `env_cfg_entry_point` from the gym registry at main() call time,
# so we can repoint the spec's kwargs in-place here.
if args_cli.play_cfg:
    spec = gym.spec(task)
    if "play_env_cfg_entry_point" in spec.kwargs:
        spec.kwargs["env_cfg_entry_point"] = spec.kwargs["play_env_cfg_entry_point"]
        print(f"[INFO] --play-cfg: using play_env_cfg_entry_point for {task}.")
    else:
        print(f"[WARN] --play-cfg: no play_env_cfg_entry_point registered for {task}; using train cfg.")


@hydra_task_config(task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg): # TODO: Add SB3 config support

    if agent_cfg is None:
        agent_cfg = run_agent_cfg

    if not os.path.exists(playback_dir):
        os.makedirs(playback_dir)
    print(f"[INFO] Created playback directory: {playback_dir}")

    ####################################
    #### POLICY LOADING CODE ####
    ####################################

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": playback_dir,
            "step_trigger": lambda step: step % args_cli.steps == 0,
            "video_length": args_cli.steps, # updated to use args_cli
            "disable_logger": True,
            "name_prefix": args_cli.play_name,
        }
        print(f"[INFO] Recording video of playback to: {playback_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    ############################################
    ########### BEGIN PLAYBACK SETUP ###########
    ############################################

    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict())
    ppo_runner.load(policy_resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Data storage
    data = {
        'observations': [],
        'actions': [],
    }

    # Optional per-env tiled-camera frame capture (used by --dump-camera).
    # Stored as lists of CPU uint8 tensors so the GPU isn't holding playback
    # buffers across steps.
    cam_frames_rgb: list[torch.Tensor] = []   # each (H, W, 3) uint8
    cam_frames_policy: list[torch.Tensor] = []  # each (H', W) uint8 grayscale
    if args_cli.dump_camera:
        cam_sensor = env.unwrapped.scene.sensors["camera"]
        cam_env_id = args_cli.dump_camera_env_id
        print(f"[INFO] Will dump tiled camera frames for env {cam_env_id}.")

    ### PLAY POLICY ###

    # reset environment
    obs, _ = env.get_observations()
    # simulate environment
    for _ in tqdm(range(args_cli.steps), desc="Playing policy"):
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        # save data
        data['observations'].append(obs)
        data['actions'].append(actions)

        if args_cli.dump_camera:
            # Mirror the deterministic part of camera_data_rgb_flattened_aug:
            # crop top 1/3 (sky) + grayscale. Skip color jitter / blur / norm
            # so successive frames are visually comparable.
            rgb = cam_sensor.data.output["rgb"][cam_env_id].detach().cpu()  # (H, W, 3) uint8
            cam_frames_rgb.append(rgb)
            H = rgb.shape[0]
            cropped = rgb[H // 3:].float() / 255.0          # (H', W, 3)
            chw = cropped.permute(2, 0, 1)                  # (3, H', W)
            # ITU-R BT.601 luma weights — same as torchvision.transforms.Grayscale.
            gray = (0.2989 * chw[0] + 0.5870 * chw[1] + 0.1140 * chw[2])  # (H', W)
            cam_frames_policy.append((gray * 255.0).clamp(0, 255).byte())

    ###

    ########################
    ###### SAVE DATA #######
    ########################

    if args_cli.save_data:
        for key in data.keys():
            data[key] = torch.stack(data[key], dim=0)
        save_path = os.path.join(playback_dir, f"{args_cli.play_name}-rollouts.pt")
        torch.save(data, save_path)
        print(f"[INFO] Saved episode data to: {save_path}")

    if args_cli.dump_camera and len(cam_frames_rgb) > 0:
        # imageio + ffmpeg backend is already pulled in by gym.wrappers.RecordVideo.
        import imageio.v2 as imageio
        # Frame rate = policy step rate = 1 / (sim.dt * decimation).
        sim_cfg = env.unwrapped.cfg.sim
        dec = env.unwrapped.cfg.decimation
        fps = max(1, int(round(1.0 / (sim_cfg.dt * dec))))

        rgb_path = os.path.join(playback_dir, f"{args_cli.play_name}-camera-rgb.mp4")
        rgb_arr = torch.stack(cam_frames_rgb, dim=0).numpy()    # (T, H, W, 3) uint8
        imageio.mimsave(rgb_path, rgb_arr, fps=fps, codec="libx264")
        print(f"[INFO] Saved raw camera RGB to: {rgb_path}  ({rgb_arr.shape}, {fps} fps)")

        policy_path = os.path.join(playback_dir, f"{args_cli.play_name}-camera-policy.mp4")
        gray_arr = torch.stack(cam_frames_policy, dim=0)        # (T, H', W) uint8
        gray_rgb = gray_arr.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous().numpy()  # mp4 wants 3-ch
        imageio.mimsave(policy_path, gray_rgb, fps=fps, codec="libx264")
        print(f"[INFO] Saved policy-view (cropped+gray) to: {policy_path}  ({gray_arr.shape}, {fps} fps)")

    print("Done playing policy. Closing environment.")
    env.close()

if __name__ == "__main__":
    main()