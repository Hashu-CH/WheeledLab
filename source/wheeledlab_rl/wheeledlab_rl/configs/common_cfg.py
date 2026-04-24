import os
import random
from omegaconf import MISSING
from typing import Dict, Any, Optional

from isaaclab.utils import configclass

from wheeledlab_rl import WHEELEDLAB_RL_LOGS_DIR


@configclass
class LogConfig:
    '''
    Configuration for logging during training
    '''
    logs_dir: str = WHEELEDLAB_RL_LOGS_DIR              # Directory for creating logs (if log_dir is not provided)
    no_log: bool = False                                # Disable logging
    log_every: int = 10                                 # Log every n updates
    video: bool = True                                  # Record videos during training
    video_length: int = 500                             # Length of the recorded video (in steps)
    video_interval: int = 5000                          # Interval between video recordings (in steps)
    no_checkpoints: bool = False                        # Disable saving checkpoints
    checkpoint_every: int = 1000                        # Save checkpoint every n updates
    no_wandb: bool = False                              # Disable wandb logging
    wandb_project: str = "WheeledLab"                   # Wandb project name
    test_mode: bool = False                             # Test mode (disable logging, wandb, video, checkpoints). Overrides other flags
    model_save_dirname: str = "models"                  # Path to save the model under log_dir
    video_resolution: tuple[int, int] = (1280, 720)     # Resolution of the recorded video, Width x Height
    video_crf: int = 30                                 # Constant Rate Factor for video compression
    # Policy-camera recorder: dumps the per-env tiled camera obs to wandb on the
    # same trigger as the viewer video. Useful when the viewer perspective is too
    # zoomed-out to see the agent (e.g. racing task with many envs).
    log_policy_camera: bool = False                     # Enable PolicyCameraRecorder wrapper
    policy_camera_sensor_name: str = "camera"           # Key in env.scene.sensors to read RGB from
    policy_camera_env_id: int = 0                       # Which env's camera to record
    policy_camera_fps: int = 10                         # Playback fps for the saved mp4
    run_name: str = f"run-{random.randint(0, 1e7)}"     # Name of the run

    @property
    def run_log_dir(self):
        ''' Directory for run log '''
        return os.path.join(self.logs_dir, self.run_name)

    @property
    def model_save_path(self):
        return os.path.join(self.run_log_dir, self.model_save_dirname)


@configclass
class TrainConfig:
    ''' Configuration for training '''
    seed: int = 0                       # Seed used for the environment
    device: str = "cuda:0"              # Device to use
    load_run: Optional[str] = None      # Load a previously trained model
    load_run_checkpoint: int = 0        # Load a specific checkpoint from a previously trained model

    log: LogConfig = LogConfig()

@configclass
class EnvSetup:
    num_envs: int = 1024                # Number of environments to simulate
    task_name: str = MISSING            # Name of the task

@configclass
class AgentSetup:
    # TODO: Is this necessary?
    entry_point: str = "rsl_rl_cfg_entry_point" # Entry point key to resolve the agent's configuration file


### PUT ABOVE CONFIGS TOGETHER ###

@configclass
class RunConfig:

    train: TrainConfig = TrainConfig()
    env_setup: EnvSetup = EnvSetup()
    agent_setup: AgentSetup = AgentSetup()

    # --- RESOLVED FROM ISAACLAB REGISTRY --- #
    env: Any = MISSING
    agent: Any = MISSING
