from isaaclab.utils import configclass

from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)
from wheeledlab_tasks.racing.config import CONFIG as _RACING_CFG

_RACING_RUN = _RACING_CFG["run"]
_RACING_LOG = _RACING_CFG["logging"]

@configclass
class RSS_DRIFT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrDriftRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_VISUAL_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=512,
        task_name="Isaac-MushrVisualRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_ELEV_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrElevationRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_RACING_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=_RACING_RUN["num_envs"],
        task_name="Isaac-MushrRacingRL-v0",
    )
    train = RLTrainConfig(
        num_iterations=_RACING_RUN["num_iterations"],
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video=_RACING_LOG["video"],
            video_length=_RACING_LOG["video_length"],
            video_interval=_RACING_LOG["video_interval"],
            video_resolution=tuple(_RACING_LOG["video_resolution"]),
            video_crf=_RACING_LOG["video_crf"],
            log_every=_RACING_LOG["log_every"],
            checkpoint_every=_RACING_LOG["checkpoint_every"],
            no_checkpoints=_RACING_LOG["no_checkpoints"],
            no_wandb=_RACING_LOG["no_wandb"],
            wandb_project=_RACING_LOG["wandb_project"],
            no_log=_RACING_LOG["no_log"],
            test_mode=_RACING_LOG["test_mode"],
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )
