from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from wheeledlab_tasks.racing.config import CONFIG

_PPO = CONFIG["ppo"]
_POL = CONFIG["policy"]
_ALGO = _PPO["algorithm"]


@configclass
class MushrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training hyperparameters
    num_steps_per_env = _PPO["num_steps_per_env"]
    max_iterations = _PPO["max_iterations"]
    save_interval = _PPO["save_interval"]
    experiment_name = _PPO["experiment_name"]

    # Policy architecture hyperparameter
    empirical_normalization = _PPO["empirical_normalization"]
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=float(_POL["init_noise_std"]),
        actor_hidden_dims=list(_POL["actor_hidden_dims"]),
        critic_hidden_dims=list(_POL["critic_hidden_dims"]),
        activation=_POL["activation"],
    )

    # PPO algo hyperparameter
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=float(_ALGO["value_loss_coef"]),
        use_clipped_value_loss=_ALGO["use_clipped_value_loss"],
        clip_param=float(_ALGO["clip_param"]),
        entropy_coef=float(_ALGO["entropy_coef"]),
        num_learning_epochs=_ALGO["num_learning_epochs"],
        num_mini_batches=_ALGO["num_mini_batches"],
        learning_rate=float(_ALGO["learning_rate"]),
        schedule=_ALGO["schedule"],
        gamma=float(_ALGO["gamma"]),
        lam=float(_ALGO["lam"]),
        desired_kl=float(_ALGO["desired_kl"]),
        max_grad_norm=float(_ALGO["max_grad_norm"]),
    )
