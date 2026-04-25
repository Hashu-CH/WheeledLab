from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from wheeledlab_tasks.racing.config import CONFIG

# unpack config file 
_PPO = CONFIG["ppo"]
_POL = CONFIG["policy"]
_ALGO = _PPO["algorithm"]


@configclass
class MushrCNNGRUPolicyCfg:
    """Policy cfg for ActorCriticCNNGRU.

    This is essentially a new config and class injected into rsl_rl 
    on_policy_runner (see modified_rsl_rl). This version of rsl_rl 
    doesn't have cnn+rnn. 

    As always, modify parameters through racing_config.yaml (for racing task).
    """

    # rsl_rl on policy runner constructs module named class_name.pop() 
    class_name: str = "ActorCriticCNNGRU"
    init_noise_std: float = 1.0
    activation: str = "relu"
    actor_hidden_dims: list = (64, 64)
    critic_hidden_dims: list = (64, 64)
    # CNN frontend
    image_shape: tuple = (1, 40, 80)
    cnn_channels: list = (16, 32)
    cnn_kernel_sizes: list = (5, 3)
    cnn_strides: list = (2, 2)
    cnn_out_dim: int = 64
    # GRU
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1


@configclass
class MushrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training hyperparameters
    num_steps_per_env = _PPO["num_steps_per_env"]
    max_iterations = _PPO["max_iterations"]
    save_interval = _PPO["save_interval"]
    experiment_name = _PPO["experiment_name"]

    # CNN + GRU + MLP head. See actor_critic_cnn_gru.py.
    empirical_normalization = _PPO["empirical_normalization"]
    policy = MushrCNNGRUPolicyCfg(
        class_name=_POL["class_name"],
        init_noise_std=float(_POL["init_noise_std"]),
        activation=_POL["activation"],
        actor_hidden_dims=list(_POL["actor_hidden_dims"]),
        critic_hidden_dims=list(_POL["critic_hidden_dims"]),
        image_shape=tuple(_POL["image_shape"]),
        cnn_channels=list(_POL["cnn_channels"]),
        cnn_kernel_sizes=list(_POL["cnn_kernel_sizes"]),
        cnn_strides=list(_POL["cnn_strides"]),
        cnn_out_dim=int(_POL["cnn_out_dim"]),
        rnn_type=_POL["rnn_type"],
        rnn_hidden_dim=int(_POL["rnn_hidden_dim"]),
        rnn_num_layers=int(_POL["rnn_num_layers"]),
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
