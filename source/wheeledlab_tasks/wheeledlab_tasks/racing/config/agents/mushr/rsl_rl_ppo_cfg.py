import dataclasses

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from wheeledlab_tasks.racing.config import CONFIG

# unpack config file 
_PPO = CONFIG["ppo"]
_POL = CONFIG["policy"]
_ALGO = _PPO["algorithm"]


# ---------------------------------------------------------------------------
# Policy Config Definitions
# ---------------------------------------------------------------------------
@configclass
class MushrCNNGRUPolicyCfg:
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
class MushrMLPPolicyCfg:
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    activation: str = "relu"
    actor_hidden_dims: list = (64, 64)
    critic_hidden_dims: list = (64, 64)


@configclass
class MushrCNNPolicyCfg:
    class_name: str = "ActorCriticCNN"
    init_noise_std: float = 1.0
    activation: str = "relu"
    actor_hidden_dims: list = (64, 64)
    critic_hidden_dims: list = (64, 64)
    image_shape: tuple = (1, 40, 80)
    cnn_channels: list = (16, 32)
    cnn_kernel_sizes: list = (5, 3)
    cnn_strides: list = (2, 2)
    cnn_out_dim: int = 64


@configclass
class MushrRNNPolicyCfg:
    class_name: str = "ActorCriticRecurrent"
    init_noise_std: float = 1.0
    activation: str = "relu"
    actor_hidden_dims: list = (64, 64)
    critic_hidden_dims: list = (64, 64)
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1


# ---------------------------------------------------------------------------
# Policy dispatch — picks the wrapper configclass from YAML `class_name`
# so swapping architectures across runs needs no in-code edits.
# ---------------------------------------------------------------------------
_POLICY_CFG_BY_NAME = {
    "ActorCritic":          MushrMLPPolicyCfg,
    "ActorCriticCNN":       MushrCNNPolicyCfg,
    "ActorCriticRecurrent": MushrRNNPolicyCfg,
    "ActorCriticCNNGRU":    MushrCNNGRUPolicyCfg,
}


def _build_policy_cfg(pol: dict):
    name = pol["class_name"]
    if name not in _POLICY_CFG_BY_NAME:
        raise KeyError(
            f"Unknown policy class_name {name!r}. "
            f"Known: {sorted(_POLICY_CFG_BY_NAME)}"
        )
    cls = _POLICY_CFG_BY_NAME[name]
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in pol.items() if k in fields}
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Runner Config
# ---------------------------------------------------------------------------
@configclass
class MushrPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Training hyperparameters
    num_steps_per_env = _PPO["num_steps_per_env"]
    max_iterations = _PPO["max_iterations"]
    save_interval = _PPO["save_interval"]
    experiment_name = _PPO["experiment_name"]

    # Wrapper class is dispatched from YAML `policy.class_name`.
    empirical_normalization = _PPO["empirical_normalization"]
    policy = _build_policy_cfg(_POL)

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
