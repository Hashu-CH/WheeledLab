"""CNN actor-critics using rsl_rl ver 2.3.3.

For racing: [(1,40,80) camera, (vel, ang, a_t-1) proprioceptive data]

Note:
- actor and critic share the same CNN. Will need to split if future
  distillation or asymmetric critic is done.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


def _build_cnn(
    image_shape: tuple[int, int, int],
    channels: list[int],
    kernel_sizes: list[int],
    strides: list[int],
    out_dim: int,
    activation: str,
) -> nn.Module:
    """Conv stack -> flatten -> linear projection to `out_dim`.

    activation string supports anything usable in rsl_rl : relu 
    elu, tanh, sigmoid, leakyrelu, etc
    """
    if not (len(channels) == len(kernel_sizes) == len(strides)):
        raise ValueError(
            "cnn_channels, cnn_kernel_sizes, cnn_strides must have equal length; "
            f"got {len(channels)}, {len(kernel_sizes)}, {len(strides)}"
        )

    # build conv model from params
    channel_in = image_shape[0]
    conv_layers: list[nn.Module] = []
    for channel_out, k, s in zip(channels, kernel_sizes, strides):
        conv_layers.append(nn.Conv2d(channel_in, channel_out, kernel_size=k, stride=s))
        conv_layers.append(resolve_nn_activation(activation))
        channel_in = channel_out
    conv_layers.append(nn.Flatten())
    conv = nn.Sequential(*conv_layers)

    # compute flat dimension on fly 
    with torch.no_grad():
        flat_dim = conv(torch.zeros(1, *image_shape)).shape[-1]

    # final conv model with last lin output
    return nn.Sequential(conv, nn.Linear(flat_dim, out_dim), resolve_nn_activation(activation))


class ActorCriticCNN(ActorCritic):
    """
    CNN + MLP actor-critic, no recurrence.

    Sibling of ActorCriticCNNGRU for ablation: same image+proprio split,
    same CNN encoder, but the encoded vector flows directly into the MLP
    heads without a memory module.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (1, 40, 80),
        cnn_channels: list[int] = (16, 32),
        cnn_kernel_sizes: list[int] = (5, 3),
        cnn_strides: list[int] = (2, 2),
        cnn_out_dim: int = 64,
        actor_hidden_dims: list[int] = (64, 64),
        critic_hidden_dims: list[int] = (64, 64),
        activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # shape checks before parent init so a bad cfg fails fast
        image_shape_t = tuple(image_shape)
        n_img = int(image_shape_t[0] * image_shape_t[1] * image_shape_t[2])
        if n_img > num_actor_obs:
            raise ValueError(
                f"ActorCriticCNN: image_shape numel ({n_img}) exceeds "
                f"num_actor_obs ({num_actor_obs}). Obs layout likely changed — "
                f"update image_shape in the policy cfg."
            )
        if n_img > num_critic_obs:
            raise ValueError(
                f"ActorCriticCNN: image_shape numel ({n_img}) exceeds "
                f"num_critic_obs ({num_critic_obs})."
            )

        # MLP head input = cnn projection + remaining proprio scalars
        num_proprio_a = num_actor_obs - n_img
        num_proprio_c = num_critic_obs - n_img
        enc_dim_a = cnn_out_dim + num_proprio_a
        enc_dim_c = cnn_out_dim + num_proprio_c

        super().__init__(
            num_actor_obs=enc_dim_a,
            num_critic_obs=enc_dim_c,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        self.image_shape = image_shape_t
        self._n_img = n_img

        # shared cnn for actor and critic
        self.cnn = _build_cnn(
            self.image_shape,
            list(cnn_channels),
            list(cnn_kernel_sizes),
            list(cnn_strides),
            cnn_out_dim,
            activation,
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        leading, D = obs.shape[:-1], obs.shape[-1]
        flat = obs.reshape(-1, D)
        img = flat[:, : self._n_img].reshape(-1, *self.image_shape)
        proprio = flat[:, self._n_img:]
        feat = self.cnn(img)
        enc = torch.cat([feat, proprio], dim=-1)
        return enc.reshape(*leading, -1)

    def act(self, observations, **kwargs):
        return super().act(self._encode(observations))

    def act_inference(self, observations):
        return super().act_inference(self._encode(observations))

    def evaluate(self, critic_observations, **kwargs):
        return super().evaluate(self._encode(critic_observations))


class ActorCriticCNNGRU(ActorCritic):
    """
    sub classes rsl_rl's actor critic to add cnn image encoder 
    
    Passes observations through the cnn encoder defined above,
    Then to GRU encoder, GRU latent rep is passed to finda mlp
    
    copies the pattern from rsl_rl's actual recurrent network.
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        image_shape: tuple[int, int, int] = (1, 40, 80),
        cnn_channels: list[int] = (16, 32),
        cnn_kernel_sizes: list[int] = (5, 3),
        cnn_strides: list[int] = (2, 2),
        cnn_out_dim: int = 64,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        actor_hidden_dims: list[int] = (64, 64),
        critic_hidden_dims: list[int] = (64, 64),
        activation: str = "relu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # define MLP head
        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )

        # shape checks
        self.image_shape = tuple(image_shape)
        self._n_img = int(self.image_shape[0] * self.image_shape[1] * self.image_shape[2])

        if self._n_img > num_actor_obs:
            raise ValueError(
                f"ActorCriticCNNGRU: image_shape numel ({self._n_img}) exceeds "
                f"num_actor_obs ({num_actor_obs}). Obs layout likely changed — "
                f"update image_shape in the policy cfg."
            )
        if self._n_img > num_critic_obs:
            raise ValueError(
                f"ActorCriticCNNGRU: image_shape numel ({self._n_img}) exceeds "
                f"num_critic_obs ({num_critic_obs})."
            )

        # build shared cnn for both actor and critic.
        # TODO: whole network uses same wired 'activation'
        self.cnn = _build_cnn(
            self.image_shape,
            list(cnn_channels),
            list(cnn_kernel_sizes),
            list(cnn_strides),
            cnn_out_dim,
            activation,
        )

        # concatenate size of image observation and cnn latent rep
        num_proprio_a = num_actor_obs - self._n_img
        num_proprio_c = num_critic_obs - self._n_img
        enc_dim_a = cnn_out_dim + num_proprio_a
        enc_dim_c = cnn_out_dim + num_proprio_c

        # separate gru memory encoders for both actor and critic 
        self.memory_a = Memory(
            enc_dim_a, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )
        self.memory_c = Memory(
            enc_dim_c, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim
        )

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Obs are Batch, sensor Dim, (optionally seq len) 
        This is the cnn step that is preprocess for rest of overriden steps
        """
        # shape seq len ppo update into (b, d) size or (b,t,d)
        leading, D = obs.shape[:-1], obs.shape[-1]
        flat = obs.reshape(-1, D)  
        
        # split camera dims from proprio
        img = flat[:, : self._n_img].reshape(-1, *self.image_shape)
        proprio = flat[:, self._n_img:]
        feat = self.cnn(img)                                # (N, cnn_out_dim)
        enc = torch.cat([feat, proprio], dim=-1)            # (N, enc_dim)
        return enc.reshape(*leading, -1)                    # (..., enc_dim)

    # ------------------------------------------------------------------
    # Actor Critic calls using the CNN encode step as preprocess
    # ------------------------------------------------------------------
    def act(self, observations, masks=None, hidden_states=None):
        enc = self._encode(observations)
        # __call__ to memory does forward pass to GRU
        hid = self.memory_a(enc, masks, hidden_states)
        # training mask zeros terminates epsiodes (no gradient flow) 
        return super().act(hid.squeeze(0))

    def act_inference(self, observations):
        # inference only
        enc = self._encode(observations)
        hid = self.memory_a(enc)
        return super().act_inference(hid.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """value estimate computation"""
        enc = self._encode(critic_observations)
        hid = self.memory_c(enc, masks, hidden_states)
        return super().evaluate(hid.squeeze(0))

    def reset(self, dones=None):
        # dones = mask from act step
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states
