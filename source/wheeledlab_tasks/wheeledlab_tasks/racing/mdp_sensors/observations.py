"""
Contains sensor helpers for the Racing Task. Fed into observation mdp
where the overall Observation Config is defined.

TODO: wire the hyperparams from somewhere else.

Notes:

- Domain randomizations applied: ColorJitter (hue covers cone color variation), GaussianBlur
- RGB output (3 channels) — color is the primary left/right boundary cue (orange vs blue cones)
"""


from __future__ import annotations

import torch
import torchvision.transforms as transforms
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.envs.mdp import *

from wheeledlab.envs.mdp import root_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Sensor Helpers
# ---------------------------------------------------------------------------

# Hue jitter ±0.3 covers cone color variation (dirty/faded cones, lighting shifts).
# Grayscale step removed — color is load-bearing for left (orange) vs right (blue) boundary.
rgb_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.8, hue=0.3)
gaussian_blur = transforms.GaussianBlur(5, sigma=(0.1, 5.0))


def camera_data_rgb_flattened_aug(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """See top of file"""
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output["rgb"]
    B, H, W, C = images.shape
    images = images[:, H // 3:, :, :]
    images = images.permute(0, 3, 1, 2).float() / 255.
    images = color_jitter(images)
    images = gaussian_blur(images)
    normalized_imgs = rgb_normalize(images)
    return normalized_imgs.reshape(B, -1)


# ---------------------------------------------------------------------------
# Privileged ("perfect perception") observation
# ---------------------------------------------------------------------------
def cones_in_car_frame(
    env: ManagerBasedEnv,
    num_per_side: int = 5,
    lookahead_m: float = 5.0,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Next K cone gates ahead (by centerline arc-length), in the car body frame.

    Privileged track perception: bypasses the camera, reads the exact authored
    cone positions from the track cache (env.scene.terrain), and returns the
    next `num_per_side` gates on each side (orange left, then blue right) within
    `lookahead_m` arc-length ahead, as car-frame (x=forward, y=left) pairs.
    Ordering is by arc-length, not Euclidean distance, so the gate sequence is
    stable as the car yaws. Empty slots are `pad_value`. Output dim = 4 * K.

    Use to test whether the policy/controller can race given flawless
    perception; if it can, perception is the bottleneck.
    """
    terrain = env.scene.terrain
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pos_xy = root_pos_w(env)[:, :2]
    yaw = root_euler_xyz(env)[:, 2]
    return terrain.cones_in_car_frame(
        root_pos_xy, yaw, env_ids, num_per_side, lookahead_m, pad_value
    )
