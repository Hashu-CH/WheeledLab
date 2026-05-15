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
