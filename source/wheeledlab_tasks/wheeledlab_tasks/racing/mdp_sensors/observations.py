"""
Contains sensor helpers for the Racing Task. Fed into observation mdp
where the overall Observation Config is defined.

TODO: wire the hyperparams from somewhere else.

Notes:

- Current domain randomizations applied: Jitter and Blur
- No depth camera or RGB utilized -- images are converted to grayscale
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

# Defining all the transformations to apply to camera inputs
grayscale = transforms.Grayscale()
gray_normalize = transforms.Normalize([0.5], [0.5])
color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.8, hue=0.5)
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
    normalized_imgs = gray_normalize(grayscale(images))
    return normalized_imgs.reshape(B, -1)
