"""
Wraps observation terms for the racing task.

Conceputally, this defines the obserrvations that are exposed to 
our policy. Thse are NOT the sensors. The observation terms read from 
the sensors via the scene config. 

Notes:

- Proprioception is defined here -- randomization toggle via the 
  enable_corruption flag. Camera observations are defined elsewhere 
  and are not affected by the flag.
- TODO: It would be useful to define in racing/config a file that 
  houses all hyper paramters for a one-location site to tune.
  Right now hardcoded values are scattered throughout.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .. import mdp_sensors
from ..config import CONFIG

_OBS = CONFIG["observations"]


def _unoise(key: str) -> Unoise:
    lo, hi = _OBS[key]
    return Unoise(n_min=float(lo), n_max=float(hi))


# ---------------------------------------------------------------------------
# Config Definitions (passed to racing_env)
# ---------------------------------------------------------------------------
@configclass
class RacingObsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """
        VISION + proprio: [camera (3×40×80 RGB flattened), base_lin_vel(3),
        base_ang_vel(3), last_action(2)].

        Term order matters: the camera term comes FIRST so the CNN encoder
        (ActorCriticCNN/CNNGRU) can split obs as [image | proprio] by image_shape
        numel — the proprio scalars are whatever follows the image. So image_shape
        stays (3,40,80); no encoder change is needed when proprio is added/removed.
        """
        camera = ObsTerm(
            func=mdp_sensors.camera_data_rgb_flattened_aug,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=_unoise("base_lin_vel_noise"))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=_unoise("base_ang_vel_noise"))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=tuple(_OBS["last_action_clip"]),
            noise=_unoise("last_action_noise"),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = bool(_OBS["enable_corruption"])
            self.concatenate_terms = bool(_OBS["concatenate_terms"])

    policy: PolicyCfg = PolicyCfg() # policy group term
    # can define actor as well differently


@configclass
class RacingPrivilegedObsCfg:
    """Privileged ("perfect perception") observations for the racing task.

    Replaces the camera with the exact cone positions (car frame) drawn from the
    track cache, plus the same proprioception the vision policy uses. Train an MLP
    (ActorCritic) on this to test whether the controller/planner can race given
    flawless perception — if it succeeds here but the camera policy fails, the
    bottleneck is perception, not control. See mdp_sensors.cones_in_car_frame.
    """
    @configclass
    class PolicyCfg(ObsGroup):
        cones = ObsTerm(
            func=mdp_sensors.cones_in_car_frame,
            params={
                "num_per_side": int(_OBS["cones_num_per_side"]),
                "lookahead_m": float(_OBS["cones_lookahead_m"]),
            },
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=_unoise("base_lin_vel_noise"))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=_unoise("base_ang_vel_noise"))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=tuple(_OBS["last_action_clip"]),
            noise=_unoise("last_action_noise"),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = bool(_OBS["enable_corruption"])
            self.concatenate_terms = bool(_OBS["concatenate_terms"])

    policy: PolicyCfg = PolicyCfg()