import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from .. import mdp_sensors


@configclass
class RacingObsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """
        [camera, vx, vy, vz, wx, wy, wz, action1(vel), action2(steering)]
        """
        camera = ObsTerm(
            func=mdp_sensors.camera_data_rgb_flattened_aug,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-.1, n_max=.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-.1, n_max=.1))
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-1., 1.),
            noise=Unoise(-.1, .1),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
