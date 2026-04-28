"""
Loader for the racing task's YAML hyperparameter file.

The YAML is resolved in this order:
  1. $WHEELEDLAB_RACING_CONFIG if set
  2. the sibling `racing_config.yaml` in this directory

Used by train.sh
"""

import os
import yaml
from functools import lru_cache


# just specify new path in train.sh to point to custom yaml file for large tunes
ENV_VAR = "WHEELEDLAB_RACING_CONFIG"
_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "racing_config.yaml")
_REQUIRED_SECTIONS = (
    "run", "logging", "ppo", "policy", "env", "terrain",
    "rewards", "events", "observations",
)


def get_config_path() -> str:
    return os.environ.get(ENV_VAR, _DEFAULT_PATH)


@lru_cache(maxsize=1)
def load_racing_config() -> dict:
    path = get_config_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Racing config YAML not found at {path}. "
            f"Set {ENV_VAR} to override or restore {_DEFAULT_PATH}."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    missing = [k for k in _REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise KeyError(f"Racing config at {path} missing sections: {missing}")
    print(f"[wheeledlab_tasks.racing] Loaded hyperparameters from {path}")
    return cfg
