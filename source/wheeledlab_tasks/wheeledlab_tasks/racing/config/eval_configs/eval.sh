#!/usr/bin/env bash
# testing script
#
# Usage:
#   bash ./source/wheeledlab_tasks/wheeledlab_tasks/racing/config/eval_configs/eval.sh -p ./source/wheeledlab_rl/logs/cnn ./source/wheeledlab_rl/logs/cnnrnn --num-tracks 256 --seeds 0 1 2

set -eo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WHEELEDLAB_ROOT="${WHEELEDLAB_ROOT:-$HOME/WheeledLab}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"
CONDA_ENV="${CONDA_ENV:-WL}"

export WHEELEDLAB_RACING_CONFIG="${WHEELEDLAB_RACING_CONFIG:-$WHEELEDLAB_ROOT/source/wheeledlab_tasks/wheeledlab_tasks/racing/config/eval_configs/racing_eval.yaml}"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$ISAACSIM_ROOT"
# shellcheck disable=SC1091
source setup_conda_env.sh

cd "$WHEELEDLAB_ROOT"

python source/wheeledlab_rl/scripts/eval_racing.py --headless "$@"
