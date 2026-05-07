#!/usr/bin/env bash
# testing script 
#
# Usage:
#
#   bash source/wheeledlab_tasks/wheeledlab_tasks/racing/config/eval_configs/eval.sh \
#     /path/to/run1 /path/to/run2 --seeds 0 1 2

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

# If user passed bare run paths (no -p), prepend -p to each path-looking arg.
# Heuristic: any arg that starts with / or ./ and is a directory.
ARGS=()
for a in "$@"; do
  if [[ -d "$a" ]]; then
    ARGS+=("-p" "$a")
  else
    ARGS+=("$a")
  fi
done

python source/wheeledlab_rl/scripts/eval_racing.py --headless "${ARGS[@]}"
