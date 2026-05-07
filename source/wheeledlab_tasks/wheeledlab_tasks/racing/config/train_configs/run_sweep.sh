# training script
# Each run takes like an hour or two to converge. 
# It's helpful to be able to queue multiple at once. 
#
# Usage:
#   bash run_sweep.sh                  # both archs (cnn, cnn_rnn)
#   ARCHS=cnn bash run_sweep.sh        # subset (single arch)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WHEELEDLAB_ROOT="${WHEELEDLAB_ROOT:-$HOME/WheeledLab}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"
CONDA_ENV="${CONDA_ENV:-WL}"

set +u
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$ISAACSIM_ROOT"
# shellcheck disable=SC1091
source setup_conda_env.sh
set -u

cd "$WHEELEDLAB_ROOT"

# ---- Sweep config ----------------------------------------------------------
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SWEEP_DIR/../../../../../wheeledlab_rl/scripts/train_rl.py}"
RUN_CONFIG_NAME="${RUN_CONFIG_NAME:-RSS_RACING_CONFIG}"
ARCHS="${ARCHS:-cnn cnn_rnn}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: train script not found at $TRAIN_SCRIPT" >&2
  echo "Set TRAIN_SCRIPT=/path/to/train_rl.py" >&2
  exit 1
fi

for arch in $ARCHS; do
  yaml="$SWEEP_DIR/racing_${arch}.yaml"
  if [[ ! -f "$yaml" ]]; then
    echo "ERROR: missing config $yaml" >&2
    exit 1
  fi
  echo "================================================================"
  echo "[sweep] arch=$arch  config=$yaml"
  echo "================================================================"
  WHEELEDLAB_RACING_CONFIG="$yaml" \
    python "$TRAIN_SCRIPT" --headless -r "$RUN_CONFIG_NAME"
done

echo "[sweep] done."
