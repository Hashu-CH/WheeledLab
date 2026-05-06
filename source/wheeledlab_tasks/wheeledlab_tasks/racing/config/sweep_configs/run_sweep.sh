#!/usr/bin/env bash
# Sequential architecture sweep for the racing task.
#
# For each YAML, sets WHEELEDLAB_RACING_CONFIG and invokes train_rl.py.
# Each run is its own python process, so the @lru_cache on the YAML loader
# does not bleed configs between runs.
#
# Usage:
#   bash run_sweep.sh                  # all five archs
#   ARCHS="mlp cnn" bash run_sweep.sh  # subset
#
# Override TRAIN_SCRIPT if your entrypoint lives elsewhere.

set -euo pipefail

SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SWEEP_DIR/../../../../../wheeledlab_rl/scripts/train_rl.py}"
RUN_CONFIG_NAME="${RUN_CONFIG_NAME:-RSS_RACING_CONFIG}"
ARCHS="${ARCHS:-mlp cnn rnn cnn_rnn mlp_rnn}"

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
    python "$TRAIN_SCRIPT" -r "$RUN_CONFIG_NAME"
done

echo "[sweep] done."
