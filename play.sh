#~/WheeledLab/play.sh
# Play a trained racing policy with the PlayEnvCfg variant (no rewards,
# no terminations) and record a viewer-perspective video. 
#
# Usage:
#   ./play.sh <run-path> [extra play_policy.py args]
#   ./play.sh source/wheeledlab_rl/logs/run-<id>

# Examples:
#   ./play.sh logs/run-1234567
#   ./play.sh logs/run-1234567 --steps 1000 --checkpoint 950

set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <run-path> [extra play_policy.py args]"
  exit 1
fi

RUN_PATH="$1"
shift

# Match train.sh environment setup so the play env matches what was trained.
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate WL

cd ~/isaacsim
source setup_conda_env.sh

cd ~/WheeledLab

# Same YAML resolution as train.sh — keeps terrain params consistent.
export WHEELEDLAB_RACING_CONFIG="${WHEELEDLAB_RACING_CONFIG:-$HOME/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/racing/config/racing_config.yaml}"

# --play-cfg routes to MushrRacingPlayEnvCfg (no rewards/terminations).
# --video records the viewer perspective to <run-path>/playback/.
# --dump-camera dumps the per-env tiled camera frames the policy actually
# consumes (raw RGB + cropped/grayscale "policy view") as separate mp4s.
# Override which env is dumped with --dump-camera-env-id <int>.
python source/wheeledlab_rl/scripts/play_policy.py \
    -p "$RUN_PATH" \
    --play-cfg \
    --video \
    --dump-camera \
    --steps 500 \
    "$@"
