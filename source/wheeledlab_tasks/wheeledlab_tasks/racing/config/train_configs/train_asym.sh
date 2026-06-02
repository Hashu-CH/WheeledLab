# Asymmetric-critic + distillation pipeline (teacher -> student), end to end.
#
# Runs the full sequential recipe under ONE shared run name so the three stages
# chain automatically and their log folders are grouped:
#
#   <RUN_NAME>_teacher   privileged MLP teacher          (RSS_RACING_PRIVILEGED_CONFIG)
#   <RUN_NAME>_distill   DAgger distill teacher->student (scripts/distill_policy.py)
#   <RUN_NAME>_finetune  RL fine-tune student w/ priv.   (RSS_RACING_ASYM_CONFIG)
#                        critic
#
# Usage:
#   bash train_asym.sh                          # full pipeline, auto run name
#   RUN_NAME=asym_v0 bash train_asym.sh         # name the run
#   TEACHER_RUN=/abs/path/to/priv_run \         # reuse an existing teacher,
#     RUN_NAME=asym_v0 bash train_asym.sh       #   skip teacher training
#   SMOKE=1 bash train_asym.sh                  # tiny/fast end-to-end sanity run
#
# Knobs (env vars, all optional):
#   RUN_NAME        base name for the three stages (default: asym_<timestamp>)
#   TEACHER_RUN     path to an existing privileged run folder; if set, the
#                   teacher-training stage is skipped and this is distilled
#   DAGGER_ITERS    distillation iterations            (default 200)
#   STEPS_PER_ITER  env steps per distill iteration     (default 256)
#   NUM_ENVS        parallel envs for distillation      (default 256)
#   SMOKE           if set, overrides the above with tiny values for a fast test
#   CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF, WHEELEDLAB_ROOT,
#   ISAACSIM_ROOT, CONDA_ENV                            (same as run_sweep.sh)
#
# wandb: ON by default — all three phases log plots (teacher/finetune via
# train_rl; distill logs distill/bc_mse + distill/beta). Folder names stay
# deterministic ($RUN_NAME_*) for load_run chaining because each stage exports
# WANDB_NAME (train_rl sets run_name = wandb.run.name, so this pins both). Set
# NO_WANDB=1 to disable (then run_name pins the folder directly).

set -euo pipefail

# ---- Environment preprocessing (mirrors run_sweep.sh) ----------------------
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

# ---- Paths -----------------------------------------------------------------
SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_DIR="$SWEEP_DIR/../../../../../wheeledlab_rl"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$RL_DIR/scripts/train_rl.py}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-$RL_DIR/scripts/distill_policy.py}"

# Racing-only logs path (train_configs -> config -> racing). Override with LOGS_DIR.
# train_rl resolves both its output and train.load_run against train.log.logs_dir,
# and distill writes under --logs-dir, so all three stages share this directory.
RACING_DIR="$(cd "$SWEEP_DIR/../.." && pwd)"
LOGS_DIR="${LOGS_DIR:-$RACING_DIR/logs}"
mkdir -p "$LOGS_DIR"

PRIV_YAML="$SWEEP_DIR/racing_privileged.yaml"
ASYM_YAML="$SWEEP_DIR/racing_asym.yaml"

for f in "$TRAIN_SCRIPT" "$DISTILL_SCRIPT" "$PRIV_YAML" "$ASYM_YAML"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: required file not found: $f" >&2
    exit 1
  fi
done

# ---- Run names + distillation knobs ----------------------------------------
RUN_NAME="${RUN_NAME:-asym_$(date +%Y%m%d_%H%M%S)}"
TEACHER_RUN_NAME="${RUN_NAME}_teacher"
DISTILL_RUN_NAME="${RUN_NAME}_distill"
FINETUNE_RUN_NAME="${RUN_NAME}_finetune"

if [[ -n "${SMOKE:-}" ]]; then
  DAGGER_ITERS="${DAGGER_ITERS:-2}"
  STEPS_PER_ITER="${STEPS_PER_ITER:-16}"
  NUM_ENVS="${NUM_ENVS:-16}"
  NO_WANDB="${NO_WANDB:-1}"   # SMOKE: default wandb off to avoid clutter
  echo "[train_asym] SMOKE mode: tiny end-to-end sanity run."
else
  DAGGER_ITERS="${DAGGER_ITERS:-200}"
  STEPS_PER_ITER="${STEPS_PER_ITER:-256}"
  NUM_ENVS="${NUM_ENVS:-256}"
  NO_WANDB="${NO_WANDB:-0}"   # wandb on by default (set NO_WANDB=1 to disable)
fi

# wandb is kept ON by default so all three phases produce plots. We preserve
# deterministic folder names (needed for load_run chaining) by exporting
# WANDB_NAME per stage: train_rl sets run_name = wandb.run.name, so WANDB_NAME
# pins both the wandb run AND the log folder. With NO_WANDB=1 we instead pass
# no_wandb/--no-wandb and the explicit run_name pins the folder directly.
if [[ "$NO_WANDB" == "1" ]]; then
  TRAIN_WANDB_ARG="train.log.no_wandb=true"   # single token, expanded unquoted
  DISTILL_WANDB_ARG="--no-wandb"
else
  TRAIN_WANDB_ARG=""
  DISTILL_WANDB_ARG=""
fi

echo "================================================================"
echo "[train_asym] RUN_NAME=$RUN_NAME"
echo "[train_asym] logs dir=$LOGS_DIR"
echo "================================================================"

# ---- Stage 0: teacher (privileged) -----------------------------------------
# Skipped when TEACHER_RUN points at an existing privileged run folder.
if [[ -n "${TEACHER_RUN:-}" ]]; then
  echo "[train_asym] Using existing teacher run: $TEACHER_RUN  (skipping teacher training)"
else
  echo "[train_asym] === Stage 0: train privileged teacher -> $TEACHER_RUN_NAME ==="
  WANDB_NAME="$TEACHER_RUN_NAME" WHEELEDLAB_RACING_CONFIG="$PRIV_YAML" \
    python "$TRAIN_SCRIPT" --headless -r RSS_RACING_PRIVILEGED_CONFIG \
      train.log.run_name="$TEACHER_RUN_NAME" train.log.logs_dir="$LOGS_DIR" \
      $TRAIN_WANDB_ARG
  TEACHER_RUN="$LOGS_DIR/$TEACHER_RUN_NAME"
fi

if [[ ! -d "$TEACHER_RUN/models" ]]; then
  echo "ERROR: teacher run has no models/ dir: $TEACHER_RUN" >&2
  exit 1
fi

# ---- Stage A: distill teacher -> camera student ----------------------------
echo "[train_asym] === Stage A: distill -> $DISTILL_RUN_NAME ==="
WHEELEDLAB_RACING_CONFIG="$ASYM_YAML" \
  python "$DISTILL_SCRIPT" --headless \
    --teacher-run "$TEACHER_RUN" \
    --run-name "$DISTILL_RUN_NAME" \
    --logs-dir "$LOGS_DIR" \
    --num-envs "$NUM_ENVS" \
    --dagger-iters "$DAGGER_ITERS" \
    --steps-per-iter "$STEPS_PER_ITER" \
    $DISTILL_WANDB_ARG

if [[ ! -d "$LOGS_DIR/$DISTILL_RUN_NAME/models" ]]; then
  echo "ERROR: distillation produced no checkpoint at $LOGS_DIR/$DISTILL_RUN_NAME/models" >&2
  exit 1
fi

# ---- Stage B: RL fine-tune the student with the privileged critic ----------
echo "[train_asym] === Stage B: RL fine-tune -> $FINETUNE_RUN_NAME ==="
WANDB_NAME="$FINETUNE_RUN_NAME" WHEELEDLAB_RACING_CONFIG="$ASYM_YAML" \
  python "$TRAIN_SCRIPT" --headless -r RSS_RACING_ASYM_CONFIG \
    train.load_run="$DISTILL_RUN_NAME" \
    train.log.run_name="$FINETUNE_RUN_NAME" train.log.logs_dir="$LOGS_DIR" \
    $TRAIN_WANDB_ARG

echo "================================================================"
echo "[train_asym] done."
echo "[train_asym]   teacher : $TEACHER_RUN"
echo "[train_asym]   distill : $LOGS_DIR/$DISTILL_RUN_NAME"
echo "[train_asym]   finetune: $LOGS_DIR/$FINETUNE_RUN_NAME"
echo "================================================================"
