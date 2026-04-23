#~/WheeledLab/train.sh 
# train script for convenience
set -e

# nvidia-smi check before to see which card is open -- if open
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate WL

cd ~/isaacsim
source setup_conda_env.sh

cd ~/WheeledLab

# Point to the canonical racing hyperparameter YAML. Override for experiments by
# exporting WHEELEDLAB_RACING_CONFIG=/path/to/my_variant.yaml before running.
export WHEELEDLAB_RACING_CONFIG="${WHEELEDLAB_RACING_CONFIG:-$HOME/WheeledLab/source/wheeledlab_tasks/wheeledlab_tasks/racing/config/racing_config.yaml}"

python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_RACING_CONFIG

