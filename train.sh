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
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_RACING_CONFIG

