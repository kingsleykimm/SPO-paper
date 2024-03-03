#!/usr/bin/env bash
#SBATCH --job-name="spo-test-run"
#SABTCH --partition=gpu
#SBATCH --time=0:30:00
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn
module load anaconda
conda deactivate
conda activate rl
python config.py --env_name control:quadruped:walk --preference_function maximum_reward --iterations 30