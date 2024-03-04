#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=bii_dsc_community
#SBATCH --mem=32GB



source ~/.bashrc

module load cuda cudnn
module load anaconda
conda deactivate
conda activate rl
python config.py --env_name control:quadruped:walk --preference_function maximum_reward --iterations 30