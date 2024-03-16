#!/usr/bin/env bash
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=bii_dsc_community
#SBATCH --mem=16GB
#SBATCH -N 1
#SBATCH --cpus-per-task=4



module purge
module load cuda cudnn
module load anaconda
conda deactivate
conda activate rl
python config.py --env_name control:hopper:stand --preference_function maximum_reward --iterations 1000 --run_number hopper --agent sac