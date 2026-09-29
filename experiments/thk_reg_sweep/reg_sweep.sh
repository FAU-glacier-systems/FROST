#!/bin/bash -l
# Regularisation sweep of the thickness inversion (reg_sweep.py) on one Alex
# GPU. Submit from an Alex login node, in the repo root:
#   sbatch experiments/thk_reg_sweep/reg_sweep.sh

#SBATCH --job-name=reg_sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/reg_sweep_%j.out

module load python
conda activate igm32-frost

python experiments/thk_reg_sweep/reg_sweep.py --run "$@"
