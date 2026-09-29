#!/bin/bash -l
# Regularisation sweep of the tau_ref inversion (tau_sweep.py) on one Alex
# GPU. Submit from an Alex login node, in the repo root:
#   sbatch experiments/continuix/tau_sweep.sh --glaciers G03 G05

#SBATCH --job-name=tau_sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/tau_sweep_%j.out

module load python
conda activate igm32-frost

python experiments/continuix/tau_sweep.py --run "$@"
