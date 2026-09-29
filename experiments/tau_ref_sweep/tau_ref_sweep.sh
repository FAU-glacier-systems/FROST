#!/bin/bash -l
# Inversions of the tau_ref sweep (tau_ref_sweep.py) on one Alex GPU.
# Submit from an Alex login node, in the repo root, one job per glacier:
#   sbatch experiments/tau_ref_sweep/tau_ref_sweep.sh --glaciers RGI2000-v7.0-G-11-01706

#SBATCH --job-name=tau_ref_sweep
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/tau_ref_sweep_%j.out

module load python
conda activate igm32-frost

python experiments/tau_ref_sweep/tau_ref_sweep.py --run "$@"
