#!/bin/bash -l
# IGM velocities for the provided ContinuIX thickness (thk_forward.py) on one
# Alex GPU. Submit from an Alex login node, in the repo root:
#   sbatch experiments/continuix/thk_forward.sh

#SBATCH --job-name=thk_forward
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/thk_forward_%j.out

module load python
conda activate igm32-frost

python experiments/continuix/thk_forward.py --run "$@"
