#!/bin/bash -l
# Inversion + forward run check (run_inversion_forward.py) on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/inversion_forward/run_inversion_forward.sh

#SBATCH --job-name=inv_forward
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=data/results/log/inv_forward_%j.out

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

python experiments/inversion_forward/run_inversion_forward.py "$@"
