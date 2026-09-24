#!/bin/bash -l
# End-to-end test (tests/test_end_to_end.py) on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch tests/run_end_to_end.sh

#SBATCH --job-name=frost_e2e
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=data/results/log/frost_e2e_%j.out

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

nvidia-smi
FROST_E2E=1 python -m pytest tests/test_end_to_end.py -v -s -rA
