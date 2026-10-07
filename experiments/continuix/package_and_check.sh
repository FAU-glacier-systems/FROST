#!/bin/bash -l
# Rebuild the ContinuIX upload folder GROUP_FAU2 and check every file.
# Reads and writes all result files (several GB), so not on the login node.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/continuix/package_and_check.sh

#SBATCH --job-name=continuix-package
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/continuix_package_%j.out

module load python
conda activate igm32-frost

python -u experiments/continuix/package_submission.py
python -u experiments/continuix/check_submission.py \
    | tee data/results/continuix/check_GROUP_FAU2.txt
