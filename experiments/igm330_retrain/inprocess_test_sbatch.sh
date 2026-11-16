#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# IGM several times in one process (inprocess_test.py), one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_retrain/inprocess_test_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:20:00
#SBATCH --job-name=igm330_inproc
#SBATCH --output=logs/igm330_inproc_%j.out
#SBATCH --error=logs/igm330_inproc_%j.err

module load python
conda activate igm32-frost
python experiments/igm330_retrain/inprocess_test.py
