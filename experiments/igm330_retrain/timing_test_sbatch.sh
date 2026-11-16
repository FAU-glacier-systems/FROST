#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Time budget of one calibration member (timing_test.py), one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_retrain/timing_test_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=igm330_timing
#SBATCH --output=logs/igm330_timing_%j.out
#SBATCH --error=logs/igm330_timing_%j.err

module load python
conda activate igm32-frost
python experiments/igm330_retrain/timing_test.py
