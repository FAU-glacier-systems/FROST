#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Live IGM model prototype (live_model_test.py), one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_retrain/live_model_test_sbatch.sh          (fresh + live)
#   sbatch experiments/igm330_retrain/live_model_test_sbatch.sh live     (reuse fresh_*)
#   RULE=credit sbatch experiments/igm330_retrain/live_model_test_sbatch.sh   (other retrain rule)

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:40:00
#SBATCH --job-name=igm330_live
#SBATCH --output=logs/igm330_live_%j.out
#SBATCH --error=logs/igm330_live_%j.err

module load python
conda activate igm32-frost
RULE=${RULE:-igm_default}
[ "$1" = live ] || python experiments/igm330_retrain/live_model_test.py fresh "$RULE"
python experiments/igm330_retrain/live_model_test.py live "$RULE"
python experiments/igm330_retrain/live_model_test.py compare "$RULE"
