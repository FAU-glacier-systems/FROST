#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# IGM 3.3.0 retraining rules vs the solver (forward_test.py), one glacier of
# glaciers_forward.txt per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-3 experiments/igm330_retrain/forward_test_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=igm330_fwd
#SBATCH --output=logs/igm330_fwd_%A_%a.out
#SBATCH --error=logs/igm330_fwd_%A_%a.err

module load python
conda activate igm32-frost

EXPERIMENT=experiments/igm330_retrain
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers_forward.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
python "$EXPERIMENT/forward_test.py" --rgi_id "$RGI_ID" "$@"
