#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI calibration (OGGM temperature-index SMB) of the glaciers that move, with
# the setup of central_europe_retrain (frozen inversion, emulator retrained
# in the forward runs). Inputs first: copy_inputs.sh. ES-MDA calibration of one
# glacier of glaciers_igm.txt per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-9 experiments/central_europe_TI/central_europe_TI_igm_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --job-name=ce_TI
#SBATCH --output=logs/ce_TI_%A_%a.out
#SBATCH --error=logs/ce_TI_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

EXPERIMENT=experiments/central_europe_TI
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers_igm.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
# glacier name as the job name in squeue (pending tasks keep ce_TI);
# <array id>_<task id>, not SLURM_JOB_ID: one task's SLURM_JOB_ID is the
# array's own id, and renaming that renames every task
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$NAME" || true
python frost_pipeline.py --config "$EXPERIMENT/pipeline_config_igm.yaml" --rgi_id "$RGI_ID"
