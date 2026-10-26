#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# central_europe_retrain without the inversion: uniform default tau_ref (0.213 MPa),
# emulator retrained in the forward runs, all 25 glaciers. Full FROST
# pipeline (download, inversion, ES-MDA ELA calibration) for one
# glacier of glaciers.txt per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-25 experiments/central_europe_tau_default/central_europe_tau_default_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ce_taudef
#SBATCH --output=logs/ce_taudef_%A_%a.out
#SBATCH --error=logs/ce_taudef_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

EXPERIMENT=experiments/central_europe_tau_default
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
# glacier name as the job name in squeue (pending tasks keep ce_taudef);
# <array id>_<task id>, not SLURM_JOB_ID: one task's SLURM_JOB_ID is the
# array's own id, and renaming that renames every task
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$NAME" || true
python frost_pipeline.py --config "$EXPERIMENT/pipeline_config.yaml" --rgi_id "$RGI_ID"
