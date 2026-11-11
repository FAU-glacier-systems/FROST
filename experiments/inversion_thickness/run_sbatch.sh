#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inversions of experiments/inversion_thickness (all variants, or the ones
# given after the list) for one glacier of glaciers.txt per array task.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-2 experiments/inversion_thickness/run_sbatch.sh experiments/inversion_thickness/glaciers.txt
# then: python experiments/inversion_thickness/run.py --collect

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=inv_thk
#SBATCH --output=logs/inv_thk_%A_%a.out
#SBATCH --error=logs/inv_thk_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

LIST=$1
shift
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="inv_${NAME}" || true
if [ $# -gt 0 ]; then
    python experiments/inversion_thickness/run.py --rgi_id "$RGI_ID" --variants "$@"
else
    python experiments/inversion_thickness/run.py --rgi_id "$RGI_ID"
fi
