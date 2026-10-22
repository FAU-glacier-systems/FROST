#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Full FROST pipeline (download, inversion, ES-MDA ELA calibration), one line
# of runs.txt (config folder, glacier) per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-16 experiments/thickness_esmda/thickness_esmda_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=thk_esmda
#SBATCH --output=logs/thk_esmda_%A_%a.out
#SBATCH --error=logs/thk_esmda_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

EXPERIMENT=experiments/thickness_esmda
read -r PRODUCT RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/runs.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${PRODUCT} ${RGI_ID} (${NAME})"
python frost_pipeline.py --config "$EXPERIMENT/$PRODUCT/pipeline_config.yaml" --rgi_id "$RGI_ID"
