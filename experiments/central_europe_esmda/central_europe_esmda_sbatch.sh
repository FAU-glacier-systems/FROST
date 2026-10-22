#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Full FROST pipeline (download, inversion, ES-MDA ELA calibration) for one
# glacier of glaciers.txt per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-25 experiments/central_europe_esmda/central_europe_esmda_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:40:00
#SBATCH --job-name=ce_esmda
#SBATCH --output=logs/ce_esmda_%A_%a.out
#SBATCH --error=logs/ce_esmda_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

EXPERIMENT=experiments/central_europe_esmda
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
python frost_pipeline.py --config "$EXPERIMENT/pipeline_config.yaml" --rgi_id "$RGI_ID"
