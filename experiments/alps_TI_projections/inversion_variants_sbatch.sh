#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inversion variants (inversion_variants/*.yaml) for one glacier per array
# task on one Alex GPU. The glacier list has "<rgi_id> <name>" per line.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-10 experiments/alps_TI_projections/inversion_variants_sbatch.sh \
#       experiments/alps_TI_projections/largest10.txt
# then collect: python experiments/alps_TI_projections/inversion_variants.py --collect

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=inv_variants
#SBATCH --output=logs/inv_variants_%A_%a.out
#SBATCH --error=logs/inv_variants_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
python experiments/alps_TI_projections/inversion_variants.py --rgi_id "$RGI_ID"
