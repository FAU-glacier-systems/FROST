#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Full TI chain for one glacier per array task on one Alex GPU: download,
# inversion and TI calibration (pipeline_config.yml), then the CORDEX
# projections and their plot. The glacier list has "<rgi_id> <name>" per
# line. Submit from an Alex login node, in the repo root:
#   sbatch --array=1-11 experiments/alps_TI_projections/alps_TI_sbatch.sh \
#       experiments/alps_TI_projections/test_glaciers.txt

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --job-name=frost_TI
#SBATCH --output=logs/frost_TI_%A_%a.out
#SBATCH --error=logs/frost_TI_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"

python frost_pipeline.py --config experiments/alps_TI_projections/pipeline_config.yml \
    --rgi_id "$RGI_ID" || exit 1
python experiments/alps_TI_projections/run_projections.py --rgi_id "$RGI_ID" \
    --workers 16 || exit 1
python experiments/alps_TI_projections/plot_projections.py --rgi_id "$RGI_ID" \
    --name "${NAME//_/ }"
