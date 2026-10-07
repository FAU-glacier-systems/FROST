#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI projections (posterior mean x all CORDEX runs) for one glacier on one
# Alex GPU. Submit from an Alex login node, in the repo root:
#   sbatch experiments/alps_TI_projections/projections_sbatch.sh <rgi_id>

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --job-name=frost_proj
#SBATCH --output=logs/frost_proj_%j.out
#SBATCH --error=logs/frost_proj_%j.err

module load python
conda activate igm32-frost

python experiments/alps_TI_projections/run_projections.py \
    --rgi_id "${1:-RGI2000-v7.0-G-11-01706}" --workers 16
