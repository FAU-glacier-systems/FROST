#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Full FROST pipeline for one glacier on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch hpc_sbatch.sh <pipeline_config.yml> <rgi_id>

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=frost
#SBATCH --output=logs/frost_%j.out
#SBATCH --error=logs/frost_%j.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

python frost_pipeline.py --config "$1" --rgi_id "$2"
