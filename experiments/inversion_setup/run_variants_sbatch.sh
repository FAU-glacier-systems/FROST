#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inversion setups (variants/*.yaml) for one glacier per array
# task on one Alex GPU. The glacier list has "<rgi_id> <name>" per line.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-2 experiments/inversion_setup/run_variants_sbatch.sh \
#       experiments/inversion_setup/glaciers.txt [--variants <name> ...]
# then collect: python experiments/inversion_setup/run_variants.py --collect

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=inv_setup
#SBATCH --output=logs/inv_setup_%A_%a.out
#SBATCH --error=logs/inv_setup_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

LIST=$1
shift  # further arguments go to run_variants.py, e.g. --variants a b
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
python experiments/inversion_setup/run_variants.py --rgi_id "$RGI_ID" "$@"
