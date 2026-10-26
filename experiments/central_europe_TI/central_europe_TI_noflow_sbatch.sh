#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI calibration without ice flow (EnKF ice_flow: none) of the glaciers of
# glaciers_noflow.txt on one Fritz CPU node, 8 glaciers at a time with 9 cores
# each (no GPU needed). Inputs first: copy_inputs.sh.
# Submit from a Fritz login node, in the repo root:
#   sbatch experiments/central_europe_TI/central_europe_TI_noflow_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=singlenode
#SBATCH --time=01:00:00
#SBATCH --job-name=ce_TI_noflow
#SBATCH --output=logs/ce_TI_noflow_%j.out
#SBATCH --error=logs/ce_TI_noflow_%j.err

module load python
conda activate igm32-frost
export OMP_NUM_THREADS=9 OPENBLAS_NUM_THREADS=9 MKL_NUM_THREADS=9

EXPERIMENT=experiments/central_europe_TI
mkdir -p logs
xargs -P 8 -L 1 bash -c 'python frost_pipeline.py --config '"$EXPERIMENT"'/pipeline_config_noflow.yaml \
    --rgi_id "$0" > logs/TI_noflow_$0.log 2>&1 && echo "$1 done" || echo "$1 FAILED"' \
    < "$EXPERIMENT/glaciers_noflow.txt"
