#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# central_europe_TI with the melted-out bands left out: TI calibration without
# ice flow (EnKF ice_flow: none) of the glaciers of glaciers_noflow.txt, here
# on one Alex A40 node (CPU only; the GPU is just the unit Alex allocates),
# 8 glaciers at a time. Inputs first: copy_inputs.sh.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/central_europe_TI_meltout/central_europe_TI_meltout_noflow_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ce_TIm_noflow
#SBATCH --output=logs/ce_TIm_noflow_%j.out
#SBATCH --error=logs/ce_TIm_noflow_%j.err

module load python
conda activate igm32-frost
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""

EXPERIMENT=experiments/central_europe_TI_meltout
mkdir -p logs
xargs -P 8 -L 1 bash -c 'python frost_pipeline.py --config '"$EXPERIMENT"'/pipeline_config_noflow.yaml \
    --rgi_id "$0" > logs/TIm_noflow_$0.log 2>&1 && echo "$1 done" || echo "$1 FAILED"' \
    < "$EXPERIMENT/glaciers_noflow.txt"
