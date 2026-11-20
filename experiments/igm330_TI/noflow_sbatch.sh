#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI calibration without ice flow of the 8 glaciers of glaciers_noflow.txt,
# all at once on one node (CPU only, but Alex jobs need a GPU: one A40 is
# requested and stays idle).
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_TI/noflow_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=igm330_TI_noflow
#SBATCH --output=logs/igm330_TI_noflow_%j.out
#SBATCH --error=logs/igm330_TI_noflow_%j.err

module load python
conda activate igm32-frost
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=

EXPERIMENT=experiments/igm330_TI
xargs -P 8 -L 1 bash -c 'python frost_pipeline.py --config '"$EXPERIMENT"'/pipeline_config_noflow.yaml \
    --rgi_id "$0" > logs/igm330_TI_noflow_$0.log 2>&1 && echo "$1 done" || echo "$1 FAILED"' \
    < "$EXPERIMENT/glaciers_noflow.txt"
