#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI calibrations without ice flow of central_europe_TI_snow and
# central_europe_TI_snow_lapse (16 glaciers each) on the 16 cores of one
# Alex A40 job (CPU only; the GPU stays idle), 4 runs at a time with 4
# threads each. Inputs first: copy_inputs.sh of both experiments.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/central_europe_TI_snow/central_europe_TI_snow_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=ce_TI_snow
#SBATCH --output=logs/ce_TI_snow_%j.out
#SBATCH --error=logs/ce_TI_snow_%j.err

module load python
conda activate igm32-frost
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""

mkdir -p logs
for EXPERIMENT in central_europe_TI_snow central_europe_TI_snow_lapse; do
    sed "s|^|experiments/$EXPERIMENT |" "experiments/$EXPERIMENT/glaciers.txt"
done | xargs -P 4 -L 1 bash -c 'python frost_pipeline.py --config "$0/pipeline_config.yaml" \
    --rgi_id "$1" > "logs/$(basename "$0")_$1.log" 2>&1 && echo "$0 $2 done" || echo "$0 $2 FAILED"'
