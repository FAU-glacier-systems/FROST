#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# 20-year TI calibration of Aletsch and Rhone with the iterated EnKF
# (FROST v1) and with ES-MDA, 6 iterations, same seed and prior. One array
# task per glacier and method; the inversion, observations and climate are
# copied from alps_TI_projections, so both methods see the same inputs.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-4 experiments/esmda_vs_enkf/esmda_vs_enkf_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --job-name=esmda_vs_enkf
#SBATCH --output=logs/esmda_vs_enkf_%A_%a.out
#SBATCH --error=logs/esmda_vs_enkf_%A_%a.err

module load python
conda activate igm32-frost

TASKS=("RGI2000-v7.0-G-11-02596 enkf"    # Grosser Aletschgletscher
       "RGI2000-v7.0-G-11-02596 esmda"
       "RGI2000-v7.0-G-11-01706 enkf"    # Rhone
       "RGI2000-v7.0-G-11-01706 esmda")
read -r RGI_ID METHOD <<< "${TASKS[$((SLURM_ARRAY_TASK_ID - 1))]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} ${METHOD}"

SOURCE=data/results/alps_TI_projections/${RGI_ID}
TARGET=data/results/esmda_vs_enkf/${METHOD}/${RGI_ID}
rm -rf "$TARGET"
mkdir -p "$TARGET"
cp -r "$SOURCE/Preprocess" "$SOURCE/observations.nc" \
    "$SOURCE/climate_historical.nc" "$TARGET/" || exit 1

python frost_calibration.py \
    --config experiments/esmda_vs_enkf/config_${METHOD}.yml --rgi_id "$RGI_ID"
