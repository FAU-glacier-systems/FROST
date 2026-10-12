#!/bin/bash -l
# Reduce the EXP01 result and input files to the maps of the ContinuIX
# report (summary/maps_EXP01.npz). Reads ~250 MB of netCDF, so not on the
# login node. Submit from an Alex login node, in the repo root, after
# package_and_check.sh:
#   sbatch experiments/continuix/report_maps.sh
# then: python experiments/continuix/report_resolution.py

#SBATCH --job-name=continuix-maps
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/continuix_maps_%j.out

module load python
conda activate igm32-frost

python -u experiments/continuix/report_resolution.py --extract-maps
