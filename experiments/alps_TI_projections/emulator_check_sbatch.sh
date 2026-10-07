#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Emulator vs direct solver on the inverted state of a few glaciers, one
# Alex GPU. Submit from an Alex login node, in the repo root:
#   sbatch experiments/alps_TI_projections/emulator_check_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=emu_check
#SBATCH --output=logs/emu_check_%j.out
#SBATCH --error=logs/emu_check_%j.err

module load python
conda activate igm32-frost

# Oberaletsch, Gepatsch, Corbassiere (slow, too thin), Grosser Aletsch (best)
for rgi_id in RGI2000-v7.0-G-11-02530 RGI2000-v7.0-G-11-03239 \
              RGI2000-v7.0-G-11-00823 RGI2000-v7.0-G-11-02596; do
    python experiments/alps_TI_projections/emulator_check.py "$rgi_id" 5000
done
