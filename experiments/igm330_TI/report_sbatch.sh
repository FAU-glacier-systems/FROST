#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Report of igm330_TI (ti_report.py) and its inputs: sanity report,
# evaluate.py (ELA and gradients of the TI profile), evaluate_bands.py of
# this run and of the earlier TI calibration (central_europe_TI_combined),
# both with the ELA calibration igm330_live_combined. Reads every posterior
# member, so it runs as a job. CPU only, but Alex jobs need a GPU: one A40
# is requested and stays idle.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_TI/report_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=report_igm330_TI
#SBATCH --output=logs/report_igm330_TI_%j.out
#SBATCH --error=logs/report_igm330_TI_%j.err

module load python
conda activate igm32-frost
set -e

NAME=igm330_TI
PREVIOUS=central_europe_TI_combined
ELA=igm330_live_combined
EXPERIMENT=experiments/$NAME
python experiments/central_europe_esmda/sanity_report.py --results "data/results/$NAME"
python experiments/central_europe_esmda/evaluate.py --results "data/results/$NAME" --out "$EXPERIMENT"
python experiments/central_europe_TI/evaluate_bands.py --ti "data/results/$NAME" \
    --ela "data/results/$ELA" --glaciers "$EXPERIMENT/glaciers.txt" --out "$EXPERIMENT"
python experiments/central_europe_TI/evaluate_bands.py --ti "data/results/$PREVIOUS" \
    --ela "data/results/$ELA" --glaciers "$EXPERIMENT/glaciers.txt" \
    --out "$EXPERIMENT/comparison/$PREVIOUS"
python "$EXPERIMENT/ti_report.py" --results "data/results/$NAME" \
    --previous "data/results/$PREVIOUS" --ela "data/results/$ELA"
