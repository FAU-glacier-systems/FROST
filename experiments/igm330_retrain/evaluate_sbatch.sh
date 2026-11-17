#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# GLAMOS/WGMS evaluation (central_europe_esmda/evaluate.py: ELA, gradients,
# SMB profile RMSE) of the calibrations compared in report.py. Reads
# every posterior member, so it runs as a job. CPU only, but Alex jobs need a
# GPU: one A40 (and its 16 cores) is requested and stays idle.
# Submit from an Alex login node, in the repo root, after the calibrations:
#   sbatch experiments/igm330_retrain/evaluate_sbatch.sh               (all four)
#   sbatch experiments/igm330_retrain/evaluate_sbatch.sh igm330_live   (only these)

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --job-name=igm330_eval
#SBATCH --output=logs/igm330_eval_%j.out
#SBATCH --error=logs/igm330_eval_%j.err

module load python
conda activate igm32-frost

NAMES=${*:-central_europe_retrain_t1 igm330_default igm330_credit igm330_live}
for NAME in $NAMES; do
    python experiments/central_europe_esmda/evaluate.py --results "data/results/$NAME" \
        --out "experiments/igm330_retrain/eval_$NAME"
done
