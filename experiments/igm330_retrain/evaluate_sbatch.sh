#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# GLAMOS/WGMS evaluation (central_europe_esmda/evaluate.py: ELA, gradients,
# SMB profile RMSE) of the three calibrations compared in report.py. Reads
# every posterior member, so it runs as a job (CPU).
# Submit from a Fritz login node, in the repo root, after the calibrations:
#   sbatch experiments/igm330_retrain/evaluate_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=singlenode
#SBATCH --time=01:30:00
#SBATCH --job-name=igm330_eval
#SBATCH --output=logs/igm330_eval_%j.out
#SBATCH --error=logs/igm330_eval_%j.err

module load python
conda activate igm32-frost

for NAME in central_europe_retrain_t1 igm330_default igm330_credit; do
    python experiments/central_europe_esmda/evaluate.py --results "data/results/$NAME" \
        --out "experiments/igm330_retrain/eval_$NAME"
done
