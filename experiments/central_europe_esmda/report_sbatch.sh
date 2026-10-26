#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Evaluation (evaluate.py) and dark sanity report (sanity_report.py) of one
# results folder on a Fritz CPU node, not on the login node: the report
# reads every posterior member of every glacier. For a combined run, make
# the links first (make_combined.py, light). JetBrains Mono comes from
# ~/.local/share/fonts (shared home).
# Submit from a Fritz login node, in the repo root, e.g.:
#   sbatch experiments/central_europe_esmda/report_sbatch.sh central_europe_combined_pretrain

#SBATCH --nodes=1
#SBATCH --partition=singlenode
#SBATCH --time=00:30:00
#SBATCH --job-name=report
#SBATCH --output=logs/report_%j.out
#SBATCH --error=logs/report_%j.err

module load python
conda activate igm32-frost

NAME=${1:?results name, e.g. central_europe_combined_pretrain}
scontrol update JobId="$SLURM_JOB_ID" JobName="report_$NAME" || true
python experiments/central_europe_esmda/evaluate.py --results "data/results/$NAME" \
    --out "experiments/$NAME"
python experiments/central_europe_esmda/sanity_report.py --results "data/results/$NAME" --dark
