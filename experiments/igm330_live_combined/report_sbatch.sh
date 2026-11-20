#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Evaluation (evaluate.py) and light sanity report (sanity_report.py) of the
# 25 GLAMOS / WGMS glaciers on IGM 3.3.0 with the live IGM model: the 17
# glaciers with ice flow from igm330_live, the 8 slow ones from
# central_europe_noflow (links made by make_combined.py, see selection.tsv).
# Reads every posterior member, so it runs as a job. CPU only, but Alex jobs
# need a GPU: one A40 is requested and stays idle.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/igm330_live_combined/report_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=report_igm330_live_combined
#SBATCH --output=logs/report_igm330_live_combined_%j.out
#SBATCH --error=logs/report_igm330_live_combined_%j.err

module load python
conda activate igm32-frost

NAME=igm330_live_combined
python experiments/central_europe_esmda/evaluate.py --results "data/results/$NAME" \
    --out "experiments/$NAME"
python experiments/central_europe_esmda/sanity_report.py --results "data/results/$NAME"
