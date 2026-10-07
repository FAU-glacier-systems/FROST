#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Time-step collapse in the TI projections: one RCP8.5 projection to 2035
# with yearly 2D output, for glaciers that slowed down (Fiescher BE, Gorner)
# with the new (Millan 2000, tau_ref) and the old (IGM initial guess)
# inversion, and Rhone (new) as a control. One Alex GPU. Submit from an
# Alex login node, in the repo root:
#   sbatch experiments/alps_TI_projections/diagnose_projection_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=diag_proj
#SBATCH --output=logs/diag_proj_%j.out
#SBATCH --error=logs/diag_proj_%j.err

module load python
conda activate igm32-frost

# projections.nc from the finished runs of the glaciers that hit the time limit
for rgi_id in RGI2000-v7.0-G-11-01225 RGI2000-v7.0-G-11-01581 RGI2000-v7.0-G-11-00823; do
    python experiments/alps_TI_projections/run_projections.py --rgi_id "$rgi_id" --collect_only
done

SCRIPT=experiments/alps_TI_projections/diagnose_projection.py
NEW=data/results/alps_TI_projections
OLD=data/results/alps_TI_projections_igm_guess
for case in "RGI2000-v7.0-G-11-01581 $NEW new" "RGI2000-v7.0-G-11-01581 $OLD old" \
            "RGI2000-v7.0-G-11-01225 $NEW new" "RGI2000-v7.0-G-11-01225 $OLD old" \
            "RGI2000-v7.0-G-11-01706 $NEW new"; do
    read -r rgi_id results label <<< "$case"
    python $SCRIPT --rgi_id "$rgi_id" --results "$results" --label "$label" \
        --end_year 2035 > "logs/diag_proj_${SLURM_JOB_ID}_${label}_${rgi_id: -5}.log" 2>&1 &
done
wait
grep -h -A40 "^== " logs/diag_proj_${SLURM_JOB_ID}_*.log
