#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Smoother tau_ref and the velocity cap, one glacier per array task on one
# Alex GPU: inversions slide_only_tau1e11 and slide_only_tau1e12 (scores in
# data/results/alps_TI_inversion_variants/), and for Fiescher BE and Gorner
# the RCP8.5 diagnostic run to 2035 with the 1000 m/yr cap, on the current
# inversion and on both smoother ones. Submit from an Alex login node, in
# the repo root:
#   sbatch --array=1-11 experiments/alps_TI_projections/tau_cap_test_sbatch.sh \
#       experiments/alps_TI_projections/test_glaciers.txt
# then: python experiments/alps_TI_projections/inversion_variants.py --collect

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=tau_cap
#SBATCH --output=logs/tau_cap_%A_%a.out
#SBATCH --error=logs/tau_cap_%A_%a.err

module load python
conda activate igm32-frost

read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"

python experiments/alps_TI_projections/inversion_variants.py --rgi_id "$RGI_ID" \
    --variants slide_only_tau1e11 slide_only_tau1e12

case "$RGI_ID" in
    RGI2000-v7.0-G-11-01581|RGI2000-v7.0-G-11-01225)
        SCRIPT=experiments/alps_TI_projections/diagnose_projection.py
        VARIANTS=data/results/alps_TI_inversion_variants
        LOG=logs/tau_cap_${SLURM_ARRAY_JOB_ID}_${RGI_ID: -5}
        python $SCRIPT --rgi_id "$RGI_ID" --label new_cap --max_velbar 1000 \
            > ${LOG}_new_cap.log 2>&1 &
        python $SCRIPT --rgi_id "$RGI_ID" --label tau1e11_cap --max_velbar 1000 \
            --inversion $VARIANTS/slide_only_tau1e11 > ${LOG}_tau1e11_cap.log 2>&1 &
        python $SCRIPT --rgi_id "$RGI_ID" --label tau1e12_cap --max_velbar 1000 \
            --inversion $VARIANTS/slide_only_tau1e12 > ${LOG}_tau1e12_cap.log 2>&1 &
        wait
        grep -h -A40 "^== " ${LOG}_*_cap.log
        ;;
esac
