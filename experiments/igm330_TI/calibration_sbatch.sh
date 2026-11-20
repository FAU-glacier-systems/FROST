#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# TI calibration (ES-MDA) of the 17 glaciers with ice flow on IGM 3.3.0 with
# the live IGM model and credit retraining, one glacier per array task on
# one Alex GPU; inputs from copy_inputs.sh.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-17 experiments/igm330_TI/calibration_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=igm330_TI
#SBATCH --output=logs/igm330_TI_%A_%a.out
#SBATCH --error=logs/igm330_TI_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export FROST_LIVE_IGM=1

module load python
conda activate igm32-frost

EXPERIMENT=experiments/igm330_TI
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers_igm.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME}), $EXPERIMENT, FROST_LIVE_IGM=$FROST_LIVE_IGM"
python -c 'import importlib.metadata as m; print("IGM", m.version("igm-model"))'
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="TI_$NAME" || true
python frost_pipeline.py --config "$EXPERIMENT/pipeline_config.yaml" --rgi_id "$RGI_ID"
