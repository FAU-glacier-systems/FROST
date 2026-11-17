#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# ES-MDA ELA calibration of the 17 glaciers with ice flow on IGM 3.3.0 with
# one retraining rule (igm330_default or igm330_credit), one glacier per
# array task on one Alex GPU; inputs from copy_inputs.sh.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-17 experiments/igm330_retrain/calibration_sbatch.sh igm330_default
#   sbatch --array=1-17 experiments/igm330_retrain/calibration_sbatch.sh igm330_credit
#   sbatch --array=1-17 experiments/igm330_retrain/calibration_sbatch.sh igm330_live

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:30:00
#SBATCH --job-name=igm330
#SBATCH --output=logs/igm330_%A_%a.out
#SBATCH --error=logs/igm330_%A_%a.err

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm32-frost

EXPERIMENT=experiments/${1:?experiment: igm330_default, igm330_credit or igm330_live}
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME}), $EXPERIMENT"
python -c 'import importlib.metadata as m; print("IGM", m.version("igm-model"))'
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$NAME" || true
python frost_pipeline.py --config "$EXPERIMENT/pipeline_config.yaml" --rgi_id "$RGI_ID"
