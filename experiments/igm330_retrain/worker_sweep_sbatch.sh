#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Forward workers per GPU (worker_sweep.py): 3 glaciers x 6 worker counts,
# one Alex A40 (and its 16 cores) per task. Submit from an Alex login node,
# in the repo root:
#   sbatch --array=0-17 experiments/igm330_retrain/worker_sweep_sbatch.sh
# then: python experiments/igm330_retrain/worker_sweep.py summary

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --job-name=igm330_sweep
#SBATCH --output=logs/igm330_sweep_%A_%a.out
#SBATCH --error=logs/igm330_sweep_%A_%a.err

module load python
conda activate igm32-frost

GLACIERS=(Silvretta Rhone Aletsch)
WORKERS=(4 8 12 16 24 32)
G=${GLACIERS[$((SLURM_ARRAY_TASK_ID / 6))]}
N=${WORKERS[$((SLURM_ARRAY_TASK_ID % 6))]}
echo "Task ${SLURM_ARRAY_TASK_ID}: $G, $N workers"
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="sweep_${G}_${N}" || true
python experiments/igm330_retrain/worker_sweep.py "$G" "$N"
