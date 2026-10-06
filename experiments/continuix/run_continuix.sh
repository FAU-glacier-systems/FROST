#!/bin/bash -l
# FROST for ContinuIX on one Alex GPU, one job per experiment and glacier.
# Submit from an Alex login node, in the repo root:
#   sbatch experiments/continuix/run_continuix.sh --exp EXP01 --glacier S01
# or all tasks of a list (lines "EXP01 S01"), as a job array:
#   sbatch --array=1-$(wc -l < tasks.txt) experiments/continuix/run_continuix.sh tasks.txt

#SBATCH --job-name=continuix
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/continuix_%A_%a.out

module load python
conda activate igm32-frost

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    read exp glacier <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
    set -- --exp "$exp" --glacier "$glacier" "${@:2}"
fi
echo "$@"
nvidia-smi -L
# -u: print lines reach the log as they happen, not when the buffer fills
python -u experiments/continuix/run_continuix.py "$@"
