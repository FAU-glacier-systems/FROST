#!/bin/bash -l
# Native-grid test (config_native.yml): EXP01 on one Alex GPU, then
# the posterior band score of the run and of the current 50 m one
# as in lam_test.py. Submit from an Alex login node, in the
# repo root:
#   sbatch --array=1 --time=01:00:00 experiments/continuix/native_test.sh   # G03 20 m
#   sbatch --array=2 --time=08:00:00 experiments/continuix/native_test.sh   # G04 2 m
#   sbatch --array=3 --time=24:00:00 experiments/continuix/native_test.sh   # G05 10 m
#   sbatch --array=4 --time=04:00:00 experiments/continuix/native_test.sh   # G06 10 m

#SBATCH --job-name=native_test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/native_test_%A_%a.out

module load python
conda activate igm32-frost

read exp glacier <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/continuix/tasks_native.txt)"
echo "$exp $glacier"
nvidia-smi -L
python -u experiments/continuix/run_continuix.py --exp "$exp" --glacier "$glacier" \
    --config experiments/continuix/config_native.yml || exit 1
python -u experiments/continuix/lam_test.py --score data/results/continuix/res_native/$exp/$glacier
python -u experiments/continuix/lam_test.py --score data/results/continuix/$exp/$glacier
