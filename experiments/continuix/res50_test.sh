#!/bin/bash -l
# 50 m test of G01 and S01 (config_res50.yml): EXP01 on one Alex GPU, then
# the posterior band score of the run and of the current one (G01 100 m,
# S01 25 m), as in lam_test.py. Submit from an Alex login node, in the
# repo root:
#   sbatch --array=1 --time=04:00:00 experiments/continuix/res50_test.sh   # G01
#   sbatch --array=2 --time=01:00:00 experiments/continuix/res50_test.sh   # S01

#SBATCH --job-name=res50_test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/res50_test_%A_%a.out

module load python
conda activate igm32-frost

read exp glacier <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/continuix/tasks_res50.txt)"
echo "$exp $glacier"
nvidia-smi -L
python -u experiments/continuix/run_continuix.py --exp "$exp" --glacier "$glacier" \
    --config experiments/continuix/config_res50.yml || exit 1
python -u experiments/continuix/lam_test.py --score data/results/continuix/res50/$exp/$glacier
python -u experiments/continuix/lam_test.py --score data/results/continuix/$exp/$glacier
