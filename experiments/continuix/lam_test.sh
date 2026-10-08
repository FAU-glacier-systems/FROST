#!/bin/bash -l
# lam and grid test (lam_test.py): full EXP01 run of one glacier and
# variant on one Alex GPU, then its posterior band score.
# Submit from an Alex login node, in the repo root, after
# python experiments/continuix/lam_test.py --setup:
#   sbatch --array=1-18 experiments/continuix/lam_test.sh

#SBATCH --job-name=lam_test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/lam_test_%A_%a.out

module load python
conda activate igm32-frost

read glacier variant <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/continuix/lam_test_tasks.txt)"
dir=data/results/continuix/lam_test/$variant
echo "$glacier $variant"
nvidia-smi -L
python -u experiments/continuix/run_continuix.py --exp EXP01 --glacier "$glacier" \
    --config "$dir/config.yml" || exit 1
python -u experiments/continuix/lam_test.py --score "$dir/EXP01/$glacier"
# the lam 3e10 tasks also score the reference run (lam 1e11) of their grid
case $variant in
    r50_lam3e10) python -u experiments/continuix/lam_test.py --score data/results/continuix/EXP01/$glacier ;;
    r25_lam3e10) python -u experiments/continuix/lam_test.py --score data/results/continuix/res25/EXP01/$glacier ;;
esac
