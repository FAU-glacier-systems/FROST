#!/bin/bash -l
# Inversion settings on a 25 m grid (inversion_test.py), one Alex GPU per
# glacier and variant. Submit from an Alex login node, in the repo root:
#   sbatch --array=1-$(wc -l < experiments/continuix/inversion_test_tasks.txt) experiments/continuix/inversion_test.sh

#SBATCH --job-name=inversion_test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/inversion_test_%A_%a.out

module load python
conda activate igm32-frost

read glacier variant <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" experiments/continuix/inversion_test_tasks.txt)"
echo "$glacier $variant"
nvidia-smi -L
python -u experiments/continuix/inversion_test.py --glacier "$glacier" --variant "$variant"
