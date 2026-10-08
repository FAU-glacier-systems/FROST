#!/bin/bash -l
# Timing of the ES-MDA forward runs (bench_forward.py) on one Alex GPU.
# Submit from an Alex login node, in the repo root, with a calibrated glacier:
#   sbatch experiments/continuix/bench_forward.sh data/results/continuix/res25/EXP01/G02

#SBATCH --job-name=bench_forward
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_forward_%j.out

module load python
conda activate igm32-frost

nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 5 &
python -u experiments/continuix/bench_forward.py "$@"
