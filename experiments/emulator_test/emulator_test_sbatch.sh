#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Forward runs of one posterior member with the frozen emulator, the
# retrained emulator and the solver (run_emulator_test.py), one glacier of
# glaciers.txt per array task on one Alex GPU.
# Submit from an Alex login node, in the repo root:
#   sbatch --array=1-4 experiments/emulator_test/emulator_test_sbatch.sh
# Further arguments go to run_emulator_test.py, e.g. --variants retrain_1

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --job-name=emu_test
#SBATCH --output=logs/emu_test_%A_%a.out
#SBATCH --error=logs/emu_test_%A_%a.err

module load python
conda activate igm32-frost

EXPERIMENT=experiments/emulator_test
read -r RGI_ID NAME <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPERIMENT/glaciers.txt")"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${RGI_ID} (${NAME})"
python "$EXPERIMENT/run_emulator_test.py" --rgi_id "$RGI_ID" "$@"
