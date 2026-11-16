#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Does TensorFlow's persistent XLA compilation cache remove the per-member
# compile time? timing_test.py twice with the cache on: the first process
# fills it, the second reads it. Compare with the run without cache
# (job 5014988). One Alex GPU, from the repo root:
#   sbatch experiments/igm330_retrain/xla_cache_test_sbatch.sh

#SBATCH --nodes=1
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --job-name=igm330_xla
#SBATCH --output=logs/igm330_xla_%j.out
#SBATCH --error=logs/igm330_xla_%j.err

module load python
conda activate igm32-frost

CACHE="$TMPDIR/xla_cache"
mkdir -p "$CACHE"
export TF_XLA_FLAGS="--tf_xla_persistent_cache_directory=$CACHE --tf_xla_disable_strict_signature_checks=true"
echo "TF_XLA_FLAGS=$TF_XLA_FLAGS"
for pass in 1 2; do
    echo "=== process $pass"
    python experiments/igm330_retrain/timing_test.py
    echo "cache files: $(ls "$CACHE" | wc -l)"
done
