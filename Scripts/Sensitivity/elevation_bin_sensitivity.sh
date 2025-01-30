#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=0:59:00
#SBATCH --job-name=frost
#SBATCH --output=Experiments/Log/frost_%j.out
#SBATCH --error=Experiments/Log/frost_%j.err
module load python
conda activate igm

# Default value for rgi_id
forward_parallel=true
#!/bin/bash
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rgi_id) rgi_id="$2"; shift 2 ;;  # Store the rgi_id
    esac
done

# Define seeds and inflation factors
seeds=(1 2 3 4 5)
elevation_bin=(10 20 30 40 50)

echo "Starting calibration..."
# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Iterate over each inflation factor
    for num_bins in "${elevation_bin[@]}"; do
        echo "Running with seed: $seed and num_bin: $num_bins"
        python -u FROST_RUN.py --rgi_id "$rgi_id" --ensemble_size 50 \
        --forward_parallel "$forward_parallel" --iterations 5 \
        --seed "$seed" --inflation 1.1 --num_bins "$num_bins"
    done
done

