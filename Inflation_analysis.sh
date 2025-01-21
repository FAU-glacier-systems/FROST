#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --job-name=frost
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
seeds=(1 2 3 4 5 6 7 8 9)
inflation_factors=(1.0 1.2 1.4 1.6 1.8 2.0)

echo "Starting calibration..."
# Iterate over each seed
for seed in "${seeds[@]}"; do
    # Iterate over each inflation factor
    for inflation in "${inflation_factors[@]}"; do
        echo "Running with seed: $seed and inflation: $inflation"
        python -u FROST_RUN.py --rgi_id "$rgi_id" --ensemble_size 50 \
        --forward_parallel "$forward_parallel" --iterations 5 \
        --seed "$seed" --inflation "$inflation"  --elevation_bin 50
    done
done

