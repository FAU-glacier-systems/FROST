#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

#SBATCH --nodes=1
#SBATCH --time=5:59:00
#SBATCH --job-name=frost
#SBATCH --output=Experiments/Log/frost_%j.out
#SBATCH --error=Experiments/Log/frost_%j.err
module load python
conda activate igm

# Default value for rgi_id
rgi_id="RGI2000-v7.0-G-11-01706_v3"
download=false
inversion=false
synth_obs=false
calibrate=true
forward_parallel=true
seed=1

#!/bin/bash
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rgi_id) rgi_id="$2"; shift 2 ;;
        --download) download=true; shift ;;
        --inversion) inversion=true; shift ;;
        --synth_obs) synth_obs=true; shift ;;
        --calibrate) calibrate=true; shift ;;
        --seed) seed="$2"; shift 2 ;;
    esac
done

# Display the selected rgi_id
echo "Running pipeline for RGI ID: $rgi_id"
# 0. create folders
#pushd  ../Preprocess
#python -u create_folder.py --rgi_id "$rgi_id"
#popd

# 1. Download data with OGGM_shop (if --download is set)
if [ "$download" = true ]; then
    echo "Downloading data..."
    pushd ../Preprocess
    python -u download_data.py --rgi_id "$rgi_id" \
    --download_oggm_shop
    popd
fi

# 2. IGM inversion step (if --inversion is set)
if [ "$inversion" = true ]; then
    echo "Starting IGM inversion..."
    pushd ../Preprocess
    python -u igm_inversion.py --rgi_id "$rgi_id"
    popd
fi

# 3. Generate synthetic observations
if [ "$synth_obs" = true ]; then
    echo "Starting IGM inversion..."
    pushd ../Preprocess
    python -u synth_observation.py --rgi_id "$rgi_id"
    popd
fi

# 4. Calibration step (if --calibrate is set)
if [ "$calibrate" = true ]; then
    # Define seeds and inflation factors
    pushd ../..
    seeds=(1 2 3 4 5 6 7 8 9 10)
    ensemble_size=32
    iterationss=(2 4 6 8 10)
    elevation_step=50
    obs_uncertainty=20
    init_offset=20
    synthetic=true

    # Iterate over each seed
    for seed in "${seeds[@]}"; do
        # Iterate over each inflation factor
        for iterations in "${iterationss[@]}"; do
            echo "Starting calibration with iteration=$iterations..."
            padded_iteration=$(printf "%03d" "$iterations")
            padded_seed=$(printf "%03d" "$seed")
            results_dir="Experiments/${rgi_id}/iterations/${padded_iteration}/Seed_${padded_seed}"

            cmd="python -u FROST_calibration.py \
                  --rgi_id \"$rgi_id\" \
                  --synthetic \"$synthetic\" \
                  --forward_parallel \"$forward_parallel\" \
                  --ensemble_size \"$ensemble_size\" \
                  --iterations \"$iterations\" \
                  --seed \"$seed\" \
                  --elevation_step \"$elevation_step\" \
                  --obs_uncertainty \"$obs_uncertainty\" \
                  --init_offset \"$init_offset \" \
                  --results_dir \"$results_dir\""

            echo "$cmd"
            eval $cmd
        done

    done
fi
