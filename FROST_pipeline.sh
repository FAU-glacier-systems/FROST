#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --job-name=frost
#SBATCH --output=Experiments/Log/frost_%j.out
#SBATCH --error=Experiments/Log/frost_%j.err
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python
conda activate igm

# Default value for rgi_id
rgi_id="RGI2000-v7.0-G-11-01706"
download=false
scale_factor=1
inversion=false
calibrate=false
forward_parallel=false
seed=1
inflation=1

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rgi_id) rgi_id="$2"; shift 2 ;;  # Store the rgi_id
        --download) download=true; shift ;; # Set download flag to true
        --scale_factor) scale_factor="$2"; shift 2 ;; # Set download flag to true
        --inversion) inversion=true; shift ;;
        --calibrate) calibrate=true; shift ;; # Set calibrate flag to true
        --forward_parallel) forward_parallel=true; shift ;;
        --seed) seed="$2"; shift 2 ;;
        --inflation) inflation="$2"; shift 2 ;;
        *) echo "Unknown parameter $1"; exit 1 ;;  # Exit on unknown argument
    esac
done

# Display the selected rgi_id
echo "Running pipeline for RGI ID: $rgi_id"

# 0. create folders
SOURCE_FOLDER="Experiments/default"
DEST_FOLDER="Experiments/$rgi_id"
mkdir "$DEST_FOLDER"
cp "$SOURCE_FOLDER"/*.json "$DEST_FOLDER"

# 1. Download data with OGGM_shop (if --download is set)
if [ "$download" = true ]; then
    echo "Downloading data..."
    pushd Scripts/Preprocess
    python -u download_data.py --rgi_id "$rgi_id" --scale_factor $scale_factor \
    --download_oggm_shop --download_hugonnet --year_interval 20
    popd
fi

# 2. IGM inversion step (if --inversion is set)
if [ "$inversion" = true ]; then
    echo "Starting IGM inversion..."
    pushd Scripts/Preprocess
    python -u igm_inversion.py --rgi_id "$rgi_id"
    popd
fi

# 3. Calibration step (if --calibrate is set)
if [ "$calibrate" = true ]; then
    echo "Starting calibration..."
    python -u run_calibration.py --rgi_id "$rgi_id" --ensemble_size 100 \
    --forward_parallel "$forward_parallel" --iterations 6 --seed "$seed" \
    --inflation "$inflation" --elevation_step 30
fi
