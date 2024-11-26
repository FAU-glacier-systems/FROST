#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=frost
module load python
conda activate igm

# Default value for rgi_id
rgi_id="RGI2000-v7.0-G-11-01706"
download=false
scale_factor=0.5
inversion=false
calibrate=false


# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rgi_id) rgi_id="$2"; shift 2 ;;  # Store the rgi_id
        --download) download=true; shift ;; # Set download flag to true
        --scale_factor) scale_factor="$2"; shift 2 ;; # Set download flag to true
        --inversion) inversion=true; shift ;;
        --calibrate) calibrate=true; shift ;; # Set calibrate flag to true
        *) echo "Unknown parameter $1"; exit 1 ;;  # Exit on unknown argument
    esac
done

# Display the selected rgi_id
echo "Running pipeline for RGI ID: $rgi_id"

# 1. Download data with OGGM_shop (if --download is set)
if [ "$download" = true ]; then
    echo "Downloading data..."
    pushd Preprocess
    python download_data.py --rgi_id "$rgi_id" --scale_factor $scale_factor \
    --download_oggm_shop --download_hugonnet
    popd
fi

# 3. IGM inversion step (if --inversion is set)
if [ "$inversion" = true ]; then
    echo "Starting IGM inversion..."
    pushd Preprocess
    python igm_inversion.py --rgi_id "$rgi_id"
    popd
fi

# 4. Calibration step (if --calibrate is set)
if [ "$calibrate" = true ]; then
    echo "Starting calibration..."
    # Forward the rgi_id argument to the Python script
    python FROST_RUN.py --rgi_id "$rgi_id"
fi
