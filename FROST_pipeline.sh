#!/bin/bash -l
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

#SBATCH --nodes=1
#SBATCH --time=00:59:59
#SBATCH --job-name=frost
#SBATCH --gres=gpu:a100:1
#SBATCH --output=Experiments/Log/frost_%j.out
#SBATCH --error=Experiments/Log/frost_%j.err


export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module load python
conda activate igm_gpu

#####################################
#                                   #
# GENERAL SETTINGS                  #
# (set as default)                  #
#                                   #
#####################################

# Organisation 
#####################################

# specify an individual experiment ID
exp_version='v02'

# results output directory
#results_dir=""

# glacier ID selection
rgi_id="RGI2000-v7.0-G-11-01706"

# SMB model decision
# options:
#           ELA: equlibrium line altitude, vertical SMB gradients
#           TI:  OGGM temperature index implementation
SMB_model="ELA"

# select FROST steps
# STEP 1
download=false
# STEP 2
inversion=false
# STEP 3
calibrate=true


# STEP 1: SETUP GENERATION
#####################################
hugonnet_storage_dir="../../Data/Hugonnet/11_rgi60_2000-01-01_2020-01-01"
# specify desired resolution
# float value given in metres
# if "none" --> the OGGM default resolution is used (size dependent)
target_resolution="none"
#target_resolution=100.0 

# STEP 3: EnKF specifications
#####################################

# seed for randome initial parameter set
EnKF_seed=6
# inflation parameter during KF update
EnKF_inflation=1
# ensemble size
EnKF_ensemble_size=10
# KF iterations
EnKF_iterations=6
# offset of initial ensemble parameters
EnKF_init_offset=0
# elevation binning (in meters)
EnKF_elev_band_height=50
# optional parallel flag for ensemble forwaring
EnKF_forward_parallel=true
# option for synthetic experiment
EnKF_synthetic=false

# Alternative command line parsing
#####################################

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --rgi_id) rgi_id="$2"; shift 2 ;;  # Store the rgi_id
        --download) download=true; shift ;; # Set download flag to true
        --inversion) inversion=true; shift ;;
        --calibrate) calibrate=true; shift ;; # Set calibrate flag to true
        --forward_parallel) EnKF_forward_parallel=true; shift ;;
        --seed) EnKF_seed="$2"; shift 2 ;;
        --inflation) EnKF_inflation="$2"; shift 2 ;;
        --results_dir) results_dir="$2"; shift 2 ;;  # NEW ARG

        *) echo "Unknown parameter $1"; exit 1 ;;  # Exit on unknown argument
    esac
done

# results output directory
results_dir=""
#results_dir="Experiments/${rgi_id}/Ensemble_${exp_version}_${rgi_id}_${SMB_model}_${target_resolution}_${EnKF_ensemble_size}_${EnKF_iterations}_${EnKF_elev_band_height}m"


#####################################
#                                   #
# STEP 1: SETUP GENERATION          #
#         (input data, domain, ...) #
#                                   #
#####################################

# display the selected rgi_i
echo "Running pipeline for RGI ID: $rgi_id"
echo "with download: $download, inversion: $inversion, calibration: $calibration"

# 0. create folders
pushd  Scripts/Preprocess > /dev/null
python -u create_folder.py --rgi_id "$rgi_id" --SMB_model "SMB_model"
popd > /dev/null


# 1. Download data with OGGM_shop (if --download is set)
if [ "$download" = true ]; then
    echo "Downloading data..."
    pushd Scripts/Preprocess > /dev/null
    python -u download_data.py --rgi_id "$rgi_id" \
    --download_oggm_shop --download_hugonnet --year_interval 20 \
    --target_resolution "$target_resolution" \
    --hugonnet_directory "$hugonnet_storage_dir" \
    --SMB_model "$SMB_model"
    popd > /dev/null
fi


#####################################
#                                   #
# STEP 2: STATIONARY INVERSION      #
#         (ice dynamics, ...)       #
#                                   #
#####################################

# 2. IGM inversion step (if --inversion is set)
if [ "$inversion" = true ]; then
    echo "Starting IGM inversion..."
    pushd Scripts/Preprocess > /dev/null
    python -u igm_inversion.py --rgi_id "$rgi_id"
    popd > /dev/null
fi


#####################################
#                                   #
# STEP 3: TRANSIENT ASSIMILATION    #
#         (ELAs, melt params., ...) #
#                                   #
#####################################

# 3. Calibration step (if --calibrate is set)
if [ "$calibrate" = true ]; then
    echo "Starting calibration..."
    python -u FROST_calibration.py --rgi_id "$rgi_id" --ensemble_size "$EnKF_ensemble_size" \
    --forward_parallel "$EnKF_forward_parallel" --iterations "$EnKF_iterations" --seed "$EnKF_seed" \
    --inflation "$EnKF_inflation" --results_dir "$exp_version" \
    --init_offset "$EnKF_init_offset" --elevation_step "$EnKF_elev_band_height" --synthetic "$EnKF_synthetic" \
    --SMB_model "$SMB_model"
fi

echo "End of pipeline"