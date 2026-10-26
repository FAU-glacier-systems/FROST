#!/bin/bash
# Copy the inputs of the ELA runs, so the TI calibration needs no download
# and no inversion (neither depends on the SMB model): Preprocess of
# central_europe_retrain (download, tau_ref inversion, emulator) for the
# glaciers with ice flow, of central_europe_noflow (download, start
# geometry) for those without; plus the OGGM climate (climate_historical.nc)
# where the TI forward runs read it. From the repo root:
#   bash experiments/central_europe_TI/copy_inputs.sh
# Files already in the target are overwritten, none are deleted.
DST=data/results/central_europe_TI
copy() {
    while read -r RGI_ID NAME; do
        [ -d "$1/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $1"; continue; }
        mkdir -p "$DST/$RGI_ID"
        cp -a "$1/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
        cp "$1/$RGI_ID/Preprocess/data/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
    done < "$2"
}
copy data/results/central_europe_retrain experiments/central_europe_TI/glaciers_igm.txt
copy data/results/central_europe_noflow experiments/central_europe_TI/glaciers_noflow.txt
