#!/bin/bash
# Inputs of the slow glaciers with ice flow: Preprocess of
# central_europe_esmda_velyear (download, tau_ref inversion on the 2018
# geometry; identical to central_europe_retrain, see its STATUS) plus the
# OGGM climate (climate_historical.nc). From the repo root:
#   bash experiments/central_europe_TI_slowflow/copy_inputs.sh
SRC=data/results/central_europe_esmda_velyear
DST=data/results/central_europe_TI_slowflow
while read -r RGI_ID NAME; do
    [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
    mkdir -p "$DST/$RGI_ID"
    cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
    cp "$SRC/$RGI_ID/Preprocess/data/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
done < experiments/central_europe_TI_slowflow/glaciers.txt
