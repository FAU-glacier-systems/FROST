#!/bin/bash
# Copy the inputs of central_europe_TI (Preprocess and the OGGM climate), so
# this variant needs no download and no inversion. From the repo root:
#   bash experiments/central_europe_TI_meltout/copy_inputs.sh
# Files already in the target are overwritten, none are deleted.
SRC=data/results/central_europe_TI
DST=data/results/central_europe_TI_meltout
while read -r RGI_ID NAME; do
    [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
    mkdir -p "$DST/$RGI_ID"
    cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
    cp "$SRC/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
done < experiments/central_europe_TI_meltout/glaciers.txt
