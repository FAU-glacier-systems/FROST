#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inputs of central_europe_TI_snow_lapse: the Preprocess (start geometry) and the
# climate of the no-flow glaciers of central_europe_TI. From the repo root:
#   bash experiments/central_europe_TI_snow_lapse/copy_inputs.sh
SRC=data/results/central_europe_TI
DST=data/results/central_europe_TI_snow_lapse
while read -r RGI_ID NAME; do
    [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
    mkdir -p "$DST/$RGI_ID"
    cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
    cp "$SRC/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
done < experiments/central_europe_TI_snow_lapse/glaciers.txt
