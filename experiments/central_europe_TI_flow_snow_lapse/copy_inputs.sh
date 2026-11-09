#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inputs of central_europe_TI_flow_snow_lapse: Preprocess (inversion,
# emulator) and climate of each glacier from the run it has in
# central_europe_TI_combined (selection.tsv: central_europe_TI for the fast
# glaciers, central_europe_TI_slowflow for p99 5-10 m/yr). From the repo root:
#   bash experiments/central_europe_TI_flow_snow_lapse/copy_inputs.sh
DST=data/results/central_europe_TI_flow_snow_lapse
awk -F'\t' 'NR > 1 && $4 == "igm" {print $1, $2, $5}' experiments/central_europe_TI_combined/selection.tsv |
while read -r RGI_ID NAME SRC; do
    [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
    mkdir -p "$DST/$RGI_ID"
    cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
    cp "$SRC/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
done
