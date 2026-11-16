#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inputs of central_europe_retrain_t1: Preprocess (download, inversion,
# emulator) of each glacier from central_europe_retrain, the v2 reference
# with ice flow. The inversion does not retrain, so only the calibration
# differs. From the repo root:
#   bash experiments/central_europe_retrain_t1/copy_inputs.sh

SRC=data/results/central_europe_retrain
DST=data/results/central_europe_retrain_t1
while read -r RGI_ID NAME; do
    [ -n "$RGI_ID" ] || continue
    [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
    mkdir -p "$DST/$RGI_ID"
    cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
done < experiments/central_europe_retrain_t1/glaciers.txt
