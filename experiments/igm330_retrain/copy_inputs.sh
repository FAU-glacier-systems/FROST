#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inputs of igm330_default and igm330_credit: Preprocess (download,
# inversion, emulator) of each glacier from central_europe_retrain, as
# central_europe_retrain_t1, so only IGM 3.3.0 and the retraining rule
# differ. From the repo root:
#   bash experiments/igm330_retrain/copy_inputs.sh

SRC=data/results/central_europe_retrain
for DST in data/results/igm330_default data/results/igm330_credit; do
    while read -r RGI_ID NAME; do
        [ -n "$RGI_ID" ] || continue
        [ -d "$SRC/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $SRC"; continue; }
        mkdir -p "$DST/$RGI_ID"
        cp -a "$SRC/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
    done < experiments/igm330_default/glaciers.txt
done
