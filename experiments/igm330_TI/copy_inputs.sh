#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Inputs of igm330_TI, so the TI calibration needs no download and no
# inversion: Preprocess of central_europe_retrain (download, tau_ref
# inversion, emulator; as igm330_live) for the 17 glaciers with ice flow, of
# central_europe_noflow (download, start geometry) for the 8 without; plus
# the OGGM climate (climate_historical.nc) where the TI runs read it. rsync,
# not cp: cp -a took minutes per glacier in an NFS server-side copy (2026-11-20).
# From the repo root:
#   bash experiments/igm330_TI/copy_inputs.sh
DST=data/results/igm330_TI
copy() {
    while read -r RGI_ID NAME; do
        [ -n "$RGI_ID" ] || continue
        [ -d "$1/$RGI_ID/Preprocess" ] || { echo "$NAME: no inputs in $1"; continue; }
        mkdir -p "$DST/$RGI_ID"
        rsync -a "$1/$RGI_ID/Preprocess" "$DST/$RGI_ID/"
        rsync -a "$1/$RGI_ID/Preprocess/data/$RGI_ID/climate_historical.nc" "$DST/$RGI_ID/"
    done < "$2"
}
copy data/results/central_europe_retrain experiments/igm330_TI/glaciers_igm.txt
copy data/results/central_europe_noflow experiments/igm330_TI/glaciers_noflow.txt
