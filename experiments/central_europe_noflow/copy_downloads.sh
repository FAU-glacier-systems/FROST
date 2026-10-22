#!/bin/bash
# Copy the downloads (Preprocess/data) of central_europe_esmda_velyear, so
# the no-flow calibration needs no OGGM shop run. From the repo root:
#   bash experiments/central_europe_noflow/copy_downloads.sh
# Files already in the target are overwritten, none are deleted.
SRC=data/results/central_europe_esmda_velyear
DST=data/results/central_europe_noflow
while read -r RGI_ID NAME; do
    [ -d "$SRC/$RGI_ID/Preprocess/data" ] || { echo "$NAME: no download"; continue; }
    mkdir -p "$DST/$RGI_ID/Preprocess/data"
    cp -a "$SRC/$RGI_ID/Preprocess/data/." "$DST/$RGI_ID/Preprocess/data/"
done < experiments/central_europe_noflow/glaciers.txt
