#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Global ice thickness products for RGI region 11 (Central Europe), for
# compare_thickness_products.py. Run from the repository root:
#   bash experiments/inversion_setup/download_thickness_products.sh
# Millan et al. (2022) is not downloaded: it is in OGGM's gridded_data.nc of
# every glacier (millan_ice_thickness).

set -e
OUT=data/raw/thickness_products
mkdir -p "$OUT/farinotti2019_consensus/RGI60-11" "$OUT/iceboost_v2.0"

# Farinotti et al. (2019) consensus, one GeoTIFF per RGI6 glacier, from the
# OGGM server that oggm.shop.bedtopo.add_consensus_thickness uses
BASE=https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite/RGI60-11
FILES=$(curl -s "$BASE/" | grep -o 'RGI60-11\.[0-9]*_thickness\.tif' | sort -u)
for attempt in 1 2 3; do  # fetch what is still missing; single requests can fail
    for f in $FILES; do
        [ -s "$OUT/farinotti2019_consensus/RGI60-11/$f" ] || echo "$f"
    done | xargs -r -P 8 -I{} curl -s -f -o "$OUT/farinotti2019_consensus/RGI60-11/{}" "$BASE/{}" || true
done
echo "Farinotti: $(ls $OUT/farinotti2019_consensus/RGI60-11 | wc -l) of $(echo $FILES | wc -w) files"

# IceBoost v2.0 (Maffezzoli et al.), RGI7 region 11, Zenodo 10.5281/zenodo.17724512
ZIP=$OUT/iceboost_v2.0/RGI70G_rgi11.zip
curl -s -f -L -o "$ZIP" \
    "https://zenodo.org/records/17724512/files/RGI70G_rgi11.zip?download=1"
echo "IceBoost: $(du -h "$ZIP" | cut -f1)"
