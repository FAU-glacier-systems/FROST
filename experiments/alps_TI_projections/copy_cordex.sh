#!/bin/bash
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

# Copy the CORDEX forcing of every glacier in glacier_selection.csv from
# Johannes' vault to data/raw/cordex_alps/<last 5 digits of rgi_id>/.
# Run in the repo root on a node that mounts /home/vault (e.g. Alex login):
#   bash experiments/alps_TI_projections/copy_cordex.sh

SRC=/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids
DST=data/raw/cordex_alps
LIST=experiments/alps_TI_projections/glacier_selection.csv

copied=0; skipped=0; missing=()
for rgi_id in $(tail -n +2 "$LIST" | cut -d, -f1); do
    id=${rgi_id: -5}
    src="$SRC/$id/climate/CORDEX_unbiased/CORDEX_merged.nc"
    dst="$DST/$id/CORDEX_merged.nc"
    if [ ! -f "$src" ]; then
        missing+=("$rgi_id")
    elif [ -f "$dst" ] && [ "$(stat -c %s "$src")" = "$(stat -c %s "$dst")" ]; then
        skipped=$((skipped + 1))
    else
        mkdir -p "$DST/$id"
        cp "$src" "$dst" && copied=$((copied + 1))
    fi
done

echo "copied: $copied, already there: $skipped, missing: ${#missing[@]}"
printf '%s\n' "${missing[@]}" > "$DST/missing.txt"
[ ${#missing[@]} -gt 0 ] && echo "missing rgi_ids in $DST/missing.txt"
