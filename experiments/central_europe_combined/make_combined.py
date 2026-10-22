#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Combined run: ice flow only where the glacier moves. Glaciers whose ITS_LIVE
speed on the ice has a 99th percentile below MAX_P99 take the calibration
without ice flow (central_europe_noflow: dh/dt = smb with melt-out), the
others the IGM calibration (central_europe_esmda_velyear). In the IGM runs
the flux divergence of the slow glaciers was 3-24 % of the SMB, and on them
the SMB profiles without flow fit GLAMOS / WGMS as well (RMSE 0.77 against
0.84 m w.e./yr); on the fast ones not (1.64 against 0.94).

data/results/central_europe_combined/<rgi_id> is a symbolic link to the
glacier's folder in the chosen run, so evaluate.py, evaluation.py and
sanity_report.py run on it as on any experiment. Writes selection.tsv
(rgi_id, glacier, p99, ice_flow, source).

Run from the repository root:
    python experiments/central_europe_combined/make_combined.py
"""

import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
IGM = os.path.join('data', 'results', 'central_europe_esmda_velyear')
NO_FLOW = os.path.join('data', 'results', 'central_europe_noflow')
TARGET = os.path.join('data', 'results', 'central_europe_combined')
MAX_P99 = 10.0  # m/yr


def velocity_p99(rgi_dir):
    """99th percentile of the observed speed on the ice (input.nc)."""
    with Dataset(os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc')) as ds:
        ice = np.array(ds['icemask'][:]) > 0.5
        speed = np.hypot(np.array(ds['uvelsurfobs'][:], dtype=float),
                         np.array(ds['vvelsurfobs'][:], dtype=float))
    speed = speed[ice]
    speed = speed[np.isfinite(speed) & (speed > 0)]
    return float(np.percentile(speed, 99)) if speed.size else 0.0


def main():
    os.makedirs(TARGET, exist_ok=True)
    rows = []
    with open(os.path.join(HERE, 'glaciers.txt')) as f:
        glaciers = [line.split()[:2] for line in f if line.strip()]
    for rgi_id, name in glaciers:
        if not os.path.exists(os.path.join(NO_FLOW, rgi_id, 'calibration_results.json')):
            print(f'{name}: no results, skipped')
            continue
        p99 = velocity_p99(os.path.join(NO_FLOW, rgi_id))
        flow = 'none' if p99 < MAX_P99 else 'igm'
        source = NO_FLOW if flow == 'none' else IGM
        if not os.path.exists(os.path.join(source, rgi_id, 'calibration_results.json')):
            print(f'{name}: no results in {source}, skipped')
            continue
        link = os.path.join(TARGET, rgi_id)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(os.path.relpath(os.path.join(source, rgi_id), TARGET), link)
        rows.append(dict(rgi_id=rgi_id, glacier=name, p99=p99, ice_flow=flow, source=source))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(HERE, 'selection.tsv'), sep='\t', index=False, float_format='%.2f')
    print(table.to_string(index=False, float_format='%.1f'))


if __name__ == '__main__':
    main()
