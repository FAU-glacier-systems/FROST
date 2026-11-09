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
(rgi_id, glacier, p99, ice_flow, source) into experiments/<name>.

Run from the repository root:
    python experiments/central_europe_combined/make_combined.py
With the IGM glaciers from another run, e.g. the emulator retrained in the
forward runs:
    python experiments/central_europe_combined/make_combined.py \
        --igm data/results/central_europe_retrain --name central_europe_combined_retrain
Several IGM runs: the first that has the glacier. The TI calibration (slow
glaciers from central_europe_TI_slowflow, the others from central_europe_TI):
    python experiments/central_europe_combined/make_combined.py \
        --igm data/results/central_europe_TI_slowflow data/results/central_europe_TI \
        --no_flow data/results/central_europe_TI --name central_europe_TI_combined
"""

import argparse
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
IGM = os.path.join('data', 'results', 'central_europe_esmda_velyear')
NO_FLOW = os.path.join('data', 'results', 'central_europe_noflow')
MAX_P99 = 5.0  # m/yr; was 10 until 2026-10-29 (central_europe_TI_slowflow)


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
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--igm', nargs='+', default=[IGM],
                        help='runs of the glaciers with ice flow (the first that has the glacier)')
    parser.add_argument('--no_flow', default=NO_FLOW, help='run of the glaciers without ice flow')
    parser.add_argument('--name', default='central_europe_combined',
                        help='data/results/<name> (links), experiments/<name>/selection.tsv')
    args = parser.parse_args()
    target = os.path.join('data', 'results', args.name)
    out = os.path.join('experiments', args.name)
    os.makedirs(target, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(os.path.join(HERE, 'glaciers.txt')) as f:
        glaciers = [line.split()[:2] for line in f if line.strip()]
    for rgi_id, name in glaciers:
        # p99 from the inputs of the run without flow, or of the IGM run if
        # that one has none (Gietro: no download in central_europe_noflow)
        inputs = [os.path.join(run, rgi_id) for run in [args.no_flow] + args.igm
                  if os.path.exists(os.path.join(run, rgi_id, 'Preprocess', 'data', 'input.nc'))]
        if not inputs:
            print(f'{name}: no results, skipped')
            continue
        p99 = velocity_p99(inputs[0])
        flow = 'none' if p99 < MAX_P99 else 'igm'
        sources = [args.no_flow] if flow == 'none' else args.igm
        found = [run for run in sources
                 if os.path.exists(os.path.join(run, rgi_id, 'calibration_results.json'))]
        source = found[0] if found else sources[0]
        if not found:
            print(f'{name}: no results in {source}, skipped')
            continue
        link = os.path.join(target, rgi_id)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(os.path.relpath(os.path.join(source, rgi_id), target), link)
        rows.append(dict(rgi_id=rgi_id, glacier=name, p99=p99, ice_flow=flow, source=source))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, 'selection.tsv'), sep='\t', index=False, float_format='%.2f')
    print(table.to_string(index=False, float_format='%.1f'))


if __name__ == '__main__':
    main()
