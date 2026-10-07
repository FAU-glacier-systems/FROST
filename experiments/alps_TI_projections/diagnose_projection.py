#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Why do some TI projections slow down to tiny time steps? Reruns one
projection of a calibrated glacier with yearly 2D output and reports, per
year, the fastest ice: speed, place, thickness, tau_ref, surface slope and
the time step its speed allows (dt = cfl * dx / speed, cfl = 0.3).

Run from the repository root (GPU job, see diagnose_projection_sbatch.sh):
    python experiments/alps_TI_projections/diagnose_projection.py \
        --rgi_id <id> --results <experiment results folder> --label <label> \
        [--inversion <variant results folder>] [--max_velbar 1000]
Writes data/results/alps_TI_projections/diagnostics/<label>/<rgi_id>/
    yearly.tsv, fastest.png
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(EXPERIMENT_DIR)))
from frost.glacier_model import igm_wrapper
from frost.preprocess.igm_inversion import emulator_path

CFL = 0.3


def run(rgi_id_dir, inversion_dir, run_dir, forcing, year_end, max_velbar):
    with open(os.path.join(rgi_id_dir, 'calibration_results.json')) as f:
        calibration = json.load(f)
    smb = dict(zip(calibration['initial_smb'], calibration['final_mean']))
    with Dataset(os.path.join(rgi_id_dir, 'observations.nc')) as ds:
        usurf = np.array(ds['usurf'][:], dtype=np.float64)
        year_start = int(np.array(ds['time'][:])[0])
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(os.path.join(run_dir, 'experiment'))
    os.makedirs(os.path.join(run_dir, 'data'))
    shutil.copy2(os.path.join(inversion_dir, 'Preprocess', 'outputs', 'output.nc'),
                 os.path.join(run_dir, 'data', 'input.nc'))
    igm_wrapper.forward('Projection', True, True, 0, 'TI', usurf, smb,
                        year_start, year_end, run_dir, forcing,
                        os.path.abspath(emulator_path(inversion_dir)), max_velbar)
    return smb


def analyse(inversion_dir, run_dir, label):
    with Dataset(os.path.join(inversion_dir, 'Preprocess', 'outputs', 'output.nc')) as ds:
        tau_ref = np.array(ds['tau_ref'][:])
        x = np.array(ds['x'][:])
        y = np.array(ds['y'][:])
    dx = abs(float(x[1] - x[0]))
    with Dataset(os.path.join(run_dir, 'outputs', 'output.nc')) as ds:
        time = np.array(ds['time'][:])
        thk = np.array(ds['thk'][:])
        usurf = np.array(ds['usurf'][:])
        speed = np.array(ds['velsurf_mag'][:])
        icemask = np.array(ds['icemask_init'][0] if 'icemask_init' in ds.variables
                           else ds['icemask'][0]) > 0.5

    rows = []
    for k, year in enumerate(time):
        s = np.where(thk[k] > 1, speed[k], 0)
        iy, ix = np.unravel_index(np.argmax(s), s.shape)
        gy, gx = np.gradient(usurf[k], dx)
        ice = thk[k] > 1
        rows.append(dict(
            year=float(year), volume_km3=float(thk[k][ice].sum() * dx * dx / 1e9),
            speed_max=float(s[iy, ix]),
            speed_p99=float(np.percentile(s[ice], 99)) if ice.any() else 0.0,
            speed_median=float(np.median(s[ice])) if ice.any() else 0.0,
            dt_cfl=float(CFL * dx / max(s[iy, ix], 1e-6)),
            n_over_1km_per_yr=int((s > 1000).sum()),
            fastest_thk=float(thk[k][iy, ix]),
            fastest_tau_ref=float(tau_ref[iy, ix]),
            fastest_slope=float(np.hypot(gx[iy, ix], gy[iy, ix])),
            fastest_in_outline=bool(icemask[iy, ix]),
            fastest_row=int(iy), fastest_col=int(ix)))
    table = pd.DataFrame(rows)
    out_dir = os.path.dirname(run_dir)
    table.to_csv(os.path.join(out_dir, 'yearly.tsv'), sep='\t', index=False,
                 float_format='%.4g')
    print(f'== {label}: {inversion_dir}')
    print(table.drop(columns=['fastest_row', 'fastest_col']).to_string(
        index=False, float_format='%.3g'))

    # maps at the year of the fastest ice
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = int(table.speed_max.idxmax())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    origin = 'lower' if y[-1] > y[0] else 'upper'
    fields = [(np.where(thk[k] > 1, speed[k], np.nan), f'Surface speed {time[k]:.0f} (m/yr)', 'magma', True),
              (np.where(thk[k] > 1, thk[k], np.nan), f'Thickness {time[k]:.0f} (m)', 'Blues', False),
              (np.where(icemask, tau_ref, np.nan), 'tau_ref (MPa)', 'viridis', False),
              (np.where(thk[0] > 1, speed[0], np.nan), f'Surface speed {time[0]:.0f} (m/yr)', 'magma', True)]
    for ax, (field, title, cmap, log) in zip(axes, fields):
        if log:
            from matplotlib.colors import LogNorm
            im = ax.imshow(field, origin=origin, cmap=cmap,
                           norm=LogNorm(vmin=1, vmax=max(np.nanmax(field), 10)))
        else:
            im = ax.imshow(field, origin=origin, cmap=cmap)
        ax.plot(table.fastest_col[k], table.fastest_row[k], 'c+', ms=14, mew=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'{label}: fastest ice (+) in the year of the maximum speed')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fastest.png'), dpi=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgi_id', required=True)
    parser.add_argument('--results', default='data/results/alps_TI_projections',
                        help='results folder of the inversion/calibration to test')
    parser.add_argument('--inversion', default=None,
                        help='folder with the inversion (Preprocess/outputs) to use '
                             'instead of the one in --results')
    parser.add_argument('--max_velbar', type=float, default=0.0,
                        help='cap of the depth-averaged speed (m/yr), 0 for none')
    parser.add_argument('--label', required=True)
    parser.add_argument('--end_year', type=int, default=2035)
    parser.add_argument('--scenario', default='rcp_8_5')
    args = parser.parse_args()

    rgi_id_dir = os.path.join(args.results, args.rgi_id)
    forcing = sorted(glob.glob(os.path.join(rgi_id_dir, 'Projection', 'forcing',
                                            f'climate_{args.scenario}_*.nc')))[0]
    run_dir = os.path.join('data', 'results', 'alps_TI_projections', 'diagnostics',
                           args.label, args.rgi_id, 'run')
    inversion_dir = os.path.join(args.inversion, args.rgi_id) if args.inversion \
        else rgi_id_dir
    smb = run(rgi_id_dir, inversion_dir, os.path.abspath(run_dir), forcing,
              args.end_year, args.max_velbar)
    print(f'SMB {smb}, forcing {os.path.basename(forcing)}, '
          f'max_velbar {args.max_velbar}')
    analyse(inversion_dir, run_dir, args.label)
