#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann, Johannes J. Fürst
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
TI projections of one calibrated glacier: the posterior-mean SMB parameters
run from the observed start surface (2000) to 2100 under every complete
CORDEX run (RCP2.6/4.5/8.5). Writes the yearly volume and area of all runs
to <glacier>/Projection/projections.nc.

Run from the repository root:
    python experiments/alps_TI_projections/run_projections.py --rgi_id <id>
"""

import argparse
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr
from netCDF4 import Dataset

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
# the experiment folder (cordex_forcing) and the repository root (frost)
sys.path[:0] = [EXPERIMENT_DIR, os.path.dirname(os.path.dirname(EXPERIMENT_DIR))]
from cordex_forcing import read_cordex, write_forcing
from frost.glacier_model import igm_wrapper
from frost.preprocess.igm_inversion import emulator_path


def run_one(rgi_id_dir, run_dir, forcing, smb, usurf, year_start, year_end,
            emulator, max_velbar=1000.0):
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(os.path.join(run_dir, 'experiment'))
    os.makedirs(os.path.join(run_dir, 'data'))
    shutil.copy2(os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'output.nc'),
                 os.path.join(run_dir, 'data', 'input.nc'))
    igm_wrapper.forward('Projection', True, False, 0, 'TI', usurf, smb,
                        year_start, year_end, run_dir, forcing, emulator,
                        max_velbar)
    ts_file = os.path.join(run_dir, 'outputs', 'output_ts.nc')
    with Dataset(ts_file) as ds:
        return (np.array(ds['time'][:]), np.array(ds['vol'][:]),
                np.array(ds['area'][:]))


def collect(projection_dir, year_end):
    """Time series of the runs already in projection_dir; runs that did not
    reach year_end (e.g. stopped by a job time limit) are left out."""
    results = {}
    for ts_file in glob.glob(os.path.join(projection_dir, 'rcp_*', '*', 'outputs',
                                          'output_ts.nc')):
        run_dir = os.path.dirname(os.path.dirname(ts_file))
        key = (os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir))
        with Dataset(ts_file) as ds:
            series = (np.array(ds['time'][:]), np.array(ds['vol'][:]),
                      np.array(ds['area'][:]))
        if series[0][-1] >= year_end:
            results[key] = series
    return results


def main(rgi_id, experiment_name, cordex_file, year_end, workers,
         collect_only=False, max_velbar=1000.0):
    rgi_id_dir = os.path.join('data', 'results', experiment_name, rgi_id)
    projection_dir = os.path.join(rgi_id_dir, 'Projection')

    with open(os.path.join(rgi_id_dir, 'calibration_results.json')) as f:
        calibration = json.load(f)
    smb = dict(zip(calibration['initial_smb'], calibration['final_mean']))
    print(f'Posterior-mean SMB: {smb}')

    with Dataset(os.path.join(rgi_id_dir, 'observations.nc')) as ds:
        usurf = np.array(ds['usurf'][:], dtype=np.float64)
        year_start = int(np.array(ds['time'][:])[0])

    if collect_only:
        results = collect(projection_dir, year_end)
        write_projections(projection_dir, results, rgi_id, smb, None)
        return

    years, runs = read_cordex(cordex_file)
    jobs = []
    for scenario, member, temp, prcp in runs:
        forcing = os.path.join(projection_dir, 'forcing',
                               f'climate_{scenario}_{member}.nc')
        write_forcing(os.path.join(rgi_id_dir, 'climate_historical.nc'),
                      years, temp, prcp, forcing, year_end=year_end)
        jobs.append((scenario, member, forcing,
                     os.path.join(projection_dir, scenario, member)))
    print(f'{len(jobs)} CORDEX runs')

    emulator = os.path.abspath(emulator_path(rgi_id_dir))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, rgi_id_dir, run_dir, forcing,
                                   smb, usurf, year_start, year_end,
                                   emulator, max_velbar): (scenario, member)
                   for scenario, member, forcing, run_dir in jobs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:
                print(f'Failed {key}: {error}')

    write_projections(projection_dir, results, rgi_id, smb, len(jobs))


def write_projections(projection_dir, results, rgi_id, smb, n_jobs):
    """One file: vol/area over (run, time), scenario and model per run."""
    keys = sorted(results)
    time = results[keys[0]][0]
    vol = np.full((len(keys), len(time)), np.nan)
    area = np.full_like(vol, np.nan)
    for i, key in enumerate(keys):
        n = min(len(time), len(results[key][0]))
        vol[i, :n] = results[key][1][:n]
        area[i, :n] = results[key][2][:n]
    ds = xr.Dataset(
        {'vol': (('run', 'time'), vol, {'units': 'km3', 'long_name': 'Ice volume'}),
         'area': (('run', 'time'), area, {'units': 'km2', 'long_name': 'Glaciated area'})},
        coords={'time': time,
                'scenario': ('run', [k[0] for k in keys]),
                'model': ('run', [k[1] for k in keys])})
    ds.attrs.update(rgi_id=rgi_id, smb=json.dumps(smb),
                    smb_parameters='posterior mean of calibration_results.json')
    out_file = os.path.join(projection_dir, 'projections.nc')
    ds.to_netcdf(out_file)
    print(f'Wrote {out_file} ({len(keys)} of {n_jobs or "?"} runs)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TI projections of one glacier.')
    parser.add_argument('--rgi_id', default='RGI2000-v7.0-G-11-01706')
    parser.add_argument('--experiment_name', default='alps_TI_projections')
    parser.add_argument('--cordex_dir', default='data/raw/cordex_alps',
                        help='Folder with <last 5 digits of rgi_id>/CORDEX_merged.nc')
    parser.add_argument('--year_end', type=int, default=2100)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--max_velbar', type=float, default=1000.0,
                        help='cap of the depth-averaged speed (m/yr), 0 for none')
    parser.add_argument('--collect_only', action='store_true',
                        help='only collect the finished runs into projections.nc')
    args = parser.parse_args()

    main(rgi_id=args.rgi_id, experiment_name=args.experiment_name,
         cordex_file=os.path.join(args.cordex_dir, args.rgi_id[-5:],
                                  'CORDEX_merged.nc'),
         year_end=args.year_end, workers=args.workers,
         collect_only=args.collect_only, max_velbar=args.max_velbar)
