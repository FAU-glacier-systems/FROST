#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Sweep of the fixed sliding parameter tau_ref in the test_default thickness
inversion on Rhone and Aletsch. Each value is scored against the thickness
observations (GlaThiDa, not used by the inversion) and the surface velocities.

Run from the repository root, the inversions on a GPU node:
    python experiments/tau_ref_sweep/tau_ref_sweep.py --run --glaciers RGI2000-v7.0-G-11-01706
and the comparison on a login node:
    python experiments/tau_ref_sweep/tau_ref_sweep.py --plot
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from netCDF4 import Dataset

# run from the repository root, which holds the frost package
sys.path.insert(0, os.getcwd())
from frost.paths import DATA_RESULTS
from frost.preprocess import igm_inversion
from frost.preprocess.thickness_validation import read_field

HERE = Path(__file__).resolve().parent
GLACIERS = {'RGI2000-v7.0-G-11-01706': 'Rhone',
            'RGI2000-v7.0-G-11-02596': 'Aletsch'}
TAU_REFS = [0.05, 0.1, 0.15, 0.213, 0.3, 0.5, 1.0, 2.0]  # MPa, at u_ref
PARAMS = HERE.parent / 'test_default' / 'params_inversion.yaml'
SOURCE = DATA_RESULTS / 'test_default'
OUT_DIR = DATA_RESULTS / 'tau_ref_sweep'


def case_dir(rgi_id, tau_ref):
    return OUT_DIR / rgi_id / f'tau_ref_{tau_ref:.3f}'


def run(rgi_id, tau_ref):
    rgi_id_dir = case_dir(rgi_id, tau_ref)
    data_dir = rgi_id_dir / 'Preprocess' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE / rgi_id / 'Preprocess' / 'data' / 'input.nc',
                data_dir / 'input.nc')
    with open(PARAMS) as f:
        params = yaml.safe_load(f)
    params['processes']['iceflow']['physics']['sliding']['tau_ref'] = tau_ref
    params_path = rgi_id_dir / 'params_inversion.yaml'
    with open(params_path, 'w') as f:
        f.write('# @package _global_\n')
        yaml.dump(params, f, sort_keys=False)
    igm_inversion.main(rgi_id_dir=str(rgi_id_dir),
                       params_inversion_path=str(params_path))


def results(rgi_id, tau_ref):
    rgi_id_dir = case_dir(rgi_id, tau_ref)
    outputs = rgi_id_dir / 'Preprocess' / 'outputs'
    with open(outputs / 'thickness_validation.json') as f:
        validation = json.load(f)
    with Dataset(rgi_id_dir / 'Preprocess' / 'data' / 'input.nc') as nc:
        speed_obs = np.hypot(read_field(nc, 'uvelsurfobs'),
                             read_field(nc, 'vvelsurfobs'))
    speed_obs[speed_obs == 0] = np.nan  # zeros are gaps
    with Dataset(outputs / 'output.nc') as nc:
        ice = read_field(nc, 'icemask') > 0.5
        thk = read_field(nc, 'thk')
        speed = read_field(nc, 'velsurf_mag')
        speed_base = read_field(nc, 'velbase_mag')
    valid = ice & np.isfinite(speed_obs)
    return {
        'glacier': GLACIERS[rgi_id], 'tau_ref': tau_ref,
        'thk_mean': float(thk[ice].mean()),
        'glathida_bias': validation['bias'],
        'glathida_rms': validation['rms'],
        'speed_ratio': float(speed[valid].mean() / speed_obs[valid].mean()),
        'speed_rms': float(np.sqrt(np.mean((speed - speed_obs)[valid] ** 2))),
        'sliding_share': float(speed_base[valid].mean() / speed[valid].mean()),
        'bands': validation['bands'],
    }


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for name in GLACIERS.values():
        sub = [r for r in rows if r['glacier'] == name]
        tau = [r['tau_ref'] for r in sub]
        for ax, key in zip(axes, ['glathida_bias', 'glathida_rms',
                                  'speed_ratio', 'sliding_share']):
            ax.plot(tau, [r[key] for r in sub], 'o-', label=name)
    labels = ['GlaThiDa bias (m)', 'GlaThiDa RMS (m)',
              'Modelled / observed speed', 'Sliding share of surface speed']
    for ax, label in zip(axes, labels):
        ax.set_xscale('log')
        ax.set_xlabel('tau_ref (MPa)')
        ax.set_ylabel(label)
        ax.axvline(0.213, color='grey', ls=':', lw=1)
        ax.grid(alpha=0.3)
    axes[0].axhline(0, color='k', lw=0.8)
    axes[2].axhline(1, color='k', lw=0.8)
    axes[0].legend()
    fig.tight_layout()
    (HERE / 'plots').mkdir(exist_ok=True)
    fig.savefig(HERE / 'plots' / 'tau_ref_sweep.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--glaciers', nargs='+', default=list(GLACIERS))
    parser.add_argument('--tau_refs', type=float, nargs='+', default=TAU_REFS)
    args = parser.parse_args()
    if args.run:
        for rgi_id in args.glaciers:
            for tau_ref in args.tau_refs:
                run(rgi_id, tau_ref)
    if args.plot:
        rows = [results(rgi_id, tau_ref) for rgi_id in GLACIERS
                for tau_ref in TAU_REFS
                if (case_dir(rgi_id, tau_ref) / 'Preprocess' / 'outputs'
                    / 'thickness_validation.json').exists()]
        keys = ['glacier', 'tau_ref', 'thk_mean', 'glathida_bias',
                'glathida_rms', 'speed_ratio', 'speed_rms', 'sliding_share']
        lines = ['\t'.join(keys)] + ['\t'.join(
            f'{r[k]:.3g}' if isinstance(r[k], float) else str(r[k])
            for k in keys) for r in rows]
        (HERE / 'tables').mkdir(exist_ok=True)
        (HERE / 'tables' / 'tau_ref_sweep.tsv').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        plot(rows)


if __name__ == '__main__':
    main()
