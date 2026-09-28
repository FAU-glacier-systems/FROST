#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Regularisation sweep for the tau_ref inversion with the provided ContinuIX
thickness (params_inversion_tau.yaml): one inversion per glacier and lam,
then velocity misfit and tau_ref roughness per lam (L-curve) and maps.

Run from the repository root, on a GPU node:
    python experiments/continuix/tau_sweep.py --run --glaciers G03 G05
and the comparison on a login node:
    python experiments/continuix/tau_sweep.py --plot --glaciers G03 G05
"""

import argparse
import os
import shutil
import sys

import numpy as np
import yaml
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.preprocess import continuix, igm_inversion

LAMS = [1e9, 1e10, 1e11, 1e12]
OUT_DIR = os.path.join('data', 'results', 'continuix', 'tau_sweep')
PARAMS = os.path.join('experiments', 'continuix', 'params_inversion_tau.yaml')


def glacier_dir(exp, glacier, lam):
    return os.path.join(OUT_DIR, exp, glacier, f'lam_{lam:.0e}')


def run(exp, glacier, lams, cfg):
    base = os.path.join(OUT_DIR, exp, glacier, 'input')
    continuix.prepare_input(cfg['data_dir'], exp, glacier, base,
                            resolution=cfg['resolution'])
    with open(PARAMS) as f:
        params = yaml.safe_load(f)
    for lam in lams:
        rgi_id_dir = glacier_dir(exp, glacier, lam)
        data_dir = os.path.join(rgi_id_dir, 'Preprocess', 'data')
        os.makedirs(data_dir, exist_ok=True)
        shutil.copy(os.path.join(base, 'Preprocess', 'data', 'input.nc'),
                    data_dir)
        params['assimilations']['field_inversion']['objective'][
            'regularization'][0]['lam'] = float(lam)
        params_path = os.path.join(rgi_id_dir, 'params_inversion.yaml')
        with open(params_path, 'w') as f:
            f.write('# @package _global_\n')
            yaml.dump(params, f, sort_keys=False)
        igm_inversion.main(rgi_id_dir=rgi_id_dir,
                           params_inversion_path=params_path)


def results(exp, glacier, lam):
    rgi_id_dir = glacier_dir(exp, glacier, lam)
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'data',
                              'input.nc')) as nc:
        speed_obs = np.hypot(np.array(nc['uvelsurfobs'][:], dtype=float),
                             np.array(nc['vvelsurfobs'][:], dtype=float))
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                              'output.nc')) as nc:
        fields = {k: np.array(nc[k][:]) for k in
                  ['thk', 'tau_ref', 'velsurf_mag', 'icemask', 'x', 'y']}
    fields['speed_obs'] = speed_obs
    ice = fields['icemask'] > 0.5
    valid = ice & np.isfinite(speed_obs)
    log_tau = np.log10(fields['tau_ref'])
    lap = (np.roll(log_tau, 1, 0) + np.roll(log_tau, -1, 0)
           + np.roll(log_tau, 1, 1) + np.roll(log_tau, -1, 1) - 4 * log_tau)
    inner = ice & np.roll(ice, 1, 0) & np.roll(ice, -1, 0) \
        & np.roll(ice, 1, 1) & np.roll(ice, -1, 1)
    stats = {
        'lam': lam,
        'rms_speed': np.sqrt(np.mean((fields['velsurf_mag']
                                      - speed_obs)[valid] ** 2)),
        'ratio': fields['velsurf_mag'][valid].mean() / speed_obs[valid].mean(),
        'tau_median': np.median(fields['tau_ref'][ice]),
        'tau_p5': np.percentile(fields['tau_ref'][ice], 5),
        'tau_p95': np.percentile(fields['tau_ref'][ice], 95),
        'rough_log_tau': np.sqrt(np.mean(lap[inner] ** 2)),
    }
    return stats, fields


def plot(exp, glacier, lams):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rows = [results(exp, glacier, lam) for lam in lams]
    fig, axes = plt.subplots(len(lams), 4, figsize=(16, 3.6 * len(lams)),
                             squeeze=False)
    speed_obs = rows[0][1]['speed_obs']
    ice = rows[0][1]['icemask'] > 0.5
    vmax = np.nanpercentile(speed_obs[ice], 98)
    for (stats, f), ax in zip(rows, axes):
        extent = [f['x'][0], f['x'][-1], f['y'][0], f['y'][-1]]
        diff = f['velsurf_mag'] - speed_obs
        panels = [
            (speed_obs, 'observed speed (m/yr)',
             dict(cmap='magma', vmin=0, vmax=vmax)),
            (f['velsurf_mag'], 'IGM speed (m/yr)',
             dict(cmap='magma', vmin=0, vmax=vmax)),
            (diff, 'IGM - observed (m/yr)',
             dict(cmap='RdBu_r', vmin=-20, vmax=20)),
            (f['tau_ref'], 'tau_ref (MPa)',
             dict(cmap='viridis', norm=LogNorm(0.01, 2))),
        ]
        for a, (data, title, kw) in zip(ax, panels):
            im = a.imshow(np.where(ice, data, np.nan), origin='lower',
                          extent=extent, **kw)
            a.set_title(f"lam {stats['lam']:.0e}: {title}", fontsize=9)
            a.set_xticks([])
            a.set_yticks([])
            plt.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f'{glacier}: tau_ref inversion with the provided thickness')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, exp, f'tau_sweep_{glacier}.png'), dpi=110)
    plt.close(fig)
    return [stats for stats, _ in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--exp', default='EXP01')
    parser.add_argument('--glaciers', nargs='+', default=['G03', 'G05'])
    parser.add_argument('--lams', nargs='+', type=float, default=LAMS)
    parser.add_argument('--config', default='experiments/continuix/config.yml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for glacier in args.glaciers:
        if args.run:
            run(args.exp, glacier, args.lams, cfg)
        if args.plot:
            print(glacier)
            for s in plot(args.exp, glacier, args.lams):
                print('  ' + '  '.join(f'{k} {v:.3g}' for k, v in s.items()))


if __name__ == '__main__':
    main()
