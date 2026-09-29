#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Regularisation sweep of the thickness inversion (test_default setup: SIA
start, velocity misfit only, squared Laplacian of the bed) at the model
resolution and at 100 m as in the igm-examples Aletsch case. GlaThiDa
thickness (thkobs, not used by the inversion) validates the result.

The Laplacian penalty scales with 1/dx^4 for grid-scale roughness, so the
lam of the 100 m example is ~16x stronger at 50 m.

Run from the repository root, on a GPU node:
    python experiments/thk_reg_sweep/reg_sweep.py --run
and the comparison on a login node:
    python experiments/thk_reg_sweep/reg_sweep.py --plot
"""

import argparse
import os
import shutil
import sys

import numpy as np
import xarray as xr
import yaml
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.preprocess import igm_inversion

RGI_ID = 'RGI2000-v7.0-G-11-01706'
SOURCE = os.path.join('data', 'results', 'test_default', RGI_ID,
                      'Preprocess', 'data', 'input.nc')
PARAMS = os.path.join('experiments', 'test_default', 'params_inversion.yaml')
OUT_DIR = os.path.join('data', 'results', 'thk_reg_sweep')
# (resolution factor, lam): 50 m sweep and the example's lam at 100 m
CASES = [(1, 1e4), (1, 6.25e4), (1, 2.5e5), (1, 1e6), (2, 1e6)]


def case_dir(factor, lam):
    return os.path.join(OUT_DIR, f'dx{factor}_lam{lam:.2e}')


def coarsen(src, dst, factor):
    """Block average; ice where at least half the cells are ice."""
    ds = xr.open_dataset(src).load()
    # fill values (> 1e35, masked by IGM on input) must not enter the means
    for name in ds.data_vars:
        if np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].where(ds[name] < 1e35)
    out = ds.coarsen(x=factor, y=factor, boundary='trim').mean()
    out['icemask'] = (out['icemask'] >= 0.5).astype('int8')
    out.attrs = ds.attrs
    out.to_netcdf(dst)


def run(factor, lam):
    rgi_id_dir = case_dir(factor, lam)
    data_dir = os.path.join(rgi_id_dir, 'Preprocess', 'data')
    os.makedirs(data_dir, exist_ok=True)
    if factor == 1:
        shutil.copy(SOURCE, os.path.join(data_dir, 'input.nc'))
    else:
        coarsen(SOURCE, os.path.join(data_dir, 'input.nc'), factor)
    with open(PARAMS) as f:
        params = yaml.safe_load(f)
    params['assimilations']['field_inversion']['objective'][
        'regularization'][0]['lam'] = float(lam)
    params_path = os.path.join(rgi_id_dir, 'params_inversion.yaml')
    with open(params_path, 'w') as f:
        f.write('# @package _global_\n')
        yaml.dump(params, f, sort_keys=False)
    igm_inversion.main(rgi_id_dir=rgi_id_dir, params_inversion_path=params_path)


def results(factor, lam):
    rgi_id_dir = case_dir(factor, lam)
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'data',
                              'input.nc')) as nc:
        thkobs = np.ma.filled(nc['thkobs'][:].astype(float), np.nan)
        speed_obs = np.hypot(np.ma.filled(nc['uvelsurfobs'][:].astype(float), np.nan),
                             np.ma.filled(nc['vvelsurfobs'][:].astype(float), np.nan))
        dx = float(nc['x'][1] - nc['x'][0])
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                              'output.nc')) as nc:
        fields = {k: np.array(nc[k][:]) for k in
                  ['thk', 'topg', 'icemask', 'velsurf_mag', 'x', 'y']}
        costs = {k: float(nc[k][...]) for k in
                 ['da_cost_data', 'da_cost_reg'] if k in nc.variables}
    ice = fields['icemask'] > 0.5
    speed_obs[speed_obs > 1e35] = np.nan
    valid = ice & np.isfinite(speed_obs)
    glathida = ice & np.isfinite(thkobs) & (thkobs > 0)
    diff = fields['thk'][glathida] - thkobs[glathida]
    topg = fields['topg']
    lap = (np.roll(topg, 1, 0) + np.roll(topg, -1, 0) + np.roll(topg, 1, 1)
           + np.roll(topg, -1, 1) - 4 * topg) / dx ** 2
    inner = ice & np.roll(ice, 1, 0) & np.roll(ice, -1, 0) \
        & np.roll(ice, 1, 1) & np.roll(ice, -1, 1)
    stats = {
        'dx': dx, 'lam': lam,
        'thk_mean': fields['thk'][ice].mean(),
        'speed_ratio': fields['velsurf_mag'][valid].mean()
        / speed_obs[valid].mean(),
        'speed_rms': np.sqrt(np.mean((fields['velsurf_mag']
                                      - speed_obs)[valid] ** 2)),
        'glathida_bias': diff.mean(),
        'glathida_rms': np.sqrt(np.mean(diff ** 2)),
        'glathida_n': int(glathida.sum()),
        'bed_curvature_rms': np.sqrt(np.mean(lap[inner] ** 2)),
        **costs,
    }
    fields['thkobs'] = thkobs
    return stats, fields


def plot(rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 4.2))
    vmax = max(np.nanpercentile(f['thk'][f['icemask'] > 0.5], 99)
               for _, f in rows)
    for ax, (s, f) in zip(axes, rows):
        ice = f['icemask'] > 0.5
        im = ax.imshow(np.where(ice, f['thk'], np.nan), origin='lower',
                       cmap='Blues', vmin=0, vmax=vmax,
                       extent=[f['x'][0], f['x'][-1], f['y'][0], f['y'][-1]])
        ax.set_title(f"dx {s['dx']:.0f} m, lam {s['lam']:.1e}\n"
                     f"GlaThiDa bias {s['glathida_bias']:+.0f} m",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label='thickness (m)')
    fig.savefig(os.path.join(OUT_DIR, 'reg_sweep_thk.png'), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--cases', type=int, nargs='+',
                        help='indices into CASES (default: all)')
    args = parser.parse_args()
    cases = [CASES[i] for i in args.cases] if args.cases else CASES
    if args.run:
        for factor, lam in cases:
            run(factor, lam)
    if args.plot:
        rows = [results(factor, lam) for factor, lam in CASES
                if os.path.exists(os.path.join(case_dir(factor, lam),
                                               'Preprocess', 'outputs',
                                               'output.nc'))]
        keys = list(rows[0][0].keys())
        lines = ['\t'.join(keys)] + ['\t'.join(
            f'{s[k]:.3g}' if k in s else '' for k in keys) for s, _ in rows]
        with open(os.path.join(OUT_DIR, 'reg_sweep.tsv'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))
        plot(rows)


if __name__ == '__main__':
    main()
