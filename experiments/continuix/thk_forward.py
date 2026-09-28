#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Surface velocities of IGM for the provided ContinuIX thickness, compared
with the observed velocities.

Per glacier: ContinuIX input on the model grid (as run_continuix.py), the
provided THK as geometry, one IGM ice-flow solve with the physics of the
inversion (params_inversion.yaml: sliding, emulator fine-tuned for
nbit_init iterations on this geometry). No inversion, no SMB.

Run from the repository root, on a GPU node:
    python experiments/continuix/thk_forward.py --run
and the comparison (plots, table) on a login node:
    python experiments/continuix/thk_forward.py --plot
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.preprocess import continuix

GLACIERS = ['G02', 'G03', 'G04', 'G05', 'G06', 'S01', 'S02']
OUT_DIR = os.path.join('data', 'results', 'continuix', 'thk_forward')


def run(glacier, cfg, exp):
    glacier_dir = os.path.join(OUT_DIR, exp, glacier)
    continuix.prepare_input(cfg['data_dir'], exp, glacier, glacier_dir,
                            resolution=cfg['resolution'])
    workdir = os.path.join(glacier_dir, 'forward')
    os.makedirs(os.path.join(workdir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(workdir, 'experiment'), exist_ok=True)
    input_file = os.path.join(workdir, 'data', 'input.nc')
    shutil.copy(os.path.join(glacier_dir, 'Preprocess', 'data', 'input.nc'),
                input_file)
    with Dataset(input_file, 'r+') as nc:
        topg = nc.createVariable('topg', 'f4', ('y', 'x'))
        topg[:] = nc['usurf'][:] - nc['thk'][:]

    # Ice flow exactly as in the inversion
    with open(os.path.join('experiments', 'continuix',
                           'params_inversion.yaml')) as f:
        iceflow = yaml.safe_load(f)['processes']['iceflow']
    params = {
        'hydra': {'run': {'dir': 'outputs/run'}},
        'core': {'url_data': ''},
        'defaults': [
            {'override /inputs': ['local']},
            {'override /processes': ['smb', 'iceflow', 'time', 'thk']},
            {'override /outputs': ['write_ncdf']},
        ],
        'inputs': {'local': {'filename': 'input.nc'}},
        'processes': {
            'iceflow': iceflow,
            # zero SMB on the ice; only the velocities at t = 0 are used
            'smb': {'method': 'simple', 'simple': {'array': [
                ['time', 'gradabl', 'gradacc', 'ela', 'accmax'],
                [0, 0.0, 0.0, 0.0, 0.0], [1, 0.0, 0.0, 0.0, 0.0]]}},
            'time': {'start': 0.0, 'end': 1.0, 'save': 1.0},
        },
        'outputs': {'write_ncdf': {
            'output_file': '../output.nc',
            'vars_to_save': ['thk', 'usurf', 'icemask', 'velsurf_mag',
                             'uvelsurf', 'vvelsurf']}},
    }
    with open(os.path.join(workdir, 'experiment', 'params.yaml'), 'w') as f:
        f.write('# @package _global_\n')
        yaml.dump(params, f, sort_keys=False)
    subprocess.run(['igm_run', '+experiment=params'], cwd=workdir, check=True)


def compare(glacier, exp):
    glacier_dir = os.path.join(OUT_DIR, exp, glacier)
    with Dataset(os.path.join(glacier_dir, 'Preprocess', 'data',
                              'input.nc')) as nc:
        uobs = np.array(nc['uvelsurfobs'][:], dtype=float)
        vobs = np.array(nc['vvelsurfobs'][:], dtype=float)
        icemask = np.array(nc['icemask'][:]) > 0
        x = np.array(nc['x'][:])
        y = np.array(nc['y'][:])
    with Dataset(os.path.join(glacier_dir, 'forward', 'outputs',
                              'output.nc')) as nc:
        thk = np.array(nc['thk'][0])
        u = np.array(nc['uvelsurf'][0])
        v = np.array(nc['vvelsurf'][0])
    # the FROST run with the inverted thickness, if there is one
    inverted = os.path.join('data', 'results', 'continuix', exp, 'glaciers',
                            glacier, 'Preprocess', 'outputs', 'output.nc')
    speed_inv = thk_inv = None
    if os.path.exists(inverted):
        with Dataset(inverted) as nc:
            speed_inv = np.array(nc['velsurf_mag'][:])
            thk_inv = np.array(nc['thk'][:])

    speed_obs = np.hypot(uobs, vobs)
    speed = np.hypot(u, v)
    valid = icemask & np.isfinite(speed_obs)
    stats = {
        'glacier': glacier,
        'thk_mean': thk[icemask].mean(),
        'obs_mean': speed_obs[valid].mean(),
        'model_mean': speed[valid].mean(),
        'ratio': speed[valid].mean() / speed_obs[valid].mean(),
        'rms_vector': np.sqrt(np.mean((u - uobs)[valid] ** 2
                                      + (v - vobs)[valid] ** 2)),
        'coverage': valid.sum() / icemask.sum(),
    }
    if speed_inv is not None:
        stats['inv_thk_mean'] = thk_inv[icemask].mean()
        stats['inv_rms_speed'] = np.sqrt(np.mean(
            (speed_inv - speed_obs)[valid] ** 2))
    return stats, dict(x=x, y=y, thk=thk, speed=speed, speed_obs=speed_obs,
                       icemask=icemask)


def plot(glacier, fields, stats, path):
    import matplotlib.pyplot as plt
    ice = fields['icemask']
    extent = [fields['x'][0], fields['x'][-1], fields['y'][0], fields['y'][-1]]

    def show(ax, data, title, **kw):
        im = ax.imshow(np.where(ice, data, np.nan), origin='lower',
                       extent=extent, **kw)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    vmax = np.nanpercentile(fields['speed_obs'][ice], 98)
    diff = fields['speed'] - fields['speed_obs']
    dmax = np.nanpercentile(np.abs(diff[ice]), 95)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    show(axes[0], fields['thk'], 'provided thickness (m)', cmap='Blues')
    show(axes[1], fields['speed_obs'], 'observed speed (m/yr)', cmap='magma',
         vmin=0, vmax=vmax)
    show(axes[2], fields['speed'], 'IGM speed, provided THK (m/yr)',
         cmap='magma', vmin=0, vmax=vmax)
    show(axes[3], diff, 'IGM - observed (m/yr)', cmap='RdBu_r', vmin=-dmax,
         vmax=dmax)
    fig.suptitle(f"{glacier}: mean speed {stats['model_mean']:.1f} vs "
                 f"{stats['obs_mean']:.1f} m/yr observed "
                 f"(ratio {stats['ratio']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--exp', default='EXP01')
    parser.add_argument('--glaciers', nargs='+', default=GLACIERS)
    parser.add_argument('--config', default='experiments/continuix/config.yml')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.run:
        for glacier in args.glaciers:
            run(glacier, cfg, args.exp)
    if args.plot:
        rows = []
        for glacier in args.glaciers:
            stats, fields = compare(glacier, args.exp)
            plot(glacier, fields, stats,
                 os.path.join(OUT_DIR, args.exp, f'thk_forward_{glacier}.png'))
            rows.append(stats)
        keys = ['glacier', 'thk_mean', 'obs_mean', 'model_mean', 'ratio',
                'rms_vector', 'coverage', 'inv_thk_mean', 'inv_rms_speed']
        lines = ['\t'.join(keys)] + [
            '\t'.join(r['glacier'] if k == 'glacier' else
                      f"{r[k]:.2f}" if k in r else '' for k in keys)
            for r in rows]
        with open(os.path.join(OUT_DIR, args.exp, 'thk_forward.tsv'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
