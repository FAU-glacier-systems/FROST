#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Compare the forward runs of run_emulator_test.py (frozen emulator,
retrained emulator, solver) with each other and the observed speeds.

Run from the repository root after the jobs:
    python experiments/emulator_test/compare.py
Writes experiments/emulator_test/tables/speeds.tsv and
plots/emulator_test.png (--dark: plots/emulator_test_dark.png).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from frost.visualization.plot_style import DARK_THEME, fg

from run_emulator_test import SOURCE, TARGET, VARIANTS

plt.rcParams.update({**{f'axes.spines.{side}': False for side in ['top', 'right', 'bottom', 'left']},
                     'axes.grid': True, 'axes.axisbelow': True, 'grid.linestyle': '--',
                     'grid.linewidth': 0.5, 'grid.color': '0.75', 'legend.frameon': False})
COLOURS = {'frozen': 'C3', 'retrain': 'C1', 'retrain_1': 'C2', 'retrain_1x20': 'C4',
           'retrain_2x20': 'C5', 'retrain_4x20': 'C6', 'adaptive_1e-3': 'C8',
           'adaptive_1e-4': 'C9', 'solver': 'C0'}


def regions(rgi_id):
    """Ice with observed speeds on the inversion geometry, whole and lower third."""
    with Dataset(os.path.join(SOURCE, rgi_id, 'Preprocess', 'outputs', 'output.nc')) as ds:
        obs = np.array(ds['velsurfobs_mag'][:], dtype=float)
        thk = np.array(ds['thk_inversion'][:], dtype=float)
        usurf = np.array(ds['usurf'][:], dtype=float)
        year = float(getattr(ds, 'inversion_year', 2018))
    ice = (thk > 1) & np.isfinite(obs) & (obs > 0)
    lower = ice & (usurf <= np.percentile(usurf[ice], 100 / 3))
    return {'glacier': ice, 'lower': lower}, obs, year


def run(rgi_id, variant):
    path = os.path.join(TARGET, rgi_id, variant, 'outputs', 'output.nc')
    if not os.path.exists(path):
        return None
    with Dataset(path) as ds:
        run = {name: np.array(ds[name][:], dtype=float) for name in ['time', 'velsurf_mag', 'thk']}
        run['dx'] = abs(float(ds['x'][1] - ds['x'][0]))
    runtime = os.path.join(TARGET, rgi_id, variant, 'runtime.txt')
    run['runtime'] = open(runtime).read().split()[0] if os.path.exists(runtime) else '-'
    return run


def mean_speed(run, mask):
    moving = run['thk'] > 1
    return np.array([np.nanmean(v[mask & m]) if (mask & m).any() else np.nan
                     for v, m in zip(run['velsurf_mag'], moving)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--dark', action='store_true')
    args = parser.parse_args()
    if args.dark:
        plt.rcParams.update(DARK_THEME)
        plt.rcParams['grid.color'] = '0.35'
    glaciers = [line.split()[:2] for line in open(os.path.join(HERE, 'glaciers.txt')) if line.strip()]

    rows = []
    fig, axes = plt.subplots(len(glaciers), 2, figsize=(8.27, 2.6 * len(glaciers)), squeeze=False)
    for (rgi_id, name), row in zip(glaciers, axes):
        masks, obs, year = regions(rgi_id)
        for ax, key in zip(row, ['glacier', 'lower']):
            ax.hlines(np.nanmean(obs[masks[key]]), 2014, 2022, color=fg(), lw=4, alpha=0.5,
                      label='observed 2014-2022')
            ax.set_title(f'{name.replace("_", " ")}: {"whole glacier" if key == "glacier" else "lower third"}',
                         fontsize=9)
            ax.set_ylabel('Mean speed (m/yr)', fontsize=8)
            ax.tick_params(labelsize=7)
        for variant in VARIANTS:
            result = run(rgi_id, variant)
            if result is None:
                continue
            t = result['time']
            series = {key: mean_speed(result, mask) for key, mask in masks.items()}
            k = int(np.argmin(np.abs(t - year)))
            volume = result['thk'].sum((1, 2)) * result['dx'] ** 2 * 1e-9
            jump = np.abs(np.diff(result['thk'], axis=0)).max()
            rows.append({'glacier': name, 'variant': variant, 'runtime_s': result['runtime'],
                         'obs': np.nanmean(obs[masks['glacier']]),
                         'speed_2000': series['glacier'][0], 'speed_2001': series['glacier'][1],
                         f'speed_{year:.0f}': series['glacier'][k], 'speed_2020': series['glacier'][-1],
                         'lower_obs': np.nanmean(obs[masks['lower']]),
                         'lower_2000': series['lower'][0], f'lower_{year:.0f}': series['lower'][k],
                         'max_speed_2000': np.nanmax(np.where(result['thk'][0] > 1, result['velsurf_mag'][0], 0)),
                         'max_speed_any': np.nanmax(np.where(result['thk'] > 1, result['velsurf_mag'], 0)),
                         'max_thk_jump_per_yr': jump,
                         'volume_change_pct': 100 * (volume[-1] / volume[0] - 1)})
            for ax, key in zip(row, ['glacier', 'lower']):
                ax.plot(t, series[key], color=COLOURS[variant], label=variant)
        for ax in row:
            ax.set_yscale('log')
        row[0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel('Year', fontsize=8)
    fig.suptitle('Forward speed of one posterior member: frozen emulator, retrained emulator, solver',
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    path = os.path.join(HERE, 'plots', f'emulator_test{"_dark" if args.dark else ""}.png')
    fig.savefig(path, dpi=150)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(HERE, 'tables', 'speeds.tsv'), sep='\t', index=False, float_format='%.3g')
    print(table.to_string(index=False, float_format='%.3g'))
    print(f'Figure: {path}')


if __name__ == '__main__':
    main()
