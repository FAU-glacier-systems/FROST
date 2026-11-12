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
# (colour, legend label); every variant starts with 300 training iterations
# unless the label says otherwise
STYLE = {
    'solver': ('C0', 'Solver (reference, no emulator)'),
    'frozen': ('C3', 'Frozen emulator'),
    'retrain': ('C1', 'Retrain 5 it. every 5 steps (v2, fast glaciers)'),
    'retrain_1': ('C9', 'Retrain 5 it. every step'),
    'retrain_1x20': ('C4', 'Retrain 20 it. every step (v2, slow glaciers)'),
    'retrain_2x20': ('C5', 'Retrain 20 it. every 2 steps'),
    'retrain_4x20': ('C6', 'Retrain 20 it. every 4 steps'),
    'adaptive_1e-3': ('C8', 'Adaptive, tol 1e-3'),
    'adaptive_1e-4': ('C9', 'Adaptive, tol 1e-4'),
    'retrain_t1x20': ('C2', 'Retrain 20 it. every 1 yr (new)'),
    'retrain_t1x10': ('C3', 'Retrain 10 it. every 1 yr'),
}
# file names: speeds.tsv / emulator_test.png for the first source, a suffix
# for later targets (emulator_test_interval -> speeds_interval.tsv)
TAG = os.path.basename(TARGET).replace('emulator_test', '')


def regions(rgi_id):
    """Ice with observed speeds on the inversion geometry, whole and lower third."""
    with Dataset(os.path.join(SOURCE, rgi_id, 'Preprocess', 'outputs', 'output.nc')) as ds:
        obs = np.array(ds['velsurfobs_mag'][:], dtype=float)
        # thk_inversion since restore_2000 (2026-10); older inversions: thk
        thk = np.array(ds['thk_inversion' if 'thk_inversion' in ds.variables else 'thk'][:],
                       dtype=float)
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
    # only glaciers with at least one run in TARGET
    glaciers = [(rgi_id, name) for rgi_id, name in glaciers
                if any(run(rgi_id, variant) is not None for variant in VARIANTS)]

    rows = []
    handles = {}
    fig, axes = plt.subplots(len(glaciers), 2, figsize=(8.27, 2.5 * len(glaciers) + 1.0),
                             squeeze=False)
    for (rgi_id, name), row in zip(glaciers, axes):
        masks, obs, year = regions(rgi_id)
        for ax, key in zip(row, ['glacier', 'lower']):
            handles['observed'] = ax.hlines(np.nanmean(obs[masks[key]]), 2014, 2022, color=fg(),
                                            lw=4, alpha=0.35, label='Observed 2014-2022 (ITS_LIVE)')
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
            colour, label = STYLE[variant]
            for ax, key in zip(row, ['glacier', 'lower']):
                handles[variant], = ax.plot(t, series[key], color=colour, label=label,
                                            lw=2.2 if variant == 'solver' else 1.3)
        for ax in row:
            # log scale only for start-up shocks (frozen emulator: 10x and more)
            low, high = ax.get_ylim()
            if low > 0 and high / low > 10:
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for ax in axes[-1]:
        ax.set_xlabel('Year', fontsize=8)
    order = ['observed'] + [v for v in VARIANTS if v in handles]
    height = fig.get_figheight()
    fig.suptitle('Forward speed 2000-2020 of one posterior member: emulator retraining vs solver',
                 fontsize=10, y=1 - 0.1 / height)
    fig.legend([handles[k] for k in order], [handles[k].get_label() for k in order],
               loc='upper center', ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1 - 0.35 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    path = os.path.join(HERE, 'plots', f'emulator_test{TAG}{"_dark" if args.dark else ""}.png')
    fig.savefig(path, dpi=150)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(HERE, 'tables', f'speeds{TAG}.tsv'), sep='\t', index=False, float_format='%.3g')
    print(table.to_string(index=False, float_format='%.3g'))
    print(f'Figure: {path}')


if __name__ == '__main__':
    main()
