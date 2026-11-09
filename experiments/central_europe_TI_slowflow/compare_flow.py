#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
The 10 slow glaciers (ITS_LIVE p99 3-10 m/yr) with ice flow
(central_europe_TI_slowflow) against the same glaciers without ice flow
(central_europe_TI), both TI with ES-MDA, 4 iterations.

Per glacier and run:
- dh/dt 2000-2020 per 50 m band against Hugonnet: glacier-wide (area
  weighted), band RMSE (area weighted), chi2 with the data covariance, mean
  band error in the lowest and highest third of the glacier area, bands
  melted out in every member; posterior median of the members
- volume change 2000-2020, posterior parameters (melt_f, prcp_fac, temp_bias)
- with ice flow: band speed against ITS_LIVE (median over the bands) and
  the flux divergence of the lowest band
- GLAMOS / WGMS: the scores of evaluate_bands.py (tables/band_scores.tsv
  of both experiments, variant TI)

Run from the repository root after evaluate_bands.py of this experiment:
    python experiments/central_europe_TI_slowflow/compare_flow.py
Writes tables/compare_flow.tsv and plots/compare_flow_dhdt_dark.{pdf,png}.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'central_europe_esmda'))
import sanity_report as sr  # noqa: E402
from frost.visualization.plot_style import COLORS, DARK_THEME  # noqa: E402

RUNS = {'no flow': 'central_europe_TI', 'ice flow': 'central_europe_TI_slowflow'}
COLOR = {'no flow': COLORS['grey'], 'ice flow': COLORS['red']}
PARAMS = ['melt_f', 'prcp_fac', 'temp_bias']
GLAMOS = [('band annual', 'rmse', 'smb_band_rmse'), ('band annual', 'bias', 'smb_band_bias'),
          ('glacier-wide annual', 'rmse', 'smb_gw_rmse'), ('glacier-wide annual', 'r', 'smb_gw_r'),
          ('mean profile', 'rmse', 'smb_profile_rmse')]


def glaciers():
    with open(os.path.join(HERE, 'glaciers.txt')) as f:
        return [line.split() for line in f if line.strip()]


def dhdt_scores(g):
    provider = g['provider']
    n = provider.num_bins
    obs = provider.observation[:n]
    w = g['band_weights']
    model = np.median(g['band_dhdt'], 0)
    error = model - obs
    # lowest and highest third of the area, bands sorted by elevation
    order = np.argsort(g['band_centre'])
    cum = np.cumsum(w[order])
    low, high = order[cum <= 1 / 3 + 1e-9], order[cum >= 2 / 3 - 1e-9]
    low = low if low.size else order[:1]
    checks = sr.dhdt_checks(g)
    row = {'dhdt_obs': float(w @ obs), 'dhdt_model': float(w @ model),
           'dhdt_band_rmse': float(np.sqrt(w @ error ** 2)),
           'dhdt_chi2': checks['dhdt_chi2_data'],
           'dhdt_err_low': float(w[low] @ error[low] / w[low].sum()),
           'dhdt_err_high': float(w[high] @ error[high] / w[high].sum()),
           'dead_bands': checks['dead_bands'], 'bands': n,
           'volume_change_pct': checks['volume_change_pct']}
    if not g['no_flow']:
        k = g['inv_index']
        speed = np.nanmedian(g['band_speed'][:, k], 0)
        observed = sr.observed_band_speed(g)
        ok = np.isfinite(speed) & (observed > 0)
        row['speed_model'] = float(np.median(speed[ok]))
        row['speed_obs'] = float(np.median(observed[ok]))
        row['div_lowest_band'] = float(g['band_div'].mean(0)[order[0]])
    return row


def main():
    plt.rcParams.update(DARK_THEME)
    scores = {run: pd.read_csv(os.path.join('experiments', folder, 'tables', 'band_scores.tsv'),
                               sep='\t') for run, folder in RUNS.items()}
    rows, profiles = [], {}
    for rgi_id, name in glaciers():
        for run, folder in RUNS.items():
            rgi_dir = os.path.join('data', 'results', folder, rgi_id)
            _, settings = sr.experiment_settings(rgi_dir)
            g = sr.load_glacier(rgi_dir, settings)
            row = {'rgi_id': rgi_id, 'glacier': name, 'run': run}
            row.update(dhdt_scores(g))
            with open(os.path.join(rgi_dir, 'calibration_results.json')) as f:
                row.update(dict(zip(PARAMS, json.load(f)['final_mean'])))
            s = scores[run]
            s = s[(s.rgi_id == rgi_id) & (s.variant == 'TI')]
            for quantity, column, key in GLAMOS:
                values = s.loc[s.quantity == quantity, column]
                row[key] = float(values.iloc[0]) if len(values) else np.nan
            rows.append(row)
            profiles[rgi_id, run] = (g['band_centre'], g['provider'].observation[:g['provider'].num_bins],
                                     np.median(g['band_dhdt'], 0), -g['band_div'].mean(0),
                                     g['no_flow'])
            print(f'{name}, {run}: band RMSE {row["dhdt_band_rmse"]:.2f} m/yr', flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(HERE, 'tables', 'compare_flow.tsv'), sep='\t', index=False,
                 float_format='%.3g')
    columns = ['dhdt_obs', 'dhdt_model', 'dhdt_band_rmse', 'dhdt_err_low', 'dhdt_err_high',
               'dead_bands', 'volume_change_pct', 'smb_band_rmse', 'smb_gw_rmse', 'smb_profile_rmse']
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(table.pivot(index='glacier', columns='run', values=columns).round(2).to_string())

    fig, axes = plt.subplots(2, 5, figsize=(20, 10), constrained_layout=True)
    for ax, (rgi_id, name) in zip(axes.flat, glaciers()):
        for run in RUNS:
            centre, obs, model, div, no_flow = profiles[rgi_id, run]
            ax.plot(model, centre, color=COLOR[run], lw=2, label=f'model, {run}')
            if not no_flow:
                ax.plot(div, centre, color=COLORS['blue'], lw=1.2, label='minus flux divergence')
        ax.plot(obs, centre, 'o', color=COLORS['fg'], ms=3, label='observed (Hugonnet)')
        ax.axvline(0, color=COLORS['muted'], lw=0.6)
        ax.set_title(name.replace('_', ' '), fontsize=12, loc='left')
        ax.set_xlabel('dh/dt 2000-2020 (m/yr)')
    axes[0, 0].set_ylabel('Elevation (m)')
    axes[1, 0].set_ylabel('Elevation (m)')
    axes[1, 0].legend(loc='lower left', fontsize=9)
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(HERE, 'plots', f'compare_flow_dhdt_dark.{ext}'), dpi=130)


if __name__ == '__main__':
    main()
