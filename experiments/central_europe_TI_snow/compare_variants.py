#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
TI calibration without ice flow of the 16 slow glaciers in three variants:
- one melt factor: central_europe_TI (OGGM's monthly TI model)
- snow 0.5: central_europe_TI_snow (snow and firn melt with 0.5 x melt_f)
- snow 0.5 + lapse rate: central_europe_TI_snow_lapse (also the temperature
  lapse rate calibrated, prior -6.5 +- 1 K/km)

Per glacier and variant:
- dh/dt 2000-2020 per band against Hugonnet (as central_europe_TI_slowflow/
  compare_flow.py): band RMSE, chi2 (data error), error of the lowest and
  highest third of the area
- SMB profile against the GLAMOS / WGMS mean 2000-2019
  (central_europe_esmda/evaluate.py member_smb): RMSE, bias, share of the
  bins within +-2 std of the members, and the gradient below and above the
  reference ELA (m w.e. per 100 m; the shape the single melt factor misses)
- GLAMOS / WGMS band-year scores of evaluate_bands.py (tables/band_scores.tsv
  of each experiment; one melt factor: experiments/central_europe_TI)
- posterior parameters

Run from the repository root after evaluate_bands.py of every run (its
tables/band_scores.tsv in experiments/<run>):
    python experiments/central_europe_TI_snow/compare_variants.py
Other runs and glaciers, e.g. the glaciers with ice flow:
    python experiments/central_europe_TI_snow/compare_variants.py \
        --glaciers experiments/central_europe_TI_flow_snow_lapse/glaciers.txt \
        --runs "one melt factor=central_europe_TI_combined" \
               "snow 0.5 + lapse rate=central_europe_TI_flow_snow_lapse" \
        --out experiments/central_europe_TI_flow_snow_lapse
Writes tables/compare_variants.tsv and plots/compare_variants_profiles_dark.{pdf,png}
into --out (default experiments/central_europe_TI_snow).
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path.insert(0, os.getcwd())
for folder in ('central_europe_esmda', 'central_europe_TI_slowflow', 'validation'):
    sys.path.insert(0, os.path.join(EXPERIMENTS, folder))
import sanity_report as sr  # noqa: E402
from compare_flow import dhdt_scores  # noqa: E402
from evaluate import member_smb  # noqa: E402
from reference_profiles import mass_balance_profiles  # noqa: E402
from frost.visualization.plot_style import COLORS, DARK_THEME, use_font  # noqa: E402

RUNS = ['one melt factor=central_europe_TI', 'snow 0.5=central_europe_TI_snow',
        'snow 0.5 + lapse rate=central_europe_TI_snow_lapse']
PALETTE = [COLORS['grey'], COLORS['yellow'], COLORS['red'], COLORS['blue']]
GLAMOS = [('band annual', 'rmse', 'band_rmse'), ('band winter', 'bias', 'winter_bias'),
          ('band summer', 'bias', 'summer_bias'), ('glacier-wide annual', 'r', 'gw_r'),
          ('glacier-wide annual', 'rmse', 'gw_rmse')]


def read_glaciers(path):
    with open(path) as f:
        return [line.split()[:2] for line in f if line.strip()]


def profile_scores(result, profile, climate_file):
    """Modelled SMB at the reference bins against the reference."""
    z = profile.elevation.to_numpy(float)
    members = member_smb(result, z, climate_file)
    model, std = members.mean(0), members.std(0)
    obs = profile.mb.to_numpy(float)
    error = model - obs
    row = {'smb_rmse': float(np.sqrt(np.mean(error ** 2))), 'smb_bias': float(error.mean()),
           'smb_inside_pct': float(100 * np.mean(np.abs(error) <= 2 * std))}
    # gradients below and above the reference ELA (zero crossing of the profile)
    ela = np.interp(0, obs, z) if obs.min() < 0 < obs.max() else np.nan
    for side, mask in [('below', z < ela), ('above', z > ela)]:
        for name, values in [('obs', obs), ('model', model)]:
            row[f'grad_{side}_{name}'] = (float(np.polyfit(z[mask], values[mask], 1)[0] * 100)
                                          if mask.sum() >= 2 else np.nan)
    return row, (z, obs, model, std)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--glaciers', default=os.path.join(HERE, 'glaciers.txt'))
    parser.add_argument('--runs', nargs='+', default=RUNS, help='label=experiment, in order')
    parser.add_argument('--out', default=HERE)
    args = parser.parse_args()
    runs = dict(run.split('=', 1) for run in args.runs)
    color = dict(zip(runs, PALETTE))
    glaciers = read_glaciers(args.glaciers)
    plt.rcParams.update(DARK_THEME)
    use_font()
    references = mass_balance_profiles([rgi_id for rgi_id, _ in glaciers])
    scores = {run: pd.read_csv(os.path.join(EXPERIMENTS, folder, 'tables', 'band_scores.tsv'),
                               sep='\t') for run, folder in runs.items()}
    rows, curves = [], {}
    for rgi_id, name in glaciers:
        for run, folder in runs.items():
            rgi_dir = os.path.join('data', 'results', folder, rgi_id)
            with open(os.path.join(rgi_dir, 'calibration_results.json')) as f:
                result = json.load(f)
            _, settings = sr.experiment_settings(rgi_dir)
            g = sr.load_glacier(rgi_dir, settings)
            row = {'rgi_id': rgi_id, 'glacier': name, 'run': run}
            row.update(dhdt_scores(g))
            names = result.get('parameters') or list(result['initial_smb'])
            row.update(dict(zip(names, result['final_mean'])))
            if rgi_id in references:
                values, curves[rgi_id, run] = profile_scores(
                    result, references[rgi_id], os.path.join(rgi_dir, 'climate_historical.nc'))
                row.update(values)
            s = scores[run]
            s = s[(s.rgi_id == rgi_id) & (s.variant == 'TI')]
            for quantity, column, key in GLAMOS:
                value = s.loc[s.quantity == quantity, column]
                row[key] = float(value.iloc[0]) if len(value) else np.nan
            rows.append(row)
        print(f'{name} done', flush=True)

    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(args.out, 'tables', 'compare_variants.tsv'), sep='\t', index=False,
                 float_format='%.3g')
    columns = ['dhdt_band_rmse', 'dhdt_chi2', 'dhdt_err_low', 'dhdt_err_high', 'smb_rmse',
               'smb_bias', 'smb_inside_pct', 'grad_below_model', 'grad_above_model',
               'band_rmse', 'winter_bias', 'summer_bias', 'gw_r', 'gw_rmse']
    columns += [c for c in ('speed_model', 'speed_obs') if c in table]
    mean = table.groupby('run', sort=False)[columns].mean()
    mean.loc['GLAMOS / WGMS', ['grad_below_model', 'grad_above_model']] = \
        table.groupby('run', sort=False)[['grad_below_obs', 'grad_above_obs']].mean().iloc[0].values
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(mean.round(2).T.to_string())
        params = [c for c in ('melt_f', 'prcp_fac', 'temp_bias', 'temp_gradient') if c in table]
        print(table.pivot(index='glacier', columns='run', values=params).round(2).to_string())

    shown = [(rgi_id, name) for rgi_id, name in glaciers if (rgi_id, list(runs)[0]) in curves]
    ncols = 4
    nrows = -(-len(shown) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5.2 * nrows), constrained_layout=True)
    for ax, (rgi_id, name) in zip(axes.flat, shown):
        for run in runs:
            z, obs, model, std = curves[rgi_id, run]
            ax.fill_betweenx(z, model - 2 * std, model + 2 * std, color=color[run], alpha=0.15, lw=0)
            ax.plot(model, z, color=color[run], lw=2.2, label=run)
        ax.plot(obs, z, 'o', color=COLORS['fg'], ms=5, label=references[rgi_id].source.iloc[0])
        ax.axvline(0, color=COLORS['muted'], lw=0.6)
        ax.set_title(name.replace('_', ' '), loc='left', fontsize=13)
        ax.set_xlabel('SMB 2000-2019 (m w.e./yr)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Elevation (m)')
    for ax in axes.flat[len(shown):]:
        ax.axis('off')
    axes.flat[0].legend(loc='lower right', fontsize=9)
    os.makedirs(os.path.join(args.out, 'plots'), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(args.out, 'plots', f'compare_variants_profiles_dark.{ext}'), dpi=110)


if __name__ == '__main__':
    main()
