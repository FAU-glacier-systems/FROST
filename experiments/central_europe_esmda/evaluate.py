#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
The new ES-MDA calibration (data/results/central_europe_esmda) and the
submitted one (central_europe_submit) against the GLAMOS/WGMS reference
ELA and gradients of the glaciers in glaciers.txt: per parameter n, r, bias,
MAE and the share of glaciers whose reference lies inside the posterior
+-1 and +-2 sigma (expected 68 and 95 %). Modelled gradients are in m ice
per km, the reference in m w.e. per km; the rows "w.e." convert the
modelled gradients with 0.91 (ice density 910 / water 1000).

Melt-out check per glacier: elevation bands whose ice is gone in 2020 in
every member and whose modelled dh/dt has no ensemble spread carry no
information on the SMB parameters (as Aletsch's tongue).

Run from the repository root after the jobs:
    python experiments/central_europe_esmda/evaluate.py
Another experiment with the same glaciers (its results as "new", tables and
plots in its own folder):
    python experiments/central_europe_esmda/evaluate.py \
        --results data/results/central_europe_esmda_velyear \
        --out experiments/central_europe_esmda_velyear
Writes tables/evaluation.tsv, tables/glaciers.tsv and plots/evaluation.png,
and tables/aggregated_results.csv for experiments/central_europe/evaluation.py
(the submitted table's metadata, SLA and references with the new ELA and
gradients, glaciers with new results only):
    python experiments/central_europe/evaluation.py \
        --Modeled_results experiments/central_europe_esmda/tables/aggregated_results.csv \
        --output_dir experiments/central_europe_esmda/plots/evaluation_central_europe
"""

import glob
import json
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
from frost.calibration.observation_provider import ObservationProvider

REFERENCE = os.path.join('experiments', 'central_europe', 'tables', 'aggregated_results.csv')
RUNS = {'new': os.path.join('data', 'results', 'central_europe_esmda'),
        'submitted': os.path.join('data', 'results', 'central_europe_submit')}
PARAMETERS = {'ela': 'ela_wgmsglamos', 'abl': 'gradabl_wgmsglamos',
              'acc': 'gradacc_wgmsglamos'}
LABELS = {'ela': 'ELA (m)', 'abl': 'Ablation gradient (per km)',
          'acc': 'Accumulation gradient (per km)'}
ICE_TO_WE = 0.91


def melt_out(rgi_dir, elevation_step=50):
    """Number of bands that melt out in every member without spread, and
    the number of bands."""
    members = sorted(glob.glob(os.path.join(rgi_dir, 'Ensemble', 'Member_*',
                                            'outputs', 'output.nc')))
    if not members:
        return np.nan, np.nan
    provider = ObservationProvider(rgi_dir, None, elevation_step, None, False)
    dhdt, gone = [], []
    for path in members:
        with Dataset(path) as ds:
            usurf = np.array(ds['usurf'][:])
            thk = np.array(ds['thk'][-1])
        dhdt.append(provider.band_mean((usurf[-1] - usurf[0]) / provider.period))
        gone.append(provider.band_mean((thk < 1).astype(float)))
    dhdt, gone = np.array(dhdt), np.array(gone)
    dead = (gone.min(axis=0) > 0.99) & (dhdt.std(axis=0) < 0.01)
    return int(dead.sum()), provider.num_bins


def glacier_rows(glaciers, reference):
    rows = []
    for rgi_id, name in glaciers:
        ref = reference.loc[rgi_id]
        for run, results in RUNS.items():
            rgi_dir = os.path.join(results, rgi_id)
            path = os.path.join(rgi_dir, 'calibration_results.json')
            if not os.path.exists(path):
                continue
            with open(path) as f:
                result = json.load(f)
            row = dict(rgi_id=rgi_id, glacier=name, run=run)
            for k, (p, column) in enumerate(PARAMETERS.items()):
                row[p] = result['final_mean'][k]
                row[f'{p}_std'] = result['final_std'][k]
                row[f'{p}_ref'] = ref[column]
            if run == 'new':
                row['dead_bands'], row['bands'] = melt_out(rgi_dir)
            rows.append(row)
    return pd.DataFrame(rows)


def scores(table):
    rows = []
    for run, group in table.groupby('run'):
        for p in PARAMETERS:
            for unit, factor in [('ice', 1.0), ('w.e.', ICE_TO_WE)]:
                if p == 'ela' and unit == 'w.e.':
                    continue
                g = group[np.isfinite(group[f'{p}_ref'])]
                model, std = factor * g[p], factor * g[f'{p}_std']
                diff = model - g[f'{p}_ref']
                z = np.abs(diff) / std
                rows.append(dict(run=run, parameter=p, unit=unit, n=len(g),
                                 r=np.corrcoef(model, g[f'{p}_ref'])[0, 1],
                                 bias=diff.mean(), mae=np.abs(diff).mean(),
                                 inside_1sigma_pct=100 * np.mean(z <= 1),
                                 inside_2sigma_pct=100 * np.mean(z <= 2),
                                 median_std=std.median()))
    return pd.DataFrame(rows)


def plot(table, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    colors = {'submitted': 'tab:gray', 'new': 'tab:red'}
    for ax, p in zip(axes, PARAMETERS):
        for run, group in table.groupby('run'):
            g = group[np.isfinite(group[f'{p}_ref'])]
            r = np.corrcoef(g[p], g[f'{p}_ref'])[0, 1]
            ax.errorbar(g[f'{p}_ref'], g[p], yerr=g[f'{p}_std'], fmt='o', ms=4,
                        capsize=2, alpha=0.8, color=colors[run],
                        label=f'{run} (r {r:.2f}, n {len(g)})')
        values = table[[p, f'{p}_ref']].to_numpy(dtype=float)
        lim = [np.nanmin(values), np.nanmax(values)]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_xlabel(f'GLAMOS / WGMS {LABELS[p]}')
        ax.set_ylabel(f'FROST {LABELS[p]}')
        ax.legend(fontsize=8)
    fig.suptitle('Posterior mean +- 1 std against the reference; modelled gradients '
                 'in m ice, reference in m w.e.', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def panel_columns(rgi_dir, elevation_step=50):
    """The columns of the elevation change, thickness and velocity panels of
    experiments/central_europe/evaluation.py for one glacier of the new run
    (the submitted table has the submitted run's values):
    - dhdt_mean, dhdt_mean_sigma: Hugonnet dh/dt averaged over the observed
      pixels and its 1-sigma from the band covariance (without model error)
    - dhdt_ensemble_mean, dhdt_ensemble_std: the posterior members' dh/dt
      2000-2020 over the same pixels, mean and spread over the members
    - Mean/Std_thk_obs, Mean/Std_thk_model_at_obs: GlaThiDa moved to 2000
      (as thickness_validation.py) and the thickness of 2000 at those cells
    - Mean/Std_velsurfobs_mag, Mean/Std_velsurf_mag: observed and inverted
      surface speed on the ice (the inversion year: 2000, or the velocity
      year with inversion_year)"""
    from frost.preprocess.thickness_validation import read_field, thkobs_at_year
    row = {}
    provider = ObservationProvider(rgi_dir, None, elevation_step, None, False)
    observed = provider.dhdt[provider.valid]
    row['dhdt_mean'] = observed.mean()
    row['dhdt_mean_sigma'] = provider.glacier_mean_sigma()
    members = []
    for path in sorted(glob.glob(os.path.join(rgi_dir, 'Ensemble', 'Member_*',
                                              'outputs', 'output.nc'))):
        with Dataset(path) as ds:
            usurf = np.array(ds['usurf'][:])
        members.append(((usurf[-1] - usurf[0]) / provider.period)[provider.valid].mean())
    row['dhdt_ensemble_mean'] = np.mean(members) if members else np.nan
    row['dhdt_ensemble_std'] = np.std(members) if members else np.nan

    preprocess = os.path.join(rgi_dir, 'Preprocess')
    with Dataset(os.path.join(preprocess, 'outputs', 'output.nc')) as ds:
        ice = read_field(ds, 'icemask') > 0.5
        thk = read_field(ds, 'thk')
        speed = read_field(ds, 'velsurf_mag')
        speed_obs = read_field(ds, 'velsurfobs_mag')
        x, y = np.array(ds['x'][:]), np.array(ds['y'][:])
    moving = ice & np.isfinite(speed_obs) & np.isfinite(speed)
    for name, values in [('velsurf_mag', speed), ('velsurfobs_mag', speed_obs)]:
        row[f'Mean_{name}'] = values[moving].mean()
        row[f'Std_{name}'] = values[moving].std()
    glathida = os.path.join(preprocess, 'data', os.path.basename(rgi_dir),
                            'glathida_data.csv')
    for column in ['Mean_thk_obs', 'Std_thk_obs', 'Mean_thk_model_at_obs',
                   'Std_thk_model_at_obs']:
        row[column] = np.nan
    if os.path.exists(glathida):
        with Dataset(os.path.join(preprocess, 'data', 'input.nc')) as ds:
            dhdt = np.nan_to_num(read_field(ds, 'dhdt'))
        thkobs, _ = thkobs_at_year(glathida, x, y, dhdt, 2000)
        cells = ice & np.isfinite(thkobs) & (thkobs > 0)
        if cells.any():
            row.update(Mean_thk_obs=thkobs[cells].mean(), Std_thk_obs=thkobs[cells].std(),
                       Mean_thk_model_at_obs=thk[cells].mean(),
                       Std_thk_model_at_obs=thk[cells].std())
    return row


def aggregated_results(reference, new, path):
    """The submitted aggregated_results.csv with the new posterior mean and
    std of ELA and gradients and the new values of the elevation change,
    thickness and velocity panels (panel_columns), restricted to the
    glaciers in new."""
    table = reference.loc[new.rgi_id].reset_index()
    columns = {'ela': 'ela', 'abl': 'gradabl', 'acc': 'gradacc'}
    values = new.set_index('rgi_id')
    for p, column in columns.items():
        table[column] = values.loc[table.rgi_id, p].to_numpy()
        table[f'{column}_std'] = values.loc[table.rgi_id, f'{p}_std'].to_numpy()
    panels = pd.DataFrame([panel_columns(os.path.join(RUNS['new'], rgi_id))
                           for rgi_id in table.rgi_id])
    for column in panels:
        table[column] = panels[column].to_numpy()
    table['ela'] = table['ela'].round().astype('Int64')
    table.to_csv(path, index=False, float_format='%.4f')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=RUNS['new'],
                        help='results of the "new" run')
    parser.add_argument('--out', default=HERE,
                        help='folder for tables/ and plots/')
    args = parser.parse_args()
    RUNS['new'] = args.results
    out = args.out

    reference = pd.read_csv(REFERENCE).set_index('rgi_id')
    with open(os.path.join(HERE, 'glaciers.txt')) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    table = glacier_rows(glaciers, reference)
    # compare on the glaciers both runs have
    both = table.groupby('rgi_id').run.nunique() == len(RUNS)
    table = table[table.rgi_id.isin(both[both].index)]
    summary = scores(table)
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    table.to_csv(os.path.join(out, 'tables', 'glaciers.tsv'), sep='\t',
                 index=False, float_format='%.4g')
    summary.to_csv(os.path.join(out, 'tables', 'evaluation.tsv'), sep='\t',
                   index=False, float_format='%.3g')
    plot(table, os.path.join(out, 'plots', 'evaluation.png'))
    aggregated_results(reference, table[table.run == 'new'],
                       os.path.join(out, 'tables', 'aggregated_results.csv'))
    print(f"{table.rgi_id.nunique()} glaciers in both runs")
    print(summary.to_string(index=False, float_format='%.2f'))
    new = table[table.run == 'new']
    print('\nBands melting out in every member (dead / all):')
    print(new[['glacier', 'dead_bands', 'bands']].to_string(index=False))


if __name__ == '__main__':
    main()
