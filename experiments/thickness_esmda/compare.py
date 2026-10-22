#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
ES-MDA ELA calibration with fixed IceBoost and fixed consensus geometry on
Rhone and Aletsch, and IceBoost with tau_ref smoothness 1e10 and 1e12 (1e11
otherwise), and IceBoost with ITS_LIVE instead of Millan velocities, against GLAMOS, the submitted run (central_europe_submit)
and the Rhone run with Millan geometry (central_europe_esmda, if still
there). Modelled gradients are in m ice per km, GLAMOS in m w.e. per km.

Run from the repository root after the jobs (thickness_esmda_sbatch.sh):
    python experiments/thickness_esmda/compare.py
Writes tables/compare.tsv and plots/compare.png (posterior ensembles per
glacier and parameter, GLAMOS as a line; inverted thickness and its bias
against GlaThiDa).
"""

import json
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join('experiments', 'central_europe', 'tables', 'aggregated_results.csv')
PARAMETERS = ['ela', 'abl', 'acc']
LABELS = {'ela': 'ELA (m)', 'abl': 'Ablation gradient (m a$^{-1}$ km$^{-1}$)',
          'acc': 'Accumulation gradient (m a$^{-1}$ km$^{-1}$)'}
COLORS = {'iceboost_lam1e10': 'yellowgreen', 'iceboost': 'tab:green',
          'iceboost_lam1e12': 'darkgreen', 'iceboost_itslive': 'tab:purple',
          'iceboost_lam1e10_me05': 'tab:red', 'iceboost_lam1e10_me05_gw': 'maroon', 'iceboost_itslive_lam1e10_me05': 'tab:pink', 'consensus': 'tab:blue', 'millan': 'tab:orange',
          'submitted': 'tab:gray'}
SHORT = {'iceboost_lam1e10': 'IB 1e10', 'iceboost': 'IB 1e11', 'iceboost_lam1e12': 'IB 1e12',
         'iceboost_itslive': 'IB ITS_LIVE', 'iceboost_lam1e10_me05': 'IB 1e10 me0.5',
         'iceboost_lam1e10_me05_gw': 'IB 1e10 me0.5 +glacier dh/dt',
         'iceboost_itslive_lam1e10_me05': 'IB ITS_LIVE 1e10 me0.5',
         'consensus': 'cons 1e11', 'millan': 'Millan 1e11', 'submitted': 'submitted'}
RUNS = {'iceboost_lam1e10': os.path.join('data', 'results', 'thickness_esmda', 'iceboost_lam1e10'),
        'iceboost': os.path.join('data', 'results', 'thickness_esmda', 'iceboost'),
        'iceboost_lam1e12': os.path.join('data', 'results', 'thickness_esmda', 'iceboost_lam1e12'),
        'iceboost_itslive': os.path.join('data', 'results', 'thickness_esmda', 'iceboost_itslive'),
        'iceboost_lam1e10_me05': os.path.join('data', 'results', 'thickness_esmda', 'iceboost_lam1e10_me05'),
        'iceboost_lam1e10_me05_gw': os.path.join('data', 'results', 'thickness_esmda',
                                                 'iceboost_lam1e10_me05_gw'),
        'iceboost_itslive_lam1e10_me05': os.path.join('data', 'results', 'thickness_esmda',
                                                      'iceboost_itslive_lam1e10_me05'),
        'consensus': os.path.join('data', 'results', 'thickness_esmda', 'consensus'),
        'millan': os.path.join('data', 'results', 'central_europe_esmda'),
        'submitted': os.path.join('data', 'results', 'central_europe_submit')}


def load(path):
    with open(path) as f:
        return json.load(f)


def run_row(rgi_dir):
    result = load(os.path.join(rgi_dir, 'calibration_results.json'))
    row = {f'{p}': m for p, m in zip(PARAMETERS, result['final_mean'])}
    row['ensemble'] = np.array(result['final_ensemble'])
    row.update({f'{p}_std': s for p, s in zip(PARAMETERS, result['final_std'])})
    if result.get('diagnostics'):
        row['misfit'] = result['diagnostics'][-1]['misfit']
    metrics = sorted(f for f in os.listdir(os.path.join(rgi_dir, 'Monitor'))
                     if f.startswith('metrics_'))
    if metrics:
        m = load(os.path.join(rgi_dir, 'Monitor', metrics[-1]))
        row.update(dhdt_obs=m['observed_elevation_change'],
                   dhdt_model=m['modelled_elevation_change'],
                   vel_obs=m['observed_surface_velocity'],
                   vel_model=m['modelled_surface_velocity'],
                   flux_div=m.get('modelled_flux_divergence'))
    validation = os.path.join(rgi_dir, 'Preprocess', 'outputs', 'thickness_validation.json')
    if os.path.exists(validation):
        v = load(validation)
        row.update(glathida_bias=v['bias'], glathida_rms=v['rms'])
    output = os.path.join(rgi_dir, 'Preprocess', 'outputs', 'output.nc')
    if os.path.exists(output):
        with Dataset(output) as ds:
            mask = np.array(ds['icemask'][:]) > 0.5
            vel, obs = np.array(ds['velsurf_mag'][:]), np.array(ds['velsurfobs_mag'][:])
            row['thk_mean'] = float(np.array(ds['thk'][:])[mask].mean())
            observed = mask & np.isfinite(obs) & (obs > 0)
            row['vel_rms'] = float(np.sqrt(np.mean((vel - obs)[observed] ** 2)))
            if 'tau_ref' in ds.variables:  # not in the submitted run
                # patchiness: mean jump of tau_ref between neighbouring ice cells
                tau_ref = np.where(mask, np.array(ds['tau_ref'][:]), np.nan)
                row['tau_ref_median'] = float(np.nanmedian(tau_ref))
                row['tau_ref_roughness'] = float(np.nanmean(np.abs(np.diff(tau_ref, axis=1))))
    return row


def main():
    reference = pd.read_csv(REFERENCE).set_index('rgi_id')
    with open(os.path.join(HERE, 'runs.txt')) as f:
        glaciers = dict.fromkeys(tuple(line.split()[1:3]) for line in f if line.strip())
    rows = []
    for rgi_id, name in glaciers:
        ref = reference.loc[rgi_id]
        rows.append(dict(glacier=name, run='GLAMOS', ela=ref.ela_wgmsglamos,
                         abl=ref.gradabl_wgmsglamos, acc=ref.gradacc_wgmsglamos))
        for run, results in RUNS.items():
            rgi_dir = os.path.join(results, rgi_id)
            if not os.path.exists(os.path.join(rgi_dir, 'calibration_results.json')):
                continue
            row = dict(glacier=name, run=run, **run_row(rgi_dir))
            for p, column in zip(PARAMETERS, ['ela_wgmsglamos', 'gradabl_wgmsglamos',
                                              'gradacc_wgmsglamos']):
                # GLAMOS in posterior standard deviations
                row[f'{p}_z'] = (row[p] - ref[column]) / row[f'{p}_std']
            row['monitor'] = os.path.join(rgi_dir, 'Monitor', 'status.png')
            rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.drop(columns='ensemble').to_csv(os.path.join(HERE, 'tables', 'compare.tsv'),
                                          sep='\t', index=False, float_format='%.4g')
    print(table.drop(columns=['monitor', 'ensemble']).to_string(
        index=False, float_format='%.2f'))
    plot(table)


def plot(table):
    names = list(dict.fromkeys(table.glacier))
    fig, axes = plt.subplots(len(names), 4, figsize=(17, 4.2 * len(names)),
                             squeeze=False)
    rng = np.random.default_rng(0)
    for ax_row, name in zip(axes, names):
        glacier = table[table.glacier == name]
        glamos = glacier[glacier.run == 'GLAMOS'].iloc[0]
        runs = glacier[glacier.run != 'GLAMOS']
        for k, (ax, p) in enumerate(zip(ax_row, PARAMETERS)):
            for i, (_, run) in enumerate(runs.iterrows()):
                values = run.ensemble[:, k]
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=8,
                           color=COLORS[run.run], alpha=0.5)
                ax.errorbar(i + 0.3, run[p], yerr=run[f'{p}_std'], fmt='o',
                            color=COLORS[run.run], capsize=4)
            ax.axhline(glamos[p], color='k', ls='--', lw=1.2, label='GLAMOS')
            ax.set_xticks(range(len(runs)))
            ax.set_xticklabels([SHORT[r] for r in runs.run], rotation=45, ha='right')
            ax.set_xlim(-0.5, len(runs) - 0.3)
            ax.set_title(f'{name}: {LABELS[p]}', fontsize=10)
        ax_row[0].legend(loc='best', fontsize=8)
        # inverted thickness and its bias against GlaThiDa
        ax = ax_row[3]
        geometry = runs[runs.thk_mean.notna()]
        x = np.arange(len(geometry))
        ax.bar(x, geometry.thk_mean, color=[COLORS[r] for r in geometry.run])
        for xi, (_, run) in zip(x, geometry.iterrows()):
            if np.isfinite(run.get('glathida_bias', np.nan)):
                ax.text(xi, run.thk_mean, f'{run.glathida_bias:+.0f} m', ha='center',
                        va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[r] for r in geometry.run], rotation=45, ha='right')
        ax.set_title(f'{name}: mean thickness (m), GlaThiDa bias on top', fontsize=10)
    fig.suptitle('Posterior ensembles (dots), mean +- 1 std (bars); modelled gradients '
                 'in m ice, GLAMOS in m w.e.', fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'plots', 'compare.png'), dpi=130)


if __name__ == '__main__':
    main()
