#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Posterior TI parameters (melt_f, prcp_fac, temp_bias) of the FROST ES-MDA
calibration against OGGM's own calibration of the same glacier (mb_calib.json,
also the prior mean). OGGM fits one number, the Hugonnet glacier-wide mass
balance 2000-2020, FROST the band dh/dt; the prior std (2.5 / 0.5 / 1.0) is
shaded, so the plot shows how far the band data pull each parameter.

Run from the repository root:
    python experiments/central_europe_TI/compare_oggm_params.py [--dark]
Writes tables/params_vs_oggm.tsv and plots/params_vs_oggm.{pdf,png}.
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
sys.path.insert(0, os.getcwd())
from frost.visualization.plot_style import COLORS as PALETTE, DARK_THEME  # noqa: E402

PARAMS = ['melt_f', 'prcp_fac', 'temp_bias']
LABELS = {'melt_f': 'melt_f (mm w.e. / (°C day))', 'prcp_fac': 'prcp_fac (-)',
          'temp_bias': 'temp_bias (°C)'}
PRIOR_STD = {'melt_f': 2.5, 'prcp_fac': 0.5, 'temp_bias': 1.0}


def glacier_row(results_dir, rgi_id, name, ice_flow):
    path = os.path.join(results_dir, rgi_id, 'calibration_results.json')
    oggm_path = os.path.join(results_dir, rgi_id, 'Preprocess', 'data', rgi_id, 'mb_calib.json')
    if not (os.path.exists(path) and os.path.exists(oggm_path)):
        return None
    with open(path) as f:
        results = json.load(f)
    with open(oggm_path) as f:
        oggm = json.load(f)
    members = np.array(results['final_ensemble'])
    corr = np.corrcoef(members.T)
    row = {'rgi_id': rgi_id, 'glacier': name.replace('_', ' '), 'ice_flow': ice_flow,
           'oggm_mb': oggm['reference_mb'] / 1000}
    for i, p in enumerate(PARAMS):
        row[f'oggm_{p}'] = oggm[p]
        row[f'{p}'] = members[:, i].mean()
        row[f'{p}_std'] = members[:, i].std()
        row[f'{p}_shift'] = (members[:, i].mean() - oggm[p]) / PRIOR_STD[p]
    row['r_melt_f_temp_bias'] = corr[0, 2]
    row['r_prcp_fac_temp_bias'] = corr[1, 2]
    row['r_melt_f_prcp_fac'] = corr[0, 1]
    misfits = [d['misfit'] for d in results['diagnostics']]
    row['misfit_prior'], row['misfit_posterior'] = misfits[0], misfits[-1]
    return row


def plot(table, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    colors = np.where(table.ice_flow, PALETTE['blue'], PALETTE['red'])
    for ax, p in zip(axes, PARAMS):
        x, y = table[f'oggm_{p}'], table[p]
        lo, hi = min(x.min(), (y - table[f'{p}_std']).min()), max(x.max(), (y + table[f'{p}_std']).max())
        pad = 0.1 * (hi - lo)
        line = np.array([lo - pad, hi + pad])
        for k, alpha in [(2, 0.12), (1, 0.2)]:
            ax.fill_between(line, line - k * PRIOR_STD[p], line + k * PRIOR_STD[p],
                            color='grey', alpha=alpha, lw=0)
        ax.plot(line, line, 'k--', lw=0.8)
        ax.errorbar(x, y, yerr=table[f'{p}_std'], fmt='none', ecolor='grey', lw=0.8)
        ax.scatter(x, y, c=colors, s=22, zorder=3)
        for _, r in table[abs(table[f'{p}_shift']) > 1.5].iterrows():
            ax.annotate(r.glacier.split()[0] if r.glacier.split()[0] not in
                        ('Glacier', 'Ghiacciaio', 'Vedretta') else r.glacier.split()[-1],
                        (r[f'oggm_{p}'], r[p]), fontsize=7, xytext=(3, 3), textcoords='offset points')
        ax.set_xlim(line)
        ax.set_ylim(line)
        ax.set_xlabel(f'OGGM {LABELS[p]}')
        ax.set_ylabel(f'FROST posterior {LABELS[p]}')
        ax.set_title(f'{p}: shift {table[f"{p}_shift"].median():+.1f} prior std (median)', fontsize=10)
    ax = axes[3]
    pairs = [('r_melt_f_temp_bias', 'melt_f - temp_bias'), ('r_prcp_fac_temp_bias', 'prcp_fac - temp_bias'),
             ('r_melt_f_prcp_fac', 'melt_f - prcp_fac')]
    for i, (col, label) in enumerate(pairs):
        ax.scatter(np.full(len(table), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(table)),
                   table[col], c=colors, s=18)
    ax.axhline(0, color=plt.rcParams['text.color'], lw=0.8)
    ax.set_xticks(range(len(pairs)), [label for _, label in pairs], fontsize=8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel('posterior correlation between members')
    ax.set_title('trade-offs between the parameters', fontsize=10)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=l)
               for c, l in [(PALETTE['red'], 'no ice flow'), (PALETTE['blue'], 'IGM (ice flow)')]]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{path}.{ext}', dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=os.path.join('data', 'results', 'central_europe_TI'))
    parser.add_argument('--out', default=HERE)
    parser.add_argument('--dark', action='store_true', help='black plots (*_dark)')
    args = parser.parse_args()
    suffix = ''
    if args.dark:
        plt.rcParams.update(DARK_THEME)
        suffix = '_dark'

    rows = []
    for list_name, ice_flow in [('glaciers_noflow.txt', False), ('glaciers_igm.txt', True)]:
        with open(os.path.join(HERE, list_name)) as f:
            for line in f:
                if line.strip():
                    row = glacier_row(args.results, *line.split()[:2], ice_flow)
                    if row is not None:
                        rows.append(row)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(args.out, 'tables', 'params_vs_oggm.tsv'), sep='\t', index=False,
                 float_format='%.3g')
    plot(table, os.path.join(args.out, 'plots', f'params_vs_oggm{suffix}'))
    columns = ['glacier', 'ice_flow'] + [c for p in PARAMS for c in (f'oggm_{p}', p)] + \
        ['r_melt_f_temp_bias', 'misfit_prior', 'misfit_posterior']
    print(table[columns].to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
