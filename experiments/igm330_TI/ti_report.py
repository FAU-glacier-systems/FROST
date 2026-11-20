#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Report of a TI calibration as one A4 PDF: the sanity report of the run
(experiments/central_europe_esmda/sanity_report.py) adapted to the
temperature-index SMB model, whose parameters (melt factor, precipitation
factor, temperature bias) give a mass balance that changes from year to
year and from winter to summer, so it is checked against GLAMOS / WGMS per
year and season, not only as a mean profile.

Pages over all glaciers:
1. Overview: key numbers, setup, flags.
2. Comparison with an earlier TI calibration and with the ELA calibration
   (same glaciers, same scores).
3. TI parameters: posterior against the prior (OGGM's own calibration of
   each glacier) and against the earlier TI calibration; spread left.
4. ES-MDA: misfit and parameter means per iteration, parameter trade-offs.
5. Glacier-wide annual balance per year (interannual variability).
6. Winter and summer balance.
7. Band balance: model against measured, residual by elevation, scores.
8. Summary table with the TI flags.
Then the topic pages of sanity_report.py (thickness, velocity, dh/dt, SMB
profiles, ELA and gradients of the TI profile) and two pages per glacier:
the sanity_report glacier page and a TI page.

Inputs, made first (report_sbatch.sh runs all of it):
    sanity_report.py --results <run>                -> <experiment>/sanity_report
    evaluate.py --results <run> --out <experiment>   -> <experiment>/tables
    evaluate_bands.py --ti <run> --ela <ELA run> --glaciers <experiment>/glaciers.txt
        --out <experiment>
    evaluate_bands.py --ti <earlier TI run> --ela <ELA run> --glaciers ...
        --out <experiment>/comparison/<earlier TI run>
Run from the repository root:
    python experiments/igm330_TI/ti_report.py [--dark]
Writes <experiment>/ti_report/ti_report.pdf (ti_report_dark.pdf), ti_summary.tsv,
REPORT.md and <glacier>/ti_glacier.png.
"""

import argparse
import json
import os
import sys
import textwrap
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'central_europe_esmda'))
import sanity_report as sr  # noqa: E402
from frost.visualization.plot_style import DARK_THEME, MONO, fg  # noqa: E402

RUNS = {'run': os.path.join('data', 'results', 'igm330_TI'),
        'previous': os.path.join('data', 'results', 'central_europe_TI_combined'),
        'ela': os.path.join('data', 'results', 'igm330_live_combined')}
PARAMETERS = [('melt_f', 'Melt factor', 'mm w.e. / (degC day)'),
              ('prcp_fac', 'Precipitation factor', '-'),
              ('temp_bias', 'Temperature bias', 'degC')]
MIN_YEARS = 10  # bands of the mean profile, as evaluate_bands.py

# TI flags (red in the table; REPORT.md); dh/dt chi2 as sanity_report.py
TI_FLAGS = {
    'prior_shift': (lambda v: v > 2, 'a posterior mean more than 2 prior std from OGGM\'s calibration'),
    'spread_ratio_min': (lambda v: v < 0.15, 'posterior spread of a parameter below 15 % of the '
                                             'prior (possible ensemble collapse)'),
    'last_change': (lambda v: v > 0.5, 'a mean still moved by > 0.5 posterior std in the last update'),
    'misfit_posterior': (lambda v: v > 1, 'band dh/dt of the posterior outside the data and model '
                                          'error (misfit chi2 / n > 1)'),
    'unphysical': (lambda v: v > 0, 'members with a melt factor or precipitation factor <= 0'),
    'gw_r': (lambda v: v < 0.5, 'year-to-year glacier-wide balance not reproduced (r < 0.5)'),
    'gw_bias': (lambda v: abs(v) > 0.3, 'glacier-wide annual balance off by > 0.3 m w.e./yr'),
    'winter_bias': (lambda v: abs(v) > 0.5, 'winter balance (bands) off by > 0.5 m w.e.'),
    'summer_bias': (lambda v: abs(v) > 0.5, 'summer balance (bands) off by > 0.5 m w.e.'),
    'dhdt_chi2_data': sr.FLAGS['dhdt_chi2_data'],
}

TABLE = [  # column, header, format (None: glacier name)
    ('glacier', 'Glacier', None),
    ('ice_flow', 'Ice\nflow', '{}'),
    ('melt_f', 'melt\nfactor', '{:.2f}'),
    ('prcp_fac', 'prcp\nfactor', '{:.2f}'),
    ('temp_bias', 'temp\nbias', '{:+.2f}'),
    ('prior_shift', 'max\n|z|', '{:.1f}'),
    ('spread_ratio_min', 'min std\nratio', '{:.2f}'),
    ('last_change', 'last\nstep', '{:.2f}'),
    ('misfit_posterior', 'misfit\npost.', '{:.2f}'),
    ('gw_r', 'gw\nr', '{:.2f}'),
    ('gw_bias', 'gw\nbias', '{:+.2f}'),
    ('gw_rmse', 'gw\nRMSE', '{:.2f}'),
    ('winter_bias', 'winter\nbias', '{:+.2f}'),
    ('summer_bias', 'summer\nbias', '{:+.2f}'),
    ('profile_rmse', 'profile\nRMSE', '{:.2f}'),
    ('dhdt_chi2_data', 'dh/dt\nchi2', '{:.1f}'),
]


def colours():
    """TI, earlier TI, ELA and the observations; the slide colours on black
    pages, darker ones on white."""
    dark = bool(sr.SUFFIX)
    return {'TI': '#F2C230' if dark else '#B8860B', 'previous': '#F39C3D' if dark else '#C0392B',
            'ELA': '#B4B4B4' if dark else '0.55', 'obs': fg(),
            'winter': '#4FA3E0' if dark else '#1F6FB2', 'summer': '#E5484D' if dark else '#B03A2E'}


# ----------------------------------------------------------------- data

def calibration(results, rgi_id):
    path = os.path.join(results, rgi_id, 'calibration_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        result = json.load(f)
    result['names'] = result.get('parameters') or list(result['initial_smb'])
    return result


def glacier_wide(records, obs, model):
    """Area-weighted balance per year over the bands with both values; the
    model std is the area-weighted std of the bands (bands move together,
    so this is close to the std of the glacier-wide balance)."""
    rows = []
    for year, group in records.groupby('year'):
        ok = np.isfinite(group[obs]) & np.isfinite(group[model])
        if not ok.any():
            continue
        g = group[ok]
        w = g.area.to_numpy(dtype=float)
        w = w if np.all(np.isfinite(w)) and w.sum() > 0 else np.ones(len(g))
        std = np.average(g[model + '_std'], weights=w) if model + '_std' in g else np.nan
        rows.append(dict(year=year, obs=np.average(g[obs], weights=w),
                         model=np.average(g[model], weights=w), std=std))
    return pd.DataFrame(rows, columns=['year', 'obs', 'model', 'std'])


def mean_profile(records):
    """Mean balance per band over the years (bands with >= MIN_YEARS)."""
    z = (records.z_min + records.z_max) / 2
    profile = records.assign(z=z).groupby('z').agg(
        obs=('ba', 'mean'), ti=('TI_ba', 'mean'), ti_std=('TI_ba_std', 'mean'),
        ela=('ELA_ba', 'mean'), years=('year', 'nunique')).reset_index()
    return profile[profile.years >= MIN_YEARS]


def score(scores, rgi_id, quantity, variant='TI', column='rmse'):
    row = scores[(scores.rgi_id == rgi_id) & (scores.quantity == quantity)
                 & (scores.variant == variant)]
    return float(row[column].iloc[0]) if len(row) else np.nan


def ti_row(rgi_id, name, result, previous, records, scores, sanity):
    """The TI numbers of one glacier (ti_summary.tsv)."""
    ens = np.array(result['final_ensemble'], dtype=float)
    diagnostics = result['diagnostics']
    row = dict(rgi_id=rgi_id, glacier=name, members=len(ens))
    z, ratio = [], []
    for k, name_ in enumerate(result['names']):
        prior, spread = result['initial_smb'][name_], result['initial_spread'][name_]
        row[name_] = ens[:, k].mean()
        row[name_ + '_std'] = ens[:, k].std()
        row[name_ + '_prior'] = prior
        row[name_ + '_prior_std'] = spread
        row[name_ + '_z'] = (row[name_] - prior) / spread
        row[name_ + '_spread_ratio'] = row[name_ + '_std'] / spread
        z.append(abs(row[name_ + '_z']))
        ratio.append(row[name_ + '_spread_ratio'])
        if previous is not None and name_ in previous['names']:
            row[name_ + '_previous'] = float(np.mean(
                np.array(previous['final_ensemble'])[:, previous['names'].index(name_)]))
    row['prior_shift'] = max(z)
    row['spread_ratio_min'] = min(ratio)
    row['last_change'] = max(diagnostics[-1].get('parameter_change', [0]))
    row['misfit_prior'] = diagnostics[0]['misfit']
    row['misfit_posterior'] = diagnostics[-1]['misfit']
    row['misfit_ratio'] = row['misfit_posterior'] / row['misfit_prior']
    names = result['names']
    if {'melt_f', 'temp_bias'} <= set(names):
        row['corr_melt_temp'] = np.corrcoef(ens[:, names.index('melt_f')],
                                            ens[:, names.index('temp_bias')])[0, 1]
    if {'melt_f', 'prcp_fac'} <= set(names):
        row['corr_melt_prcp'] = np.corrcoef(ens[:, names.index('melt_f')],
                                            ens[:, names.index('prcp_fac')])[0, 1]
    row['unphysical'] = int(sum((ens[:, names.index(p)] <= 0).sum()
                                for p in ('melt_f', 'prcp_fac') if p in names))
    if records is not None and len(records):
        wide = glacier_wide(records, 'ba', 'TI_ba')
        if len(wide) > 2:
            diff = wide.model - wide.obs
            row.update(gw_years=len(wide), gw_r=np.corrcoef(wide.obs, wide.model)[0, 1],
                       gw_bias=diff.mean(), gw_rmse=np.sqrt(np.mean(diff ** 2)))
        row['source'] = records.source.iloc[0]
    for column, quantity, variant, stat in [
            ('band_rmse', 'band annual', 'TI', 'rmse'), ('band_inside', 'band annual', 'TI', 'inside_2std_pct'),
            ('winter_bias', 'band winter', 'TI', 'bias'), ('winter_rmse', 'band winter', 'TI', 'rmse'),
            ('summer_bias', 'band summer', 'TI', 'bias'), ('summer_rmse', 'band summer', 'TI', 'rmse'),
            ('profile_rmse', 'mean profile', 'TI', 'rmse'), ('profile_bias', 'mean profile', 'TI', 'bias'),
            ('ela_gw_r', 'glacier-wide annual', 'ELA', 'r'), ('ela_profile_rmse', 'mean profile', 'ELA', 'rmse')]:
        row[column] = score(scores, rgi_id, quantity, variant, stat)
    s = sanity.set_index('rgi_id').loc[rgi_id]
    for column in ('ice_flow', 'dhdt_obs', 'dhdt_model', 'dhdt_chi2_data', 'dead_bands',
                   'volume_change_pct', 'speed_ratio_obs', 'speed_start_ratio'):
        row[column] = s.get(column, np.nan)
    return row


def flags_of(row):
    return [column for column, (test, _) in TI_FLAGS.items()
            if pd.notna(row.get(column, np.nan)) and test(row[column])]


# ---------------------------------------------------------------- pages

def tiles(fig, items, top):
    """sanity_report.metric_tiles with free tiles: (value, label, note)."""
    width, height, gap = 0.21, 0.07, 0.0133
    face = '#161616' if sr.SUFFIX else '#f0f0f0'
    for i, (value, label, note) in enumerate(items[:8]):
        x = 0.06 + (i % 4) * (width + gap)
        y = top - (i // 4) * (height + gap)
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x, y - height), width, height, boxstyle='round,pad=0,rounding_size=0.006',
            transform=fig.transFigure, facecolor=face, edgecolor='none', zorder=0))
        fig.text(x + 0.012, y - 0.008, label, fontsize=6.5, va='top', color=sr.muted())
        fig.text(x + 0.012, y - 0.024, value, fontsize=17, weight='bold', va='top')
        fig.text(x + 0.012, y - height + 0.007, note, fontsize=5.6, va='bottom', color=sr.muted())
    return top - ((min(len(items), 8) + 3) // 4) * (height + gap)


def band_value(table, quantity, variant, column):
    row = table[(table.quantity == quantity) & (table.variant == variant)]
    return float(row[column].iloc[0]) if len(row) else np.nan


def overview_page(ti, summary, band, results):
    fig = plt.figure(figsize=sr.A4)
    experiment = os.path.basename(os.path.normpath(results))
    fig.text(0.06, 0.975, 'FROST TI calibration report', fontsize=16, weight='bold', va='top')
    fig.text(0.06, 0.95, f'{experiment}  ·  {results}  ·  {date.today().isoformat()}', fontsize=7.5,
             va='top', color=sr.muted())
    flat = sr.no_flow(summary)
    moving = summary[~flat]
    items = [(f'{len(ti)}', 'glaciers', f'{(~flat).sum()} with IGM ice flow, {flat.sum()} without'),
             (f'{band_value(band, "band annual", "TI", "rmse"):.2f}', 'band annual RMSE (m w.e.)',
              f'{band_value(band, "band annual", "TI", "inside_2std_pct"):.0f} % in +-2 std; '
              f'ELA {band_value(band, "band annual", "ELA", "rmse"):.2f}'),
             (f'{ti.gw_r.median():.2f}', 'glacier-wide r, years',
              f'median; RMSE {band_value(band, "glacier-wide annual", "TI", "rmse"):.2f} m w.e.; '
              f'ELA r {ti.ela_gw_r.median():.2f}'),
             (f'{band_value(band, "mean profile", "TI", "rmse"):.2f}', 'mean profile RMSE',
              f'bias {band_value(band, "mean profile", "TI", "bias"):+.2f}; ELA '
              f'{band_value(band, "mean profile", "ELA", "rmse"):.2f}'),
             (f'{band_value(band, "band winter", "TI", "rmse"):.2f}', 'winter balance RMSE',
              f'bands, bias {band_value(band, "band winter", "TI", "bias"):+.2f} m w.e.'),
             (f'{band_value(band, "band summer", "TI", "rmse"):.2f}', 'summer balance RMSE',
              f'bands, bias {band_value(band, "band summer", "TI", "bias"):+.2f} m w.e.'),
             (f'{summary.dhdt_chi2_data.median():.1f}', 'dh/dt band chi2 / n',
              f'median, {(summary.dhdt_chi2_data > 4).sum()} glaciers > 4'),
             (f'{moving.speed_ratio_obs.median():.2f}x' if len(moving) else '-',
              'speed / ITS_LIVE', f'median of {len(moving)} glaciers with ice flow')]
    y = tiles(fig, items, 0.93) - 0.015
    left, right = sr.setup_items(summary, results)
    sr.text_column(fig, 0.06, y, left, chars=66, size=6.8)
    found = [('h', 'Flags (TI summary table)')]
    for column, (test, text) in TI_FLAGS.items():
        hits = [r.glacier for r in ti.itertuples() if column in r.flags]
        if hits:
            found.append(('t', f'{text}: ' + ', '.join(h.replace('_', ' ') for h in hits)))
    if len(found) == 1:
        found.append(('t', 'none'))
    sr.text_column(fig, 0.53, y, right + found, chars=66, size=6.8)
    return fig


def comparison_page(ti, band, band_previous, sanity, sanity_previous, sanity_ela, names):
    c = colours()
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'Comparison with the earlier TI and with the ELA calibration',
                  f'{names["run"]}: this run. {names["previous"]}: the earlier TI calibration. '
                  f'{names["ela"]}: ELA calibration of the same glaciers (constant SMB). Scores '
                  'against GLAMOS / WGMS of all glaciers with references (evaluate_bands.py), mean '
                  'over the glaciers; r: median.')
    rows = [('Band annual RMSE (m w.e.)', 'band annual', 'rmse', '{:.2f}'),
            ('Band annual bias', 'band annual', 'bias', '{:+.2f}'),
            ('Band annual inside +-2 std (%)', 'band annual', 'inside_2std_pct', '{:.0f}'),
            ('Glacier-wide annual RMSE', 'glacier-wide annual', 'rmse', '{:.2f}'),
            ('Glacier-wide annual r (years)', 'glacier-wide annual', 'r_median', '{:.2f}'),
            ('Mean profile RMSE', 'mean profile', 'rmse', '{:.2f}'),
            ('Mean profile bias', 'mean profile', 'bias', '{:+.2f}'),
            ('Winter balance RMSE (bands)', 'band winter', 'rmse', '{:.2f}'),
            ('Winter balance bias', 'band winter', 'bias', '{:+.2f}'),
            ('Summer balance RMSE (bands)', 'band summer', 'rmse', '{:.2f}'),
            ('Summer balance bias', 'band summer', 'bias', '{:+.2f}')]
    cells = []
    for label, quantity, column, fmt in rows:
        values = [band_value(band, quantity, 'TI', column),
                  band_value(band_previous, quantity, 'TI', column) if band_previous is not None else np.nan,
                  band_value(band, quantity, 'ELA', column)]
        cells.append([label] + [fmt.format(v) if np.isfinite(v) else '-' for v in values])

    def sanity_cells(label, fn, fmt):
        cells.append([label] + [fmt.format(fn(s)) if s is not None and np.isfinite(fn(s)) else '-'
                                for s in (sanity, sanity_previous, sanity_ela)])
    sanity_cells('dh/dt band chi2 / n (median)', lambda s: s.dhdt_chi2_data.median(), '{:.2f}')
    sanity_cells('dh/dt glacier mean MAE (m/yr)',
                 lambda s: (s.dhdt_model - s.dhdt_obs).abs().mean(), '{:.3f}')
    sanity_cells('Speed / ITS_LIVE (median, flow)',
                 lambda s: s[s.ice_flow == 'igm'].speed_ratio_obs.median(), '{:.2f}')
    sanity_cells('Start-up speed 2000 / obs. year',
                 lambda s: s[s.ice_flow == 'igm'].speed_start_ratio.median(), '{:.2f}')
    ax = fig.add_axes([0.06, 0.6, 0.88, 0.32])
    ax.axis('off')
    table = ax.table(cellText=cells, colLabels=['', names['run'], names['previous'], names['ela']],
                     colWidths=[0.37, 0.21, 0.21, 0.21], loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    for (r, col), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor(sr.muted())
        cell.set_facecolor(sr.TABLE_COLOURS['header'] if r == 0 else sr.TABLE_COLOURS['cell'])
        cell.get_text().set_color(fg())
        cell.get_text().set_family(MONO)
        if col == 0:
            cell.get_text().set_ha('left')

    grid = fig.add_gridspec(1, 2, left=0.16, right=0.96, top=0.55, bottom=0.17, wspace=0.55)
    order = ti.reset_index(drop=True)
    y = np.arange(len(order))
    for k, (column, previous_column, ela_column, title, text) in enumerate([
            ('gw_rmse', 'gw_rmse_previous', 'ela_gw_rmse', 'Glacier-wide annual RMSE (m w.e.)',
             '(a) Glacier-wide annual balance per year against GLAMOS / WGMS, RMSE per glacier.'),
            ('profile_rmse', 'profile_rmse_previous', 'ela_profile_rmse', 'Mean profile RMSE (m w.e./yr)',
             '(b) Mean balance per band over the measured years, RMSE per glacier.')]):
        ax = fig.add_subplot(grid[0, k])
        ax.barh(y - 0.27, order[column], height=0.27, color=c['TI'], label=names['run'])
        ax.barh(y, order.get(previous_column, np.nan), height=0.27, color=c['previous'],
                label=names['previous'])
        ax.barh(y + 0.27, order.get(ela_column, np.nan), height=0.27, color=c['ELA'], label=names['ela'])
        ax.set_yticks(y)
        ax.set_yticklabels(sr.labels(order), fontsize=6)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(title)
        sr.small(ax)
        ax.tick_params(axis='y', labelsize=6)
        sr.caption(ax, text)
        if k == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=6.5,
                       bbox_to_anchor=(0.5, 0.56), frameon=False)
    fig.text(0.06, 0.03, sr.NO_FLOW_NOTE, fontsize=6.5, color=sr.muted())
    return fig


def parameter_panel(ax, ti, p, label, unit):
    """Posterior mean +- std per glacier against the prior mean +- std
    (grey bar) and the earlier calibration's mean (open diamond)."""
    c = colours()
    colors, markers, _ = sr.glacier_style(ti)
    y = np.arange(len(ti))
    prior = ti[p + '_prior'].to_numpy()
    spread = ti[p + '_prior_std'].to_numpy()
    ax.barh(y, 2 * spread, left=prior - spread, height=0.6, color=c['ELA'], alpha=0.35)
    ax.plot(prior, y, '|', color=c['ELA'], markersize=7)
    for i, row in enumerate(ti.itertuples()):
        ax.errorbar(getattr(row, p), i, xerr=getattr(row, p + '_std'), fmt=markers[i],
                    color=colors[i], markersize=4, lw=1)
        previous = getattr(row, p + '_previous', np.nan)
        if np.isfinite(previous):
            ax.plot(previous, i, 'D', mfc='none', mec=colors[i], markersize=4, mew=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sr.labels(ti), fontsize=6)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'{label} ({unit})')
    sr.small(ax)
    ax.tick_params(axis='y', labelsize=6)


def parameter_page(ti):
    c = colours()
    colors, markers, _ = sr.glacier_style(ti)
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'TI parameters',
                  'Posterior mean +- 1 std of the members (glacier colour) against the prior mean '
                  '+- 1 std (grey bar; the mean is OGGM\'s calibration of the glacier to the '
                  'Hugonnet glacier-wide mass balance) and the posterior mean of the earlier TI '
                  'calibration (open diamond). The band dh/dt pull the parameters away from OGGM\'s '
                  'single-number fit; melt factor and temperature bias trade off (page ES-MDA).')
    axes = sr.overview_grid(fig)
    y = np.arange(len(ti))
    for ax, (p, label, unit), letter in zip(axes.flat[:3], PARAMETERS, 'abc'):
        parameter_panel(ax, ti, p, label, unit)
        sr.caption(ax, f'({letter}) {label}: posterior against prior and the earlier TI '
                       'calibration.')
    ax = axes[1, 1]
    for k, (p, label, _) in enumerate(PARAMETERS):
        ax.scatter(ti[p + '_spread_ratio'], y + (k - 1) * 0.22, s=14, color=colors,
                   marker=['o', 's', '^'][k], label=label)
    ax.axvline(1, color=c['ELA'], lw=0.8)
    ax.axvline(0.15, color=c['previous'], lw=0.8, ls='--')
    ax.set_yticks(y)
    ax.set_yticklabels(sr.labels(ti), fontsize=6)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('posterior std / prior std')
    ax.set_xlim(0, 1.1)
    sr.small(ax)
    ax.tick_params(axis='y', labelsize=6)
    handles = [Line2D([], [], ls='none', marker=m, color=fg(), markersize=4) for m in 'os^']
    ax.legend(handles, [p[0] for p in PARAMETERS], fontsize=6, loc='lower right')
    sr.caption(ax, '(d) Spread left after the calibration: posterior std over prior std (o melt '
                   'factor, s precipitation factor, ^ temperature bias). Dashed: 0.15, below it the '
                   'ensemble may have collapsed (32 members with ice flow).')
    fig.text(0.06, 0.03, sr.NO_FLOW_NOTE, fontsize=6.5, color=sr.muted())
    return fig


def esmda_page(ti, results_by_id):
    c = colours()
    colors, markers, styles = sr.glacier_style(ti)
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'ES-MDA: iterations and parameter trade-offs',
                  'Iteration 0 is the prior ensemble, the last the posterior (each with its own '
                  'forward run). Misfit: chi2 / n of the ensemble mean against the band dh/dt with '
                  'the data and model error (about 1 when the fit matches the error). Parameter '
                  'means in prior std from the prior mean; at iteration 0 they differ from 0 by the '
                  'sampling of the prior, the same for every glacier (seed 1).')
    grid = fig.add_gridspec(3, 2, left=0.16, right=0.96, top=0.9, bottom=0.14, hspace=0.6,
                            wspace=0.55)
    ax = fig.add_subplot(grid[0, 0])
    for i, row in enumerate(ti.itertuples()):
        d = results_by_id[row.rgi_id]['diagnostics']
        ax.plot(range(len(d)), [x['misfit'] for x in d], ls=styles[i], marker=markers[i],
                color=colors[i], lw=0.9, markersize=2.5)
    ax.set_yscale('log')
    sr.plain_log_axis(ax.yaxis, [0.1, 0.3, 1, 3, 10])
    ax.axhline(1, color=c['ELA'], lw=0.8)
    ax.set_xlabel('iteration')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel('misfit chi2 / n')
    sr.small(ax)
    sr.caption(ax, '(a) Misfit per iteration.')
    for k, ((p, label, _), slot) in enumerate(zip(PARAMETERS, [grid[0, 1], grid[1, 0], grid[1, 1]])):
        ax = fig.add_subplot(slot)
        for i, row in enumerate(ti.itertuples()):
            result = results_by_id[row.rgi_id]
            j = result['names'].index(p)
            prior, spread = result['initial_smb'][p], result['initial_spread'][p]
            path = [(x['parameter_mean'][j] - prior) / spread for x in result['diagnostics']]
            ax.plot(range(len(path)), path, ls=styles[i], marker=markers[i], color=colors[i],
                    lw=0.9, markersize=2.5)
        ax.axhline(0, color=c['ELA'], lw=0.8)
        ax.set_xlabel('iteration')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel(f'{p}: (mean - prior) / prior std')
        sr.small(ax)
        sr.caption(ax, f'({"bcd"[k]}) {label}: ensemble mean per iteration.')
    for k, (column, label) in enumerate([('corr_melt_temp', 'melt factor and temperature bias'),
                                         ('corr_melt_prcp', 'melt factor and precipitation factor')]):
        ax = fig.add_subplot(grid[2, k])
        sr.name_bars(ax, ti, ti[column], colors, 'correlation of the posterior members')
        ax.set_xlim(-1, 1)
        sr.caption(ax, f'({"ef"[k]}) Correlation of {label} over the posterior members: '
                       'strongly negative means the data fix only their combination.')
    sr.glacier_legend(fig, ti, colors, markers, styles=styles)
    return fig


def annual_page(ti, records_by_id):
    c = colours()
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'Glacier-wide annual balance per year',
                  'Area-weighted over the measured bands, in the measured periods (floating '
                  'GLAMOS dates, WGMS dates or the hydrological year). Measured (line with dots), '
                  'TI posterior mean +- 2 std (band), ELA calibration (dashed, constant but for the '
                  'period length). r and RMSE (m w.e.) of TI in the title.')
    n = len(ti)
    cols = 5
    rows = int(np.ceil(n / cols))
    grid = fig.add_gridspec(rows, cols, left=0.07, right=0.98, top=0.875, bottom=0.09, hspace=0.75,
                            wspace=0.35)
    for i, row in enumerate(ti.itertuples()):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        records = records_by_id.get(row.rgi_id)
        title = sr.labels(ti)[i]
        if records is None or not len(records):
            ax.text(0.5, 0.5, 'no band data', transform=ax.transAxes, ha='center', fontsize=6,
                    color=sr.muted())
            ax.set_title(title, fontsize=6.5)
            continue
        wide = glacier_wide(records, 'ba', 'TI_ba')
        ela = glacier_wide(records, 'ba', 'ELA_ba')
        ax.fill_between(wide.year, wide.model - 2 * wide['std'], wide.model + 2 * wide['std'],
                        color=c['TI'], alpha=0.3, lw=0)
        ax.plot(wide.year, wide.model, color=c['TI'], lw=1)
        ax.plot(ela.year, ela.model, color=c['ELA'], lw=0.8, ls='--')
        ax.plot(wide.year, wide.obs, color=c['obs'], lw=0.8, marker='o', markersize=1.8)
        ax.set_title(f'{title}\nr {row.gw_r:.2f}, RMSE {row.gw_rmse:.2f}', fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        ax.set_xticks([2000, 2010, 2019])
    handles = [Line2D([], [], color=c['obs'], marker='o', markersize=2, lw=0.8),
               Line2D([], [], color=c['TI'], lw=1), Line2D([], [], color=c['ELA'], lw=0.8, ls='--')]
    fig.legend(handles, ['GLAMOS / WGMS', 'TI (+- 2 std)', 'ELA'], loc='lower center', ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.025))
    return fig


def season_page(ti, records_by_id):
    c = colours()
    colors, markers, _ = sr.glacier_style(ti)
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'Winter and summer balance',
                  'Measured periods split at the end-of-winter survey (GLAMOS) or the WGMS '
                  'midseason date; the TI model of every member summed over the same days. Glaciers '
                  'without a winter date have no seasonal values. A winter balance too low with a '
                  'summer balance too high means the precipitation factor makes up for melt.')
    axes = sr.overview_grid(fig)
    for ax, (season, obs, model), letter in zip(axes[0], [('Winter', 'bw', 'TI_bw'),
                                                          ('Summer', 'bs', 'TI_bs')], 'ab'):
        allx, ally = [], []
        for i, row in enumerate(ti.itertuples()):
            records = records_by_id.get(row.rgi_id)
            if records is None or not len(records):
                continue
            wide = glacier_wide(records, obs, model)
            ax.scatter(wide.obs, wide.model, s=8, color=colors[i], marker=markers[i], lw=0)
            allx += list(wide.obs)
            ally += list(wide.model)
        allx, ally = np.array(allx), np.array(ally)
        if len(allx) > 2:
            lim = [min(allx.min(), ally.min()) - 0.2, max(allx.max(), ally.max()) + 0.2]
            ax.plot(lim, lim, color=c['ELA'], lw=0.8, ls='--')
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            diff = ally - allx
            sr.stats_box(ax, f'r {np.corrcoef(allx, ally)[0, 1]:.2f}\nbias {diff.mean():+.2f}\n'
                             f'RMSE {np.sqrt(np.mean(diff ** 2)):.2f}')
        ax.set_aspect('equal')
        ax.set_xlabel(f'{season} balance, measured (m w.e.)')
        ax.set_ylabel('TI posterior mean (m w.e.)')
        sr.small(ax)
        sr.caption(ax, f'({letter}) Glacier-wide {season.lower()} balance per year, all glaciers.')
    for ax, (column, season), letter in zip(axes[1], [('winter_bias', 'winter'),
                                                      ('summer_bias', 'summer')], 'cd'):
        sr.name_bars(ax, ti, ti[column].fillna(0), colors, f'{season} bias, TI - measured (m w.e.)')
        ax.axvline(0, color=c['ELA'], lw=0.8)
        sr.caption(ax, f'({letter}) Mean {season} bias over the band-years of each glacier '
                       '(0: no seasonal data).')
    sr.glacier_legend(fig, ti, colors, markers)
    return fig


def band_page(ti, records):
    c = colours()
    colors, markers, styles = sr.glacier_style(ti)
    index = {r: i for i, r in enumerate(ti.rgi_id)}
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'Annual balance per band',
                  'Every measured band in every year (GLAMOS / WGMS) against the TI posterior mean '
                  'over the same period. Residuals by the relative height of the band on the '
                  'glacier (0: lowest measured band, 1: highest) show whether the profile is too '
                  'steep or too flat.')
    axes = sr.overview_grid(fig)
    ax = axes[0, 0]
    ok = records.TI_ba.notna() & records.ba.notna()
    r = records[ok]
    ax.scatter(r.ba, r.TI_ba, s=2, c=[colors[index[x]] for x in r.rgi_id], lw=0, alpha=0.6)
    lim = [min(r.ba.min(), r.TI_ba.min()) - 0.3, max(r.ba.max(), r.TI_ba.max()) + 0.3]
    ax.plot(lim, lim, color=c['ELA'], lw=0.8, ls='--')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    diff = r.TI_ba - r.ba
    sr.stats_box(ax, f'n {len(r)}\nr {np.corrcoef(r.ba, r.TI_ba)[0, 1]:.2f}\nbias {diff.mean():+.2f}'
                     f'\nRMSE {np.sqrt(np.mean(diff ** 2)):.2f}')
    ax.set_xlabel('measured (m w.e.)')
    ax.set_ylabel('TI posterior mean (m w.e.)')
    sr.small(ax)
    sr.caption(ax, '(a) Annual balance of every band and year.')
    ax = axes[0, 1]
    pooled = []
    for rgi_id, g in records[ok].groupby('rgi_id'):
        z = (g.z_min + g.z_max) / 2
        rel = (z - z.min()) / max(z.max() - z.min(), 1)
        bins = np.clip((rel * 5).astype(int), 0, 4)
        res = (g.TI_ba - g.ba).groupby(bins).mean()
        ela = (g.ELA_ba - g.ba).groupby(bins).mean()
        i = index[rgi_id]
        ax.plot((res.index + 0.5) / 5, res.values, color=colors[i], ls=styles[i], lw=0.8,
                marker=markers[i], markersize=2.5)
        pooled.append(pd.DataFrame({'bin': res.index, 'ti': res.values, 'ela': ela.values}))
    pooled = pd.concat(pooled).groupby('bin').mean()
    ax.plot((pooled.index + 0.5) / 5, pooled.ti, color=c['TI'], lw=2.2, label='TI, mean')
    ax.plot((pooled.index + 0.5) / 5, pooled.ela, color=c['ELA'], lw=2.2, ls='--', label='ELA, mean')
    ax.axhline(0, color=c['ELA'], lw=0.8)
    ax.set_xlabel('relative height of the band (0 lowest, 1 highest)')
    ax.set_ylabel('model - measured (m w.e.)')
    ax.legend(fontsize=6)
    sr.small(ax)
    sr.caption(ax, '(b) Mean residual in fifths of the measured height range, per glacier and '
                   'over all glaciers (thick); ELA calibration for comparison.')
    ax = axes[1, 0]
    sr.name_bars(ax, ti, ti.gw_r.fillna(0), colors, 'r of the glacier-wide balance between years')
    ax.scatter(ti.ela_gw_r, np.arange(len(ti)), marker='|', s=40, color=fg(), zorder=3)
    ax.set_xlim(-1, 1)
    sr.caption(ax, '(c) Year-to-year glacier-wide balance: r of TI (bars) and of the ELA '
                   'calibration (|, only the period length varies).')
    ax = axes[1, 1]
    sr.name_bars(ax, ti, ti.band_inside.fillna(0), colors, 'band-years inside +- 2 std (%)')
    ax.set_xlim(0, 100)
    sr.caption(ax, '(d) Share of the band-years whose measured balance is inside the TI '
                   'posterior mean +- 2 std of the members.')
    sr.glacier_legend(fig, ti, colors, markers, styles=styles)
    return fig


def table_page(ti):
    colors, _, _ = sr.glacier_style(ti)
    cells, cell_colours = [], []
    for name, (_, row) in zip(sr.labels(ti), ti.iterrows()):
        line, colour = [], []
        for column, _, fmt in TABLE:
            value = row.get(column, np.nan)
            if fmt is None:
                line.append(name)
            elif column == 'ice_flow':
                line.append({'igm': 'IGM', 'none': 'none'}.get(value, '-'))
            else:
                line.append(fmt.format(value) if pd.notna(value) else '-')
            flagged = column in TI_FLAGS and pd.notna(value) and TI_FLAGS[column][0](value)
            colour.append(sr.TABLE_COLOURS['flag'] if flagged else sr.TABLE_COLOURS['cell'])
        cells.append(line)
        cell_colours.append(colour)
    fig = plt.figure(figsize=sr.A4)
    sr.page_title(fig, 'TI summary table')
    ax = fig.add_axes([0.03, 0.3, 0.94, 0.63])
    ax.axis('off')
    widths = [0.15] + [0.85 / (len(TABLE) - 1)] * (len(TABLE) - 1)
    table = ax.table(cellText=cells, colLabels=[t[1] for t in TABLE], cellColours=cell_colours,
                     colWidths=widths, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(5.6)
    table.scale(1, 1.3)
    for (r, col), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor(sr.muted())
        if r == 0:
            cell.set_height(cell.get_height() * 2)
            cell.set_facecolor(sr.TABLE_COLOURS['header'])
        cell.get_text().set_color(colors[r - 1] if r > 0 and col == 0 else fg())
        cell.get_text().set_family(MONO)
        if col == 0:
            cell.get_text().set_ha('left')
            cell.get_text().set_weight('bold' if r > 0 else 'normal')
    notes = ['Parameters: posterior mean; melt factor in mm w.e. / (degC day), temperature bias in '
             'degC. max |z|: largest distance of a posterior mean from the '
             'prior mean in prior std. min std / prior std: smallest spread left. last change: '
             'largest step of a mean in the last update, in posterior std. misfit post.: chi2 / n '
             'of the band dh/dt of the posterior with the data and model error. gw: glacier-wide annual balance per year (bands with data, '
             'measured periods), r between the years, bias and RMSE in m w.e. winter / summer bias '
             'and profile RMSE: bands, m w.e. dh/dt chi2: chi2 / n of the band dh/dt (data error). '
             'Red: flagged. ' + sr.NO_FLOW_NOTE + '.'] + \
        [f'- {text}' for _, text in TI_FLAGS.values()]
    fig.canvas.draw()
    y = fig.transFigure.inverted().transform(
        table.get_window_extent(fig.canvas.get_renderer()))[0, 1] - 0.02
    for note in notes:
        for line in textwrap.wrap(note, 115):
            fig.text(0.06, y, line, fontsize=7, va='top', color=sr.muted())
            y -= 0.014
    return fig


def glacier_page(row, result, previous, records, color):
    """TI page of one glacier: parameters, ES-MDA, balance per year and
    season, mean profile."""
    c = colours()
    fig = plt.figure(figsize=sr.A4)
    name = row.glacier.replace('_', ' ')
    flags = ', '.join(row.flags) or 'none'
    numbers = (f'melt_f {row.melt_f:.2f} +- {row.melt_f_std:.2f} (prior {row.melt_f_prior:.2f}), '
               f'prcp_fac {row.prcp_fac:.2f} +- {row.prcp_fac_std:.2f} ({row.prcp_fac_prior:.2f}), '
               f'temp_bias {row.temp_bias:+.2f} +- {row.temp_bias_std:.2f} ({row.temp_bias_prior:+.2f}); '
               f'misfit {row.misfit_prior:.2f} -> {row.misfit_posterior:.2f}; '
               f'glacier-wide r {row.gw_r:.2f}, RMSE {row.gw_rmse:.2f} m w.e.; flags: {flags}')
    flow = 'without ice flow' if row.ice_flow == 'none' else 'with IGM ice flow'
    sr.page_header(fig, name, row.rgi_id, f'TI calibration, {flow}, {row.members} members', numbers)
    grid = fig.add_gridspec(3, 2, left=0.09, right=0.97, top=0.86, bottom=0.07, hspace=0.62,
                            wspace=0.3)
    ens = np.array(result['final_ensemble'], dtype=float)
    names = result['names']
    for k, (px, py) in enumerate([('melt_f', 'temp_bias'), ('melt_f', 'prcp_fac')]):
        ax = fig.add_subplot(grid[0, k])
        jx, jy = names.index(px), names.index(py)
        ax.scatter(ens[:, jx], ens[:, jy], s=9, color=color, lw=0, label='posterior members')
        prior = result['initial_smb']
        spread = result['initial_spread']
        ax.add_patch(matplotlib.patches.Rectangle(
            (prior[px] - spread[px], prior[py] - spread[py]), 2 * spread[px], 2 * spread[py],
            facecolor=c['ELA'], alpha=0.2, lw=0))
        ax.plot(prior[px], prior[py], 'x', color=c['ELA'], markersize=7, label='prior (OGGM) +- 1 std')
        if previous is not None and px in previous['names'] and py in previous['names']:
            p = np.array(previous['final_ensemble'], dtype=float)
            ax.plot(p[:, previous['names'].index(px)].mean(), p[:, previous['names'].index(py)].mean(),
                    'D', mfc='none', mec=c['previous'], markersize=6, label='earlier TI, mean')
        ax.set_xlabel(f'{px} (mm w.e. / (degC day))')
        ax.set_ylabel(py + (' (degC)' if py == 'temp_bias' else ''))
        ax.legend(fontsize=5.5, loc='best')
        sr.small(ax)
        corr = row.corr_melt_temp if py == 'temp_bias' else row.corr_melt_prcp
        sr.caption(ax, f'({"ab"[k]}) Posterior members of {px} and {py} (r {corr:.2f}) with the '
                       'prior box.')
    ax = fig.add_subplot(grid[1, 0])
    d = result['diagnostics']
    for j, (p, _, _) in enumerate(PARAMETERS):
        if p not in names:
            continue
        k = names.index(p)
        ax.plot(range(len(d)), [(x['parameter_mean'][k] - result['initial_smb'][p])
                                / result['initial_spread'][p] for x in d],
                marker='os^'[j], lw=1, markersize=3, color=[c['TI'], c['winter'], c['summer']][j], label=p)
    ax.axhline(0, color=c['ELA'], lw=0.8)
    twin = ax.twinx()
    twin.plot(range(len(d)), [x['misfit'] for x in d], color=fg(), lw=0.8, ls=':', marker='.')
    twin.set_ylabel('misfit chi2 / n (dotted)', fontsize=7)
    twin.tick_params(labelsize=6.5)
    twin.grid(False)
    ax.set_xlabel('iteration (0 prior)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel('(mean - prior) / prior std')
    ax.legend(fontsize=5.5, loc='best')
    sr.small(ax)
    sr.caption(ax, '(c) ES-MDA: parameter means and misfit per iteration.')
    panels = [grid[1, 1], grid[2, 0], grid[2, 1]]
    if records is None or not len(records):
        for k, slot in enumerate(panels):
            ax = fig.add_subplot(slot)
            ax.text(0.5, 0.5, 'no GLAMOS / WGMS band data', transform=ax.transAxes, ha='center',
                    color=sr.muted(), fontsize=7)
            ax.axis('off')
        return fig
    ax = fig.add_subplot(panels[0])
    wide = glacier_wide(records, 'ba', 'TI_ba')
    ela = glacier_wide(records, 'ba', 'ELA_ba')
    ax.fill_between(wide.year, wide.model - 2 * wide['std'], wide.model + 2 * wide['std'],
                    color=c['TI'], alpha=0.3, lw=0)
    ax.plot(wide.year, wide.model, color=c['TI'], lw=1.2, label='TI (+- 2 std)')
    ax.plot(ela.year, ela.model, color=c['ELA'], lw=1, ls='--', label='ELA')
    ax.plot(wide.year, wide.obs, color=c['obs'], lw=0.9, marker='o', markersize=2.5,
            label=row.source)
    ax.set_ylabel('annual balance (m w.e.)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=5.5, loc='best')
    sr.small(ax)
    sr.caption(ax, f'(d) Glacier-wide annual balance per year: r {row.gw_r:.2f}, bias '
                   f'{row.gw_bias:+.2f}, RMSE {row.gw_rmse:.2f} m w.e. (ELA r {row.ela_gw_r:.2f}).')
    ax = fig.add_subplot(panels[1])
    seasonal = False
    for obs, model, key, label in [('bw', 'TI_bw', 'winter', 'winter'), ('bs', 'TI_bs', 'summer', 'summer')]:
        wide = glacier_wide(records, obs, model)
        if not len(wide):
            continue
        seasonal = True
        ax.fill_between(wide.year, wide.model - 2 * wide['std'], wide.model + 2 * wide['std'],
                        color=c[key], alpha=0.2, lw=0)
        ax.plot(wide.year, wide.model, color=c[key], lw=1.1, label=f'TI {label}')
        ax.plot(wide.year, wide.obs, color=c[key], lw=0, marker='o', markersize=2.5,
                label=f'measured {label}')
    if seasonal:
        ax.axhline(0, color=c['ELA'], lw=0.6)
        ax.legend(fontsize=5.5, loc='best', ncol=2)
    else:
        ax.text(0.5, 0.5, 'no seasonal data', transform=ax.transAxes, ha='center', color=sr.muted(),
                fontsize=7)
    ax.set_ylabel('balance (m w.e.)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    sr.small(ax)
    sr.caption(ax, f'(e) Glacier-wide winter and summer balance per year (bias winter '
                   f'{row.winter_bias:+.2f}, summer {row.summer_bias:+.2f} m w.e., bands).')
    ax = fig.add_subplot(panels[2])
    profile = mean_profile(records)
    if len(profile):
        ax.fill_betweenx(profile.z, profile.ti - 2 * profile.ti_std, profile.ti + 2 * profile.ti_std,
                         color=c['TI'], alpha=0.3, lw=0)
        ax.plot(profile.ti, profile.z, color=c['TI'], lw=1.2, label='TI')
        ax.plot(profile.ela, profile.z, color=c['ELA'], lw=1, ls='--', label='ELA')
        ax.plot(profile.obs, profile.z, color=c['obs'], lw=0.9, marker='o', markersize=2.5,
                label=row.source)
        ax.axvline(0, color=c['ELA'], lw=0.6)
        ax.legend(fontsize=5.5, loc='best')
    ax.set_xlabel('mean annual balance (m w.e./yr)')
    ax.set_ylabel('elevation (m)')
    sr.small(ax)
    sr.caption(ax, f'(f) Mean balance per band over the measured years (bands with >= {MIN_YEARS} '
                   f'years): RMSE {row.profile_rmse:.2f} (ELA {row.ela_profile_rmse:.2f}) m w.e./yr.')
    return fig


# ------------------------------------------------------------------ main

def write_markdown(ti, out, results):
    lines = [f'# TI calibration report: {results}', '',
             'The full report is ti_report.pdf (A4); numbers in ti_summary.tsv. Flags:', '']
    lines += [f'- **{column}**: {text}' for column, (_, text) in TI_FLAGS.items()]
    lines += ['', '| Glacier | Flow | melt_f | prcp_fac | temp_bias | gw r | gw RMSE | '
              'winter / summer bias | profile RMSE | dh/dt chi2 | Flags |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for row in ti.itertuples():
        lines.append(
            f'| {row.glacier} | {row.ice_flow} | {row.melt_f:.2f} | {row.prcp_fac:.2f} | '
            f'{row.temp_bias:+.2f} | {row.gw_r:.2f} | {row.gw_rmse:.2f} | '
            f'{row.winter_bias:+.2f} / {row.summer_bias:+.2f} | {row.profile_rmse:.2f} | '
            f'{row.dhdt_chi2_data:.1f} | {", ".join(row.flags) or "-"} |')
    with open(os.path.join(out, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', default=RUNS['run'], help='results of the TI calibration')
    parser.add_argument('--previous', default=RUNS['previous'], help='earlier TI calibration')
    parser.add_argument('--ela', default=RUNS['ela'], help='ELA calibration of the same glaciers')
    parser.add_argument('--dark', action='store_true', help='black pages (ti_report_dark.pdf)')
    parser.add_argument('--png', type=int, default=0, metavar='DPI',
                        help='also every page as <out>/pages/NN.png at this resolution')
    args = parser.parse_args()
    if args.dark:
        sr.SUFFIX = '_dark'
        plt.rcParams.update(DARK_THEME)
        plt.rcParams['grid.color'] = '0.35'
        sr.TABLE_COLOURS.update(cell='black', header='#333333', flag='#7a2424')
    suffix = sr.SUFFIX
    names = {key: os.path.basename(os.path.normpath(path)) for key, path in
             [('run', args.results), ('previous', args.previous), ('ela', args.ela)]}
    folder, settings = sr.experiment_settings(args.results)
    out = os.path.join(folder, 'ti_report')
    os.makedirs(out, exist_ok=True)

    sanity_dir = os.path.join(folder, 'sanity_report')
    summary = pd.read_csv(os.path.join(sanity_dir, 'summary.tsv'), sep='\t')
    with open(os.path.join(sanity_dir, 'overview.pkl'), 'rb') as f:
        data = pd.read_pickle(f)
    tables = os.path.join(folder, 'tables')
    records = pd.read_csv(os.path.join(tables, 'band_records.tsv'), sep='\t')
    scores = pd.read_csv(os.path.join(tables, 'band_scores.tsv'), sep='\t')
    band = pd.read_csv(os.path.join(tables, 'band_summary.tsv'), sep='\t')
    previous_tables = os.path.join(folder, 'comparison', names['previous'], 'tables')
    scores_previous = band_previous = None
    if os.path.exists(os.path.join(previous_tables, 'band_scores.tsv')):
        scores_previous = pd.read_csv(os.path.join(previous_tables, 'band_scores.tsv'), sep='\t')
        band_previous = pd.read_csv(os.path.join(previous_tables, 'band_summary.tsv'), sep='\t')

    def other_sanity(name):
        path = os.path.join('experiments', name, 'sanity_report', 'summary.tsv')
        return pd.read_csv(path, sep='\t') if os.path.exists(path) else None

    rows, results_by_id, previous_by_id, records_by_id = [], {}, {}, {}
    for s in summary.itertuples():
        result = calibration(args.results, s.rgi_id)
        if result is None:
            continue
        previous = calibration(args.previous, s.rgi_id)
        recs = records[records.rgi_id == s.rgi_id]
        recs = recs if len(recs) else None
        row = ti_row(s.rgi_id, s.glacier, result, previous, recs, scores, summary)
        if scores_previous is not None:
            row['gw_rmse_previous'] = score(scores_previous, s.rgi_id, 'glacier-wide annual')
            row['profile_rmse_previous'] = score(scores_previous, s.rgi_id, 'mean profile')
        row['ela_gw_rmse'] = score(scores, s.rgi_id, 'glacier-wide annual', 'ELA')
        rows.append(row)
        results_by_id[s.rgi_id], previous_by_id[s.rgi_id] = result, previous
        records_by_id[s.rgi_id] = recs
    ti = pd.DataFrame(rows)
    ti['flags'] = [flags_of(r) for r in ti.to_dict('records')]
    ti.drop(columns='flags').to_csv(os.path.join(out, 'ti_summary.tsv'), sep='\t', index=False,
                                    float_format='%.4g')
    write_markdown(ti, out, args.results)

    # the topic pages of sanity_report.py, as its write_pdf builds them
    summary = summary[summary.rgi_id.isin(ti.rgi_id)].reset_index(drop=True)
    if 'ela_ref_capped' not in summary:
        from reference_profiles import ela_censored_share
        share = ela_censored_share(list(summary.rgi_id))
        summary = summary.assign(ela_ref_capped=[share[r] for r in summary.rgi_id])
    profiles = sr.smb_profiles(summary, args.results)
    summary = summary.assign(smb_rmse=[profiles[r]['rmse'] if r in profiles else np.nan
                                       for r in summary.rgi_id])
    pages = [('Overview', overview_page(ti, summary, band, args.results)),
             ('Comparison', comparison_page(ti, band, band_previous, summary,
                                            other_sanity(names['previous']),
                                            other_sanity(names['ela']), names)),
             ('TI parameters', parameter_page(ti)),
             ('ES-MDA', esmda_page(ti, results_by_id)),
             ('Balance per year', annual_page(ti, records_by_id)),
             ('Seasons', season_page(ti, records_by_id)),
             ('Bands', band_page(ti, records)),
             ('TI summary table', table_page(ti)),
             ('Thickness', sr.thickness_page(summary, data))]
    if not sr.no_flow(summary).all():
        pages.append(('Velocity', sr.velocity_page(summary, data)))
    pages += [('Elevation change', sr.dhdt_page(summary, data)),
              ('SMB profiles', sr.profile_page(summary, profiles))]
    reference = sr.reference_page(summary, folder)
    if reference is not None:
        pages.append(('ELA and gradients of the TI profile', reference))
    pages.append(('Sanity summary table', sr.table_page(summary)))
    colors, _, _ = sr.glacier_style(ti)
    total = len(pages) + 2 * len(ti)
    path = os.path.join(out, f'ti_report{suffix}.pdf')

    def finish(pdf, fig, topic, number, **kwargs):
        fig.text(0.035, 0.993, f'FROST TI calibration report  ·  {names["run"]}', fontsize=6.5,
                 va='top', color=sr.muted())
        fig.text(0.965, 0.993, topic, fontsize=6.5, va='top', ha='right', color=sr.muted())
        fig.text(0.5, 0.004, f'{number} / {total}', fontsize=7, ha='center', va='bottom',
                 color=sr.muted())
        pdf.savefig(fig, **kwargs)
        if args.png:
            os.makedirs(os.path.join(out, 'pages'), exist_ok=True)
            fig.savefig(os.path.join(out, 'pages', f'{number:02d}{suffix}.png'), dpi=args.png)
        plt.close(fig)

    with PdfPages(path) as pdf:
        number = 0
        for topic, fig in pages:
            number += 1
            finish(pdf, fig, topic, number)
        for i, row in enumerate(ti.itertuples()):
            title = row.glacier.replace('_', ' ')
            png = os.path.join(sanity_dir, row.glacier, f'glacier{suffix}.png')
            fig = plt.figure(figsize=sr.A4)
            ax = fig.add_axes([0.035, 0.02, 0.93, 0.955])
            if os.path.exists(png):
                ax.imshow(plt.imread(png))
            ax.axis('off')
            number += 1
            finish(pdf, fig, f'{title}: geometry, speed, dh/dt', number, dpi=sr.DPI)
            fig = glacier_page(row, results_by_id[row.rgi_id], previous_by_id[row.rgi_id],
                               records_by_id[row.rgi_id], colors[i])
            target = os.path.join(out, row.glacier)
            os.makedirs(target, exist_ok=True)
            fig.savefig(os.path.join(target, f'ti_glacier{suffix}.png'), dpi=110)
            number += 1
            finish(pdf, fig, f'{title}: TI', number)
    print(f'PDF: {path}')


if __name__ == '__main__':
    main()
