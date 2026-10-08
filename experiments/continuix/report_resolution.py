#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
ContinuIX report of 8 Oct 2026: the submission summary (two pages from
make_summary.py, from the check report of GROUP_FAU2) and a section on the
model grid with the tests of 8 Oct: why FROST runs on a 50 m grid (G01
100 m, S02 at its 100 m data grid). Reads the band scores
(band_score.json, lam_test.py --score), the inversion test (stats.json),
the posterior metrics of the G01 reruns and the data grids of the
ContinuIX files. Light, runs on the login node.

Run from the repository root:
    python experiments/continuix/report_resolution.py
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from netCDF4 import Dataset

R = os.path.join('data', 'results', 'continuix')
OUT = os.path.join(R, 'summary', 'ContinuIX_FROST_report_2026-10-08.pdf')
TOTAL = 4
GLACIERS = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'S01', 'S02']

# reference palette (dataviz skill, categorical slots 1-4, fixed order)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e3e2dc'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SURFACE, BAND = '#fcfcfb', '#f3f2ee'
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': INK,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE}
plt.rcParams.update(STYLE)
A4 = (8.27, 11.69)


def score(path):
    p = os.path.join(R, path, 'band_score.json')
    return json.load(open(p)) if os.path.exists(p) else None


def metrics(path):
    d = os.path.join(R, path)
    try:
        cal = json.load(open(os.path.join(d, 'calibration_results.json')))
        m = json.load(open(sorted(glob.glob(os.path.join(d, 'Monitor',
                                                         'metrics_*.json')))[-1]))
        meta = json.load(open(os.path.join(d, 'continuix.json')))
    except (OSError, IndexError):
        return None
    post = cal['diagnostics'][-1]
    return dict(grid=meta['resolution_model'], chi2=post['misfit'],
                ela=post['parameter_mean'][0], smb=m['modelled_surface_mass_balance'],
                obs=m['observed_elevation_change'], model=m['modelled_elevation_change'])


def data_grid(exp, glacier):
    for p in [os.path.join('data', 'raw', 'continuix', exp, f'{exp}_{glacier}.nc'),
              os.path.join('data', 'raw', 'continuix', exp, 'optional',
                           f'{exp}_{glacier}.nc')]:
        if os.path.exists(p):
            with Dataset(p) as nc:
                x = nc['x'][:2]
            return abs(float(x[1] - x[0]))
    return None


def table(ax, rows, header, widths, title=None, fontsize=7, highlight=()):
    """Plain table: bold header, light banding, no vertical rules."""
    ax.axis('off')
    if title:
        ax.set_title(title)
    t = ax.table(cellText=rows, colLabels=header, colWidths=widths,
                 loc='upper left', cellLoc='left', colLoc='left')
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = 'B'
        cell.set_text_props(color=INK if r == 0 else INK2)
        if r == 0:
            cell.set_text_props(weight='bold', color=INK)
        elif r in highlight:
            cell.set_facecolor('#e6eef9')
        elif r % 2 == 0:
            cell.set_facecolor(BAND)
        else:
            cell.set_facecolor(SURFACE)
    return t


def text(fig, x, y, s, size=8, weight='normal', color=INK, width=None):
    import textwrap
    if width:
        s = '\n'.join(textwrap.fill(par, width) for par in s.split('\n'))
    fig.text(x, y, s, fontsize=size, weight=weight, color=color, va='top',
             linespacing=1.35)


def fmt(v, f='{:.2f}'):
    return '–' if v is None else f.format(v)


# ---------------------------------------------------------------- data ----
grids = {g: [data_grid(e, g) for e in ('EXP01', 'EXP14', 'EXP15')] for g in GLACIERS}

# finer vs. coarser grid, EXP01, lam 1e11: (label, path)
comparisons = {
    'G01': [('100 m', 'EXP01/G01'), ('50 m', 'res50/EXP01/G01')],
    'G02': [('50 m', 'EXP01/G02'), ('25 m', 'res25/EXP01/G02')],
    'G03': [('50 m', 'EXP01/G03'), ('25 m', 'res25/EXP01/G03'),
            ('20 m (data)', 'res_native/EXP01/G03')],
    'G04': [('50 m', 'EXP01/G04'), ('25 m', 'res25/EXP01/G04')],
    'G05': [('50 m', 'EXP01/G05'), ('25 m', 'res25/EXP01/G05')],
    'G06': [('50 m', 'EXP01/G06'), ('25 m', 'res25/EXP01/G06'),
            ('10 m (data)', 'res_native/EXP01/G06')],
    'S01': [('25 m', 'backup_grid_1008/EXP01/S01'), ('50 m', 'res50/EXP01/S01')],
}
scores = {g: [(lab, score(p)) for lab, p in runs] for g, runs in comparisons.items()}

LAMS = [3e9, 1e10, 3e10, 1e11]


def lam_score(g, res, lam):
    if lam == 1e11:
        return score(f'EXP01/{g}' if res == 50 else f'res25/EXP01/{g}')
    return score(f"lam_test/r{res}_lam{lam:.0e}".replace('+', '') + f'/EXP01/{g}')


inv = {}
for p in glob.glob(os.path.join(R, 'inversion_test', '*', '*', 'stats.json')):
    s = json.load(open(p))
    inv[(s['glacier'], s['variant'])] = s

# ---------------------------------------------------------------- pages ---
page = [0]


def save(pdf, fig, footer=True):
    """Page into the PDF and as PNG next to it (for a quick look)."""
    page[0] += 1
    if footer:
        fig.text(0.07, 0.025, 'GROUP_FAU2 · method01 · section 2: model grid · '
                 'details: experiments/continuix/STATUS.md', fontsize=6.5, color=MUTED)
        fig.text(0.93, 0.025, f'{page[0]}/{TOTAL}', fontsize=6.5, color=MUTED,
                 ha='right')
    pdf.savefig(fig)
    fig.savefig(OUT.replace('.pdf', f'_p{page[0]}.png'), dpi=90)
    plt.close(fig)


with PdfPages(OUT) as pdf:
    # ---- section 1: submission summary (make_summary.py)
    sys.path.insert(0, os.path.join('experiments', 'continuix'))
    import make_summary
    make_summary.load()
    for draw in (make_summary.page1, make_summary.page2):
        save(pdf, draw(total=TOTAL), footer=False)
    plt.rcParams.update(STYLE)

    # ---- page 3: the emulator, the decision and the grids
    fig = plt.figure(figsize=A4)
    text(fig, 0.07, 0.965, 'Section 2: model grid of FROST', 15, 'bold')
    text(fig, 0.07, 0.94, 'Why FROST computes on 50 m (G01: 100 m), with the tests of '
         '8 October 2026', 9, color=INK2)
    # key statement
    fig.patches.append(plt.Rectangle((0.07, 0.8), 0.86, 0.11, transform=fig.transFigure,
                                     facecolor='#e6eef9', edgecolor='none'))
    text(fig, 0.09, 0.897, 'The ice-flow emulator was trained on 50-250 m grids', 12.5,
         'bold', color='#184f95')
    text(fig, 0.09, 0.872,
         'FROST computes the ice flow with IGM\'s pretrained emulator dahunet_mini (CNN, '
         'six 3x3 convolutions, no pooling), fine-tuned on each glacier at the start of '
         'the inversion and then frozen. It was trained on grids of 50-250 m (S. Rosier, '
         'IGM, pers. comm.), so 50 m is the finest grid it knows. On finer grids it '
         'extrapolates, and its fixed window of ±7 cells shrinks to ±175 m at 25 m and '
         '±70 m at 10 m, less than the several ice thicknesses over which longitudinal '
         'stresses couple the flow. This is why every finer grid we tested performed '
         'worse.', 8.2, width=112)
    text(fig, 0.07, 0.775, 'Decision', 10, 'bold')
    text(fig, 0.07, 0.757,
         'Model grid 50 m for all glaciers and experiments, G01 (the 832 km² ice cap) '
         '100 m; where the ContinuIX grid is coarser it is kept (S02 100 m; EXP14: G02 '
         '75 m, G03 60 m, S02 300 m; EXP15: 100 m). The perturbed inputs are used exactly '
         'as provided, so the EXP14/EXP15 results stay comparable with the other methods. '
         'Changed today: S01 from 25 m to 50 m.', 8, width=118)
    text(fig, 0.07, 0.69, 'What the tests showed (EXP01, details on the next page)', 10, 'bold')
    text(fig, 0.07, 0.672, '\n'.join([
        '• Finer than 50 m is worse or at best equal: 25 m on G02-G06 (G03 band RMS 1.30 → '
        '2.71 m/yr, G05 1.55 → 1.80, G04 0.20 → 0.30), the data grids of G03 (20 m) and '
        'G06 (10 m: inversion failed, velocity r -0.08); G04 (2 m) and G05 (10 m) do not '
        'fit into GPU memory.',
        '• Tuning does not rescue 25 m: a weaker sliding regularisation fits the '
        'velocities better but the dh/dt worse; retraining the emulator is worse '
        'everywhere.',
        '• S01 fits better at 50 m than at 25 m (0.64 → 0.50) with the same recovery of '
        'its true SMB. G01 at 50 m fits worse than at 100 m (0.41 → 0.57) and its '
        'thickness-noise and GLOB calibrations take 4 h to more than a day, so it stays at '
        '100 m.']), 8, width=120)
    ax = fig.add_axes([0.07, 0.25, 0.86, 0.29])
    rows = []
    for g in GLACIERS:
        d01, d14, d15 = grids[g]
        base = 100 if g in ('S02', 'G01') else 50
        m = [max(base, d) for d in (d01, d14, d15)]
        mark = lambda a, b: 'coarser' if b > a else 'same grid'
        rows.append([g, f'{d01:g} / {d14:g} / {d15:g} m', f'{m[0]:g} m',
                     f'{m[1]:g} m ({mark(m[0], m[1])})', f'{m[2]:g} m ({mark(m[0], m[2])})'])
    table(ax, rows, ['Glacier', 'Data grid EXP01 / 14 / 15', 'Model grid EXP01',
                     'Model grid EXP14', 'Model grid EXP15'],
          [0.1, 0.25, 0.17, 0.24, 0.24],
          'Model grid = coarser of the base grid (50 m; G01 100 m) and the data grid')
    text(fig, 0.07, 0.345,
         'Mandatory glaciers of EXP14/15: G01, G05, S01, S02. Where EXP14/15 keep the EXP01 '
         'model grid, they test input detail below FROST\'s grid: the glacier-mean SMB '
         'changes by at most 0.08 m/yr there (G06; G01, G04, G05, S01 <= 0.015). All '
         'grids used for the submission are inside the emulator\'s training range except '
         'EXP14 S02 (300 m, slightly above). EXP02 (raw data) runs on the EXP01 grids.',
         7, color=INK2, width=135)
    save(pdf, fig)

    # ---- page 4: evidence
    fig = plt.figure(figsize=A4)
    text(fig, 0.07, 0.965, 'Evidence: finer grids and the sliding regularisation', 12, 'bold')
    ax = fig.add_axes([0.2, 0.66, 0.72, 0.26])
    order = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'S01']
    for i, g in enumerate(order):
        y = len(order) - 1 - i
        ref = dict(scores[g])['50 m']['band_rms']
        vals = [(lab, s['band_rms'] / ref, s) for lab, s in scores[g] if s]
        xs = [v for _, v, _ in vals]
        ax.plot([min(xs + [1]), max(xs + [1])], [y, y], color=GRID, lw=2, zorder=1)
        for lab, v, s in vals:
            sub = lab.startswith('50')
            ax.scatter(v, y, s=60 if sub else 40, color=BLUE if sub else ORANGE,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if not sub:
                below = s['velocity']['r'] < 0.2
                note = lab + (' (inversion failed)' if below else '')
                ax.annotate(note, (v, y), xytext=(0, -11 if below else 7),
                            textcoords='offset points', ha='center', fontsize=6.5,
                            color=INK2)
    ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set_xscale('log')
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3], ['0.5', '0.75', '1', '1.5', '2', '3'])
    ax.set_xlabel('Band RMS relative to the 50 m run (< 1: fits the dh/dt bands better)')
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_title('EXP01: each tested grid relative to 50 m (lam 1e11)')
    ax.scatter([], [], s=60, color=BLUE, label='50 m')
    ax.scatter([], [], s=40, color=ORANGE, label='other grid')
    ax.legend(loc='lower right', fontsize=7)

    rows, hl = [], []
    for g in order:
        for lab, s in scores[g]:
            if not s:
                continue
            rows.append([g, lab, fmt(s['band_rms']), fmt(s['chi2']),
                         fmt(s['obs_mean'], '{:+.2f}'), fmt(s['model_mean'], '{:+.2f}'),
                         fmt(s['velocity']['r'])])
            if (g == 'G01' and lab == '100 m') or (g != 'G01' and lab.startswith('50')):
                hl.append(len(rows))
    table(fig.add_axes([0.07, 0.17, 0.86, 0.42]), rows,
          ['Glacier', 'Grid', 'Band RMS', 'chi2/n', 'Obs. dh/dt', 'Model dh/dt',
           'Velocity r'], [0.1, 0.16, 0.13, 0.11, 0.15, 0.16, 0.13],
          'EXP01 results per grid (highlighted: submitted grid), dh/dt in m/yr',
          highlight=hl)
    text(fig, 0.07, 0.165, '\n'.join([
        'Band RMS: RMS over 50 m elevation bands of the posterior ensemble-mean dh/dt '
        'against the observed band means. S01 truth: ELA 2350 m, gradients 10 m/yr per km.',
        'Sliding regularisation (G04-G06, lam 3e9 to 1e11, 25 and 50 m): lower lam raises '
        'the velocity r but not the dh/dt fit (G06 band RMS 0.68 → 1.51 m/yr at 25 m); '
        'lam 1e11 kept.',
        'Faster pipeline (same results): band covariance by FFT (G01 17 min → 0.8 s), '
        'forward runs in persistent in-process workers (calibration 2.6-3.8x faster).']),
        7, color=INK2, width=135)
    save(pdf, fig)

print(OUT)
