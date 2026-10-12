#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
ContinuIX report: the submission summary (two pages from make_summary.py,
from the check report of GROUP_FAU2), a page of EXP01 maps (observed dh/dt,
submitted SMB, mismatch SMB - FDIV - dh/dt) and a page on the model grid
(why FROST runs on 50 m, G01 and S02 on 100 m, with the tests of 8 Oct).

The maps come from the full-resolution result files, so they are first
reduced in a job (summary/maps_EXP01.npz); drawing the report is light and
runs on the login node. Run from the repository root:
    sbatch experiments/continuix/report_maps.sh     # once, after packaging
    python experiments/continuix/report_resolution.py
"""

import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

R = os.path.join('data', 'results', 'continuix')
OUT = os.path.join(R, 'summary', 'ContinuIX_FROST_report_2026-10-12.pdf')
MAPS = os.path.join(R, 'summary', 'maps_EXP01.npz')
TOTAL = 4
GLACIERS = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'S01', 'S02']
MAP_CELLS = 160  # longest side of a reduced map

# reference palette (dataviz skill, categorical slots 1-4, fixed order)
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e3e2dc'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
SURFACE, BAND = '#fcfcfb', '#f3f2ee'
DIVERGING = LinearSegmentedColormap.from_list(
    'div', ['#a83c12', ORANGE, '#f1f0ea', BLUE, '#174e94'])
STYLE = {
    'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': INK,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False,
    'axes.titlesize': 9, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE}
A4 = (8.27, 11.69)


# ------------------------------------------------------ map extraction ----
def coarsen(a, k):
    """Block mean ignoring NaN; k x k blocks, edges padded with NaN."""
    if k == 1:
        return a
    ny, nx = (-(-n // k) * k for n in a.shape)
    p = np.full((ny, nx), np.nan, dtype=np.float32)
    p[:a.shape[0], :a.shape[1]] = a
    b = p.reshape(ny // k, k, nx // k, k)
    with np.errstate(invalid='ignore'):
        n = np.sum(np.isfinite(b), axis=(1, 3))
        return np.where(n > 0, np.nansum(b, axis=(1, 3)) / np.maximum(n, 1), np.nan)


def extract_maps():
    """EXP01 maps of every glacier at <= MAP_CELLS cells per side (job)."""
    from netCDF4 import Dataset
    out = {}
    for g in GLACIERS:
        with Dataset(os.path.join('data', 'raw', 'continuix', 'EXP01',
                                  f'EXP01_{g}.nc')) as nc:
            x, y = nc['x'][:].data, nc['y'][:].data
            ice = np.asarray(nc['ICEMASK'][:].filled(0)) > 0.5
            dem = np.asarray(nc['DEM'][:].filled(np.nan), dtype=np.float32)
            dhdt = np.asarray(nc['DHDT'][:].filled(np.nan), dtype=np.float32)
        with Dataset(os.path.join(R, 'GROUP_FAU2', 'EXP01',
                                  f'EXP01_{g}_method01.nc')) as nc:
            smb = np.asarray(nc['SMB'][:].filled(np.nan), dtype=np.float32)
            fdiv = np.asarray(nc['FDIV'][:].filled(np.nan), dtype=np.float32)
        dhdt[dhdt == 0] = np.nan  # ContinuIX fills gaps with 0
        for a in (dhdt, smb, fdiv):
            a[~ice] = np.nan
        dem[~ice] = np.nan
        k = max(1, int(np.ceil(max(ice.shape) / MAP_CELLS)))
        maps = dict(dhdt=dhdt, smb=smb, mismatch=smb - fdiv - dhdt, dem=dem,
                    ice=ice.astype(np.float32))
        maps = {n: coarsen(a, k) for n, a in maps.items()}
        if y[1] < y[0]:  # store with y increasing
            maps = {n: a[::-1] for n, a in maps.items()}
        dx = abs(float(x[1] - x[0]))
        extent = [0, len(x) * dx, 0, len(y) * dx]
        for n, a in maps.items():
            out[f'{g}_{n}'] = a
        out[f'{g}_extent'] = np.array(extent)
        print(g, ice.shape, '->', maps['ice'].shape, flush=True)
    np.savez_compressed(MAPS, **out)
    print(MAPS)


# -------------------------------------------------------------- drawing ----
def score(path):
    p = os.path.join(R, path, 'band_score.json')
    return json.load(open(p)) if os.path.exists(p) else None


def model_grid(g):
    return json.load(open(os.path.join(R, 'EXP01', g, 'continuix.json')))['resolution_model']


def text(fig, x, y, s, size=8, weight='normal', color=INK, width=None, spacing=1.4):
    if width:
        s = '\n'.join(textwrap.fill(par, width, subsequent_indent='   '
                                    if par.startswith('•') else '')
                      for par in s.split('\n'))
    fig.text(x, y, s, fontsize=size, weight=weight, color=color, va='top',
             linespacing=spacing)


def scale_bar(ax, extent):
    """Round-number km bar of about a quarter of the map width."""
    w = extent[1] - extent[0]
    nice = [0.25, 0.5, 1, 2, 5, 10, 20]
    km = max(n for n in nice if n * 1000 <= w / 3)
    x0, y0 = extent[0] + 0.04 * w, extent[2] + 0.06 * (extent[3] - extent[2])
    ax.plot([x0, x0 + km * 1000], [y0, y0], color=INK, lw=1.6, solid_capstyle='butt')
    ax.text(x0 + km * 1000 / 2, y0, f'{km:g} km', ha='center', va='bottom',
            fontsize=5.8, color=INK)


def page_maps(maps):
    fig = plt.figure(figsize=A4)
    text(fig, 0.07, 0.968, 'EXP01 maps: where the SMB profile fits and where it does not',
         14, 'bold')
    text(fig, 0.07, 0.944, 'Modelled dh/dt = submitted SMB − FDIV (mass conservation). '
         'Black line: SMB = 0 (calibrated ELA); grey: 100 m surface contours.',
         8, color=INK2, width=125)
    cols = [('Observed dh/dt', 'dhdt', 6), ('Submitted SMB', 'smb', 6),
            ('Modelled − observed dh/dt', 'mismatch', 2)]
    x0s, w = [0.135, 0.42, 0.705], 0.27
    top, bottom = 0.905, 0.125
    h = (top - bottom) / len(GLACIERS)
    for x0, (title, _, _) in zip(x0s, cols):
        fig.text(x0, top + 0.008, title, fontsize=9, weight='bold')
    images = {}
    for i, g in enumerate(GLACIERS):
        ext = maps[f'{g}_extent']
        y0 = top - (i + 1) * h + 0.006
        fig.text(0.07, y0 + h / 2 + 0.008, g, fontsize=11, weight='bold', va='center')
        fig.text(0.07, y0 + h / 2 - 0.012, f'{model_grid(g):g} m grid', fontsize=6.8,
                 color=INK2, va='center')
        dem = maps[f'{g}_dem']
        levels = np.arange(np.floor(np.nanmin(dem) / 100) * 100, np.nanmax(dem), 100)
        for j, (x0, (_, name, lim)) in enumerate(zip(x0s, cols)):
            ax = fig.add_axes([x0, y0, w, h - 0.012])
            ax.set_facecolor(SURFACE)
            images[name] = ax.imshow(maps[f'{g}_{name}'], origin='lower', extent=ext,
                                     cmap=DIVERGING, vmin=-lim, vmax=lim,
                                     interpolation='nearest')
            ax.contour(maps[f'{g}_ice'], levels=[0.5], origin='lower', extent=ext,
                       colors=INK2, linewidths=0.5)
            ax.contour(dem, levels=levels, origin='lower', extent=ext,
                       colors=MUTED, linewidths=0.3, alpha=0.8)
            if name == 'smb':
                ax.contour(maps[f'{g}_smb'], levels=[0], origin='lower', extent=ext,
                           colors=INK, linewidths=0.9)
            if j == 0:
                scale_bar(ax, ext)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for side in ax.spines.values():
                side.set_visible(False)
    for (x0, x1), name, label in [((x0s[0], x0s[1] + w), 'smb', 'dh/dt and SMB (m i.e. yr⁻¹)'),
                                  ((x0s[2], x0s[2] + w), 'mismatch', 'modelled − observed dh/dt (m i.e. yr⁻¹)')]:
        cax = fig.add_axes([x0, 0.098, x1 - x0, 0.008])
        cb = fig.colorbar(images[name], cax=cax, orientation='horizontal', extend='both')
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.8, length=2)
        cb.set_label(label, fontsize=7.2, color=INK2)
    text(fig, 0.07, 0.06, 'Right column: orange, the model thins faster than observed; '
         'blue, slower. The speckle is cell-scale noise of the flux divergence, which '
         'cancels in the 50 m elevation bands the calibration fits; the systematic misfit '
         'is the upper basin of G03.',
         7, color=INK2, width=140)
    return fig


def page_grid():
    fig = plt.figure(figsize=A4)
    text(fig, 0.07, 0.965, 'Model grid: 50 m (G01 and S02: 100 m)', 14, 'bold')
    fig.patches.append(plt.Rectangle((0.07, 0.872), 0.86, 0.058, transform=fig.transFigure,
                                     facecolor='#e6eef9', edgecolor='none'))
    text(fig, 0.09, 0.917,
         'IGM\'s ice-flow emulator (dahunet_mini) was pretrained on 50–250 m grids '
         '(S. Rosier, pers. comm.). 50 m is the finest grid in its training range, and '
         'every finer grid we tested did worse.', 9.5, color='#184f95', width=100)

    text(fig, 0.07, 0.85, '\n'.join([
        '• Finer grids did not help. At 25 m (G02–G06) and on the provided data grids '
        '(G03 20 m, G06 10 m), the dh/dt fit was the same or worse, except on G06, where '
        'it improved slightly while the velocity fit collapsed (r 0.62 → 0.36 at 25 m; '
        'at 10 m the sliding inversion failed). G04 (2 m) and G05 (10 m) did not fit '
        'into GPU memory.',
        '• Tuning did not rescue 25 m. A weaker sliding regularisation fitted the '
        'velocities better but the dh/dt worse; retraining the emulator was worse at '
        '25 and 50 m. λ = 1e11 stays.',
        '• S01 moved from 25 to 50 m: better fit (band RMS 0.64 → 0.50 m/yr), same '
        'recovery of its true SMB.',
        '• G01 stays at 100 m: at 50 m it fits worse (0.41 → 0.57 m/yr), and its '
        'thickness-noise and GLOB calibrations take 4 h to more than a day. S02 runs on '
        'its 100 m data grid.',
        '• For EXP14/15 the model grid follows the data where they are coarser: EXP15 '
        'changes the grid for all glaciers except G01 and S02, EXP14 only for G02, G03 '
        'and S02. Elsewhere EXP14 tests input detail below the grid (≤ 0.10 m/yr).']),
        8.8, width=108, spacing=1.5)

    order = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'S01']
    comparisons = {
        'G01': [('100 m', 'EXP01/G01'), ('50 m', 'res50/EXP01/G01')],
        'G02': [('50 m', 'EXP01/G02'), ('25 m', 'res25/EXP01/G02')],
        'G03': [('50 m', 'EXP01/G03'), ('25 m', 'res25/EXP01/G03'),
                ('20 m', 'res_native/EXP01/G03')],
        'G04': [('50 m', 'EXP01/G04'), ('25 m', 'res25/EXP01/G04')],
        'G05': [('50 m', 'EXP01/G05'), ('25 m', 'res25/EXP01/G05')],
        'G06': [('50 m', 'EXP01/G06'), ('25 m', 'res25/EXP01/G06'),
                ('10 m', 'res_native/EXP01/G06')],
        'S01': [('50 m', 'EXP01/S01'), ('25 m', 'backup_grid_1008/EXP01/S01')],
    }
    ax = fig.add_axes([0.16, 0.13, 0.76, 0.5])
    for i, g in enumerate(order):
        y = len(order) - 1 - i
        runs = [(lab, score(p)) for lab, p in comparisons[g]]
        ref = runs[0][1]['band_rms']
        vals = [(lab, s['band_rms'] / ref, s) for lab, s in runs if s]
        xs = [v for _, v, _ in vals]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=2, zorder=1)
        for k, (lab, v, s) in enumerate(vals):
            sub = k == 0
            ax.scatter(v, y, s=70 if sub else 45, color=BLUE if sub else ORANGE,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            failed = s['velocity']['r'] < 0.2
            note = lab + ('\n(inversion failed)' if failed else '')
            ax.annotate(note, (v, y), xytext=(0, 8 if sub else -9),
                        textcoords='offset points', ha='center',
                        va='bottom' if sub else 'top', fontsize=7,
                        color=BLUE if sub else INK2)
    ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(range(len(order)), order[::-1], fontsize=9)
    ax.set_xscale('log')
    ax.set_xticks([0.75, 1, 1.5, 2, 3], ['0.75', '1', '1.5', '2', '3'])
    ax.set_xlim(0.62, 3)
    ax.minorticks_off()
    ax.set_xlabel('band RMS relative to the submitted grid  (> 1: worse fit of the '
                  'dh/dt elevation profile)', fontsize=8)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_title('EXP01: every tested grid against the submitted one (blue)', fontsize=10)
    text(fig, 0.07, 0.075, 'Band RMS: RMS over 50 m elevation bands of the posterior '
         'ensemble-mean dh/dt against the observed band means; λ = 1e11 in every run. '
         'Full numbers: experiments/continuix/STATUS.md.', 7, color=INK2, width=140)
    return fig


def save(pdf, fig, footer=True):
    """Page into the PDF and as PNG next to it (for a quick look)."""
    page[0] += 1
    if footer:
        fig.text(0.07, 0.025, 'GROUP_FAU2 · method01 · details: '
                 'experiments/continuix/STATUS.md and README_GROUP_FAU2_method01.txt',
                 fontsize=6.5, color=MUTED)
        fig.text(0.93, 0.025, f'{page[0]}/{TOTAL}', fontsize=6.5, color=MUTED,
                 ha='right')
    pdf.savefig(fig)
    fig.savefig(OUT.replace('.pdf', f'_p{page[0]}.png'), dpi=90)
    plt.close(fig)


page = [0]

if __name__ == '__main__':
    if '--extract-maps' in sys.argv:
        extract_maps()
        sys.exit()
    if not os.path.exists(MAPS):
        sys.exit(f'{MAPS} missing: sbatch experiments/continuix/report_maps.sh first')
    maps = dict(np.load(MAPS))
    sys.path.insert(0, os.path.join('experiments', 'continuix'))
    import make_summary
    make_summary.load()
    with PdfPages(OUT) as pdf:
        for draw in (make_summary.page1, make_summary.page2):
            save(pdf, draw(total=TOTAL), footer=False)
        plt.rcParams.update(STYLE)
        save(pdf, page_maps(maps))
        save(pdf, page_grid())
    print(OUT)
