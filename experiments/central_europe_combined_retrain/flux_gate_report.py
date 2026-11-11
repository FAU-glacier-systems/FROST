#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
PDF report on the flux-gate check of Aletsch and Rhone (flux_gate.py).

Reads tables/flux_gate.tsv (model and observed fluxes, written by
flux_gate.py from the ensembles) and only the inputs of the two glaciers
(no ensembles, fine on the login node). Adds the required inflow of the
ITS_LIVE years: Hugonnet dh/dt 2015-2020 - GLAMOS Ba 2015-2019
(tables/flux_gate_period.tsv).

Run from the repository root:
    python experiments/central_europe_combined_retrain/flux_gate_report.py
Writes flux_gate_report.pdf (light theme, Inter).
"""

import glob
import io
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from netCDF4 import Dataset
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, os.getcwd())
from frost.visualization import plot_style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join('data', 'results', 'central_europe_retrain')
HUGONNET = os.path.join('data', 'raw', 'hugonnet', '11_rgi60_{}', 'dhdt', '*_dhdt.tif')
GLAMOS = os.path.join('data', 'raw', 'glamos', 'massbalance_observation_elevationbins.csv')
GLACIERS = [('RGI2000-v7.0-G-11-02596', 'Grosser Aletschgletscher', 'Aletsch', (2500, 2800)),
            ('RGI2000-v7.0-G-11-01706', 'Rhonegletscher', 'Rhone', (2800,))]
RHO = 0.91
TITLE = 'FROST flux-gate check: Aletsch and Rhone'
PAGE = (8.27, 11.69)
L, R = 0.09, 0.91        # side margins (fraction of the page width)
BODY = 9.5               # body text size (pt)


# ---------------------------------------------------------------- data

def read(path, name):
    with Dataset(path) as d:
        v = d[name][:]
        return np.array(v.filled(np.nan) if np.ma.isMaskedArray(v) else v, dtype=float)


def glamos_bands(name, years):
    with open(GLAMOS, encoding='latin-1') as f:
        lines = [line for line in f if line.startswith(name + ';')]
    g = pd.read_csv(io.StringIO(''.join(lines)), sep=';', header=None,
                    names='name id ds dw de Bw Bs Ba area hmin hmax obs'.split())
    g['year'] = g.de.str[:4].astype(int)
    g = g[(g.year >= years[0]) & (g.year <= years[1])]
    return (pd.to_numeric(g.Ba) / 1000).groupby(pd.to_numeric(g.hmin)).mean()


def hugonnet(period, x, y, epsg):
    """Hugonnet dh/dt of `period` bilinearly on the model grid (y ascending)."""
    dx = abs(x[1] - x[0])
    out = np.full((len(y), len(x)), np.nan, np.float32)
    transform = from_origin(x.min() - dx / 2, y.max() + dx / 2, dx, dx)
    for path in sorted(glob.glob(HUGONNET.format(period))):
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            tmp = np.full(out.shape, np.nan, np.float32)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=epsg,
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
        out = np.where(np.isfinite(out), out, tmp)
    return out[::-1] if y[0] < y[-1] else out


def load(rgi_id, name):
    base = os.path.join(RUN, rgi_id, 'Preprocess')
    inp = os.path.join(base, 'data', 'input.nc')
    with Dataset(inp) as d:
        epsg = d.epsg
    x, y = read(inp, 'x'), read(inp, 'y')
    g = dict(x=x, y=y, dx=abs(x[1] - x[0]), s=read(inp, 'usurf_2000'),
             ice=read(inp, 'thk_2000') > 1, dhdt=read(inp, 'dhdt'),
             speed=np.hypot(read(inp, 'uvelsurfobs'), read(inp, 'vvelsurfobs')),
             thkobs=read(inp, 'thkobs'),
             thk_inv=read(os.path.join(base, 'outputs', 'output.nc'), 'thk_inversion'))
    g['dhdt_1520'] = hugonnet('2015-01-01_2020-01-01', x, y, epsg)
    band = np.floor(g['s'] / 100) * 100
    for key, dhdt, years in (('em_0119', g['dhdt'], (2001, 2019)),
                             ('em_1519', g['dhdt_1520'], (2015, 2019))):
        ba = glamos_bands(name, years)
        cell = np.full(g['s'].shape, np.nan)
        cell[g['ice']] = pd.Series(band[g['ice']]).map(ba).values
        g[key] = dhdt - cell / RHO
    return g


def required(g, key):
    area = g['dx'] ** 2
    gates = np.arange(np.ceil(np.nanmin(g['s'][g['ice']]) / 100) * 100 + 100,
                      np.nanmax(g['s'][g['ice']]), 100)
    return pd.Series([np.nansum(g[key][g['ice'] & (g['s'] < h)]) * area / 1e9 for h in gates],
                     index=gates.astype(int))


# ---------------------------------------------------------------- layout

def theme(dark):
    plt.rcdefaults()
    if dark:
        plt.rcParams.update(plot_style.DARK_THEME)
    else:
        plt.rcParams.update({k: v for k, v in plot_style.DARK_THEME.items()
                             if k.startswith(('axes.spines', 'xtick.major', 'ytick.major',
                                              'axes.grid', 'axes.axisbelow', 'grid.linestyle',
                                              'grid.linewidth', 'legend.frameon'))})
        plt.rcParams['grid.color'] = '#c8c8c8'
    # reports: Inter for text (most readable), JetBrains Mono for equations
    plot_style.use_font()
    plt.rcParams['font.monospace'] = ['JetBrains Mono', 'DejaVu Sans Mono']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 9


def colors(dark):
    c = dict(plot_style.COLORS)
    if not dark:
        c.update(fg='#111111', muted='#555555', blue='#1f6fb2', red='#c62f35',
                 yellow='#c99700', grey='#888888', green='#2a9a63')
    return c


def page(pdf, n, total, topic, c, draw):
    fig = plt.figure(figsize=PAGE)
    fig.text(L, 0.975, TITLE, fontsize=7.5, color=c['muted'], va='top')
    fig.text(R, 0.975, topic, fontsize=7.5, color=c['muted'], va='top', ha='right')
    fig.text(R, 0.02, f'{n} / {total}', fontsize=7.5, color=c['muted'], ha='right')
    fig.text(L, 0.02, 'experiments/central_europe_combined_retrain/flux_gate_report.py',
             fontsize=6, color=c['muted'])
    draw(fig)
    pdf.savefig(fig)
    if os.environ.get('PREVIEW_DIR'):   # PNG of each page, to look at without a PDF viewer
        fig.savefig(os.path.join(os.environ['PREVIEW_DIR'],
                                 f'page{n}_{plt.rcParams["axes.facecolor"] != "white"}.png'),
                    dpi=80)
    plt.close(fig)


def text_block(fig, y, blocks, c, width=100):
    """blocks: list of (kind, text); kind 'h' heading, 'p' paragraph, 'b' bullet."""
    line = 0.0158
    for kind, text in blocks:
        if kind == 'h':
            y -= 0.008
            fig.text(L, y, text, fontsize=14, weight='bold', color=c['fg'], va='top')
            y -= 0.026
            continue
        indent, prefix = (L + 0.022, '') if kind == 'b' else (L, '')
        lines = textwrap.wrap(text, width - (3 if kind == 'b' else 0))
        if kind == 'b':
            fig.text(L + 0.006, y, '-', fontsize=BODY, color=c['muted'], va='top')
        for i, ln in enumerate(lines):
            fig.text(indent, y - i * line, prefix + ln, fontsize=BODY, color=c['fg'], va='top',
                     family='monospace' if kind == 'm' else None)
        y -= len(lines) * line + 0.006
    if y < 0.04:
        print(f'WARNING: text runs into the footer (y = {y:.3f})')
    return y


# ---------------------------------------------------------------- pages

def subtitle(fig, text, c, y=0.922):
    for i, line in enumerate(textwrap.wrap(text, 108)):
        fig.text(L, y - 0.014 * i, line, fontsize=8.5, color=c['muted'], va='top')


def summary_page(fig, t, c):
    a = t['Aletsch'].set_index('gate_m')
    r = t['Rhone'].set_index('gate_m')
    fig.text(L, 0.945, 'Flux-gate check: is the ice flux into the tongue consistent '
             'with the data?', fontsize=13, weight='bold', color=c['fg'], va='top')
    fig.text(L, 0.922, 'Reference ELA run (central_europe_retrain, part of '
             'central_europe_combined_retrain), state of 2026-11-11', fontsize=8,
             color=c['muted'], va='top')

    def mid(df, h, key):
        return 0.5 * (df.loc[h, key + '_lo'] + df.loc[h, key + '_hi'])

    blocks = [
        ('h', 'Question'),
        ('p', 'The calibrated SMB profile of Aletsch is poor: tongue -7.5 m w.e./yr against '
              'GLAMOS -14, profile RMSE 3.3 m w.e./yr. The calibration fits Hugonnet dh/dt, so '
              'a tongue SMB that is too weak points to too little ice coming from above '
              '(dh/dt = SMB/0.91 + emergence). The flux-gate check asks how much ice must '
              'flow across an elevation contour to close the mass budget below it, and how '
              'much the model and the observations deliver.'),
        ('h', 'Answer'),
        ('b', f'Aletsch, gate 2500 m: the data need {a.loc[2500, "required"]:.3f} km3 ice/yr '
              f'(2001-2019), the model delivers {a.loc[2500, "model_2000_2019"]:.3f} '
              f'({a.loc[2500, "model_2000_2019"] / a.loc[2500, "required"]:.0%}). '
              f'ITS_LIVE speed x inverted thickness gives {mid(a, 2500, "obs_inverted"):.3f}, '
              f'with the thickness scaled to GlaThiDa {mid(a, 2500, "obs_glathida"):.3f}. '
              'So the model is consistent with the observed speed and the inverted '
              'thickness, but the observations themselves give only about half of the '
              'required flux.'),
        ('b', f'Same years as ITS_LIVE (Hugonnet 2015-2020, GLAMOS 2015-2019): the required '
              f'flux at 2500 m is {a.loc[2500, "required_1519"]:.3f}, the model '
              f'{a.loc[2500, "model_2014_2019"]:.3f}. Taking the period into account does not '
              'close the gap.'),
        ('b', f'Rhone, gate 2800 m: required {r.loc[2800, "required"]:.4f}, model '
              f'{r.loc[2800, "model_2000_2019"]:.4f}, ITS_LIVE x GlaThiDa-scaled thickness '
              f'{mid(r, 2800, "obs_glathida"):.4f}: closed by thicker ice.'),
        ('b', 'On both glaciers the inverted (IceBoost) thickness is too thin where the ice '
              'is deep: Aletsch max 400 m inverted vs 606 m GlaThiDa (grid cell), Rhone '
              '327 vs 450 m. Modelled / ITS_LIVE speed: Aletsch 0.4-0.7 at the 2100-2600 m gates, '
              '0.9-1.0 at 2700-2800 m; Rhone 0.8-1.0 at 2700-2800 m.'),
        ('b', 'The hypsometry of the model (81.8 km2 in 2000) agrees with GLAMOS (83.0 km2, '
              'each 100 m band within ~5 %): no outline problem.'),
        ('h', 'Assessment (2026-11-11)'),
        ('b', 'GlaThiDa probably misses the deepest ice (Konkordia): the GlaThiDa-scaled '
              'flux is a lower bound, the real thickness is larger still.'),
        ('b', 'ITS_LIVE 2014-2022 is probably slower than the 2001-2019 mean; this has to be '
              'taken into account (done here with the 2015-2019 budget, see pages 4-5).'),
        ('b', 'GLAMOS ablation is not thought to be too steep: the GLAMOS profile is kept as '
              'the reference.'),
        ('h', 'Consequence'),
        ('p', 'The missing flux is a thickness (and possibly sliding) problem of the model '
              'geometry, not an SMB-model problem. Because the calibration only sees dh/dt, '
              'it compensates the missing emergence with a too weak tongue SMB. Any fix must be '
              'generic (no glacier-specific tuning, no GlaThiDa in the fit).'),
    ]
    text_block(fig, 0.89, blocks, c)


def limits_page(fig, c):
    fig.text(L, 0.945, 'Limitations and next steps', fontsize=14, weight='bold',
             color=c['fg'], va='top')
    blocks = [
        ('h', 'Limitations'),
        ('b', 'GlaThiDa is not moved to the model date (survey year x local dh/dt); thinning '
              'since the surveys makes the GlaThiDa ratio a few % too high.'),
        ('b', 'Depth average 0.8-1.0 x surface speed; on Aletsch with much sliding close to 1.'),
        ('b', 'The 2015-2019 budget does not close glacier-wide (Aletsch +0.026 km3/yr at the '
              'top, ~0.3 m/yr: Hugonnet 5-year noise and GLAMOS period); this affects the upper '
              'gates, not the tongue (2500 m: 0.081 vs 0.083 for 2001-2019).'),
        ('b', 'Gates are contours of the 2000 surface on a 100 m (Aletsch) / 50 m (Rhone) grid; '
              'the 2700 m gate of Aletsch crosses the flat Konkordia basin and is noisy.'),
        ('b', 'Only two glaciers; the same check over all 17 flowing glaciers (lowest third) '
              'gives IGM emergence lower on 16 of 17 (central_europe_TI_flow_snow_lapse).'),
        ('h', 'Next steps (focus Aletsch, generic methods)'),
        ('b', 'See page 6: thickness free in the inversion (IceBoost as prior with an '
              'uncertainty that grows with depth), ITS_LIVE speed as observation in the '
              'calibration over its own years, a thickness/flux parameter in ES-MDA.'),
    ]
    text_block(fig, 0.91, blocks, c)


def method_page(fig, gl, c):
    fig.text(L, 0.945, 'Method', fontsize=14, weight='bold', color=c['fg'], va='top')
    blocks = [
        ('p', 'Mass conservation of the ice column: dh/dt = b/0.91 - div(q), with b the SMB '
              '(m w.e./yr), q = H u the depth-integrated flux. Integrated over the region R '
              'below a gate (all ice below an elevation contour of the 2000 surface), the '
              'divergence becomes the flux through the gate (the contour):'),
        ('m', '    Q_in(h) = sum over R of (dh/dt - b/0.91) dA      [km3 ice/yr]'),
        ('p', 'Three estimates of Q_in at gates every 100 m:'),
        ('b', 'required (black line): Hugonnet dh/dt minus GLAMOS Ba of the 100 m band '
              'of each cell, 2001-2019 (and 2015-2019 with the Hugonnet 2015-2020 tile, the '
              'ITS_LIVE years). What the data need.'),
        ('b', 'model (blue): - sum of IGM divflux over R, mean of the 32 posterior members, '
              '2000-2019 (solid) and 2014-2019 (dashed).'),
        ('b', 'observed (bands): ITS_LIVE 2014-2022 velocity vector x thickness on the cell '
              'faces of the contour (finite volume), x 0.8-1.0 for depth-averaged / surface '
              'speed, with the inverted thickness (red), IceBoost (grey), and the inverted '
              'thickness x the median GlaThiDa / inverted ratio of the cells within 100 m '
              'elevation of the gate (yellow; validation only, not used in FROST).'),
        ('p', 'If model = observed < required, model and observations agree with each other '
              'and the gap lies in the observed thickness or speed (or in GLAMOS). If model < '
              'observed, the model itself is too slow or too thin.'),
    ]
    text_block(fig, 0.91, blocks, c)
    grid = fig.add_gridspec(1, 2, left=L - 0.03, right=R + 0.03, top=0.5, bottom=0.11, wspace=0.04)
    for k, (short, g, gates) in enumerate(gl):
        ax = fig.add_subplot(grid[0, k])
        ext = [g['x'].min(), g['x'].max(), g['y'].min(), g['y'].max()]
        sp = np.where(g['ice'], g['speed'], np.nan)
        im = ax.imshow(sp, origin='lower', extent=ext, cmap='viridis',
                       norm=LogNorm(1, 300), interpolation='nearest')
        s = np.where(g['ice'], g['s'], np.nan)
        lv = np.arange(1800, 4100, 100)
        ax.contour(g['x'], g['y'], s, levels=lv, colors=c['fg'], linewidths=0.3, alpha=0.5)
        ax.contour(g['x'], g['y'], s, levels=list(gates), colors=c['red'], linewidths=2.2)
        ok = g['ice'] & np.isfinite(g['thkobs'])
        yy, xx = np.nonzero(ok)
        ax.scatter(g['x'][xx], g['y'][yy], s=0.6, color=c['fg'], lw=0, alpha=0.6)
        ax.set_aspect('equal')
        ax.set_anchor('N')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        cell = grid[0, k].get_position(fig)
        fig.text(cell.x0 + 0.03, 0.52, f'{short}: ITS_LIVE speed 2014-2022', fontsize=9,
                 weight='bold', color=c['fg'])
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label('m/yr', fontsize=8)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([1, 3, 10, 30, 100, 300])
        cb.set_ticklabels(['1', '3', '10', '30', '100', '300'])
    subtitle(fig, 'Thin grey lines: 2000 surface every 100 m (the gates). Red: the gates of the '
             'table on page 1 (Aletsch 2500 and 2800 m, Rhone 2800 m). Dots: grid cells with '
             'GlaThiDa thickness.', c, y=0.075)


def profile_page(fig, t, c):
    fig.text(L, 0.945, 'Ice flux into the area below each gate', fontsize=14,
             weight='bold', color=c['fg'], va='top')
    subtitle(fig, 'Positive: ice flows downwards across the contour. Where the line of the model '
             'is left of the required line, the model brings too little ice.', c)
    grid = fig.add_gridspec(2, 1, left=L + 0.07, right=R, top=0.87, bottom=0.08, hspace=0.25)
    for k, (short, df) in enumerate(t.items()):
        ax = fig.add_subplot(grid[k, 0])
        h = df.gate_m
        ax.plot(df.required, h, color=c['fg'], lw=2.2, label='required 2001-2019 (Hugonnet - GLAMOS)')
        ax.plot(df.required_1519, h, color=c['fg'], lw=1.1, ls=':',
                label='required 2015-2019 (ITS_LIVE years)')
        ax.plot(df.model_2000_2019, h, color=c['blue'], lw=2, label='model 2000-2019')
        ax.plot(df.model_2014_2019, h, color=c['blue'], lw=1.1, ls='--', label='model 2014-2019')
        for key, col, label in (('obs_inverted', c['red'], 'ITS_LIVE x inverted thickness'),
                                ('obs_iceboost', c['grey'], 'ITS_LIVE x IceBoost'),
                                ('obs_glathida', c['yellow'], 'ITS_LIVE x inverted x GlaThiDa ratio')):
            ax.fill_betweenx(h, df[key + '_lo'], df[key + '_hi'], color=col, alpha=0.4, lw=0,
                             label=label + ' (x 0.8-1.0)')
        ax.axvline(0, color=c['muted'], lw=0.8)
        ax.set_title(df.glacier.iloc[0], fontsize=9, color=c['fg'], loc='left')
        ax.set_xlabel('km3 ice / yr')
        ax.set_ylabel('gate: 2000 surface (m)')
        if k == 0:
            ax.legend(loc='lower right', fontsize=7.5)


def ratio_page(fig, t, gl, c):
    fig.text(L, 0.945, 'Where is the gap? Ratios per gate', fontsize=12, weight='bold',
             color=c['fg'], va='top')
    subtitle(fig, 'Top: flux / required flux (1 = budget closed; gates with < 15 % of the peak '
             'flux left out). Middle: GlaThiDa / inverted thickness and model / ITS_LIVE speed '
             'within 100 m of the gate. Bottom: inverted vs GlaThiDa thickness per grid cell.', c)
    grid = fig.add_gridspec(3, 2, left=L + 0.06, right=R, top=0.85, bottom=0.08, hspace=0.5,
                            wspace=0.3)
    for k, (short, df) in enumerate(t.items()):
        lowest = df.required > 0.15 * df.required.max()   # skip gates with ~no flux
        d = df[lowest]
        h = d.gate_m
        ax = fig.add_subplot(grid[0, k])
        mid = lambda key: 0.5 * (d[key + '_lo'] + d[key + '_hi'])  # noqa: E731
        ax.plot(d.model_2000_2019 / d.required, h, color=c['blue'], lw=2,
                label='model / required (2001-19)')
        ax.plot(d.model_2014_2019 / d.required_1519, h, color=c['blue'], lw=1.1, ls='--',
                label='model / required (2015-19)')
        ax.plot(mid('obs_inverted') / d.required_1519, h, color=c['red'], lw=1.6,
                label='ITS_LIVE x inv. / required (2015-19)')
        ax.plot(mid('obs_glathida') / d.required_1519, h, color=c['yellow'], lw=1.6,
                label='ITS_LIVE x GlaThiDa / required (2015-19)')
        ax.axvline(1, color=c['fg'], lw=0.9, ls='--', label='budget closed')
        ax.set_xlim(0, 2)
        ax.set_title(short, fontsize=9, color=c['fg'], loc='left')
        ax.set_ylabel('gate (m)')
        if k == 0:
            ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0, -0.42), ncol=3,
                      bbox_transform=ax.transAxes)

        ax = fig.add_subplot(grid[1, k])
        ax.plot(df.glathida_ratio, df.gate_m, color=c['yellow'], lw=1.6, marker='o', ms=2.5,
                label='GlaThiDa / inverted thickness')
        ax.plot(df.speed_ratio, df.gate_m, color=c['blue'], lw=1.6, marker='o', ms=2.5,
                label='model / ITS_LIVE speed (2014-19)')
        ax.axvline(1, color=c['fg'], lw=0.9, ls='--', label='equal')
        ax.set_xscale('log')
        ax.set_xlim(0.3, 10)
        ax.set_xticks([0.3, 0.5, 1, 2, 5, 10])
        ax.set_xticklabels(['0.3', '0.5', '1', '2', '5', '10'])
        ax.set_ylabel('gate (m)')
        if k == 0:
            ax.legend(fontsize=7, loc='lower left', bbox_to_anchor=(0, -0.3), ncol=3,
                      bbox_transform=ax.transAxes)

        g = gl[k][1]
        ax = fig.add_subplot(grid[2, k])
        ok = g['ice'] & np.isfinite(g['thkobs']) & (g['thk_inv'] > 0)
        sc = ax.scatter(g['thkobs'][ok], g['thk_inv'][ok], c=g['s'][ok], s=3, cmap='viridis',
                        lw=0)
        top = max(np.nanmax(g['thkobs'][ok]), np.nanmax(g['thk_inv'][ok])) * 1.05
        ax.plot([0, top], [0, top], color=c['fg'], lw=0.9, ls='--', label='1:1')
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect('equal')
        ax.set_xlabel('GlaThiDa thickness (m)')
        ax.set_ylabel('inverted thickness (m)')
        ax.legend(fontsize=7, loc='upper left')
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.outline.set_visible(False)
        cb.set_label('2000 surface (m)', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)


def options_page(fig, c):
    fig.text(L, 0.945, 'How to give the model more flux, generically', fontsize=14,
             weight='bold', color=c['fg'], va='top')
    blocks = [
        ('p', 'Aim: Aletsch as the test case, but only methods that apply to every glacier '
              'without glacier-specific settings, and GlaThiDa stays validation only.'),
        ('h', '1. Thickness free in the inversion'),
        ('p', 'Now the IceBoost geometry is fixed and only tau_ref is fitted. Deep ice is too '
              'thin on both glaciers, and the speed then cannot make up for it (the model '
              'already matches ITS_LIVE). Let IGM optimise the thickness too, with IceBoost as '
              'a prior whose uncertainty grows with depth (e.g. a relative error), so that '
              'velocity and mass conservation can deepen the trough. Test: Aletsch inversion, '
              'then GlaThiDa as the independent check (does max thickness move towards 600 m '
              'and beyond?).'),
        ('h', '2. Mass conservation in the inversion'),
        ('p', 'IGM can fit div(q) against dh/dt - SMB. The SMB is not known before the '
              'calibration, but a first guess from the prior (or a TI model run with its prior '
              'parameters) gives the emergence pattern. Risk: circular with the SMB '
              'calibration; one iteration (inversion -> calibration -> inversion) may be enough.'),
        ('h', '3. ITS_LIVE speed in the ES-MDA calibration'),
        ('p', 'The calibration fits only dh/dt, so SMB and emergence trade off freely; the '
              'too flat tongue SMB is the result. Adding the band speeds of the forward run '
              'in the ITS_LIVE years (2014-2022, time-consistent) as observations, plus a '
              'flux parameter (thickness scale or tau_ref scale) in the ensemble, makes the '
              'flux part of the fit. Generic, no new data.'),
        ('h', '4. TI SMB as the constraint on the tongue'),
        ('p', 'With the TI model the tongue melt follows the temperature (one melt factor, '
              'lapse rate), so the calibration cannot simply weaken the tongue SMB. The '
              'mismatch in dh/dt then has to be absorbed by the flux parameter of point 3. '
              'Earlier TI runs showed exactly this tension (tongue melts too fast, top gains '
              'too much).'),
        ('h', 'Checks before choosing'),
        ('b', 'Repeat the flux-gate check for the lowest third of all 17 flowing glaciers in '
              'the 2015-2019 budget: is the gap general or largest on the deepest glaciers?'),
        ('b', 'Move GlaThiDa to the model date (survey year x local dh/dt) for the ratios.'),
        ('b', 'Sliding: model/ITS_LIVE speed is 0.7 at the 2500 m gate of Aletsch; with '
              'thicker ice the deformation speed rises too (H^4), so thickness first, '
              'then tau_ref.'),
        ('h', 'Not pursued now'),
        ('b', 'The 409 glaciers of central_europe (decided 2026-11-11).'),
        ('b', 'A steeper or flatter GLAMOS profile: GLAMOS is the reference.'),
    ]
    text_block(fig, 0.91, blocks, c)


def main():
    flux = pd.read_csv(os.path.join(HERE, 'tables', 'flux_gate.tsv'), sep='\t')
    tables, gl, period = {}, [], []
    for rgi_id, name, short, gates in GLACIERS:
        g = load(rgi_id, name)
        df = flux[flux.glacier == name].copy()
        req = required(g, 'em_1519')
        check = required(g, 'em_0119')
        df['required_1519'] = df.gate_m.map(req)
        # the recomputed 2001-2019 budget must match flux_gate.py
        assert np.allclose(df.gate_m.map(check), df.required, rtol=1e-3, atol=1e-6)
        tables[short] = df.reset_index(drop=True)
        gl.append((short, g, gates))
        period.append(df[['glacier', 'gate_m', 'required', 'required_1519',
                          'model_2000_2019', 'model_2014_2019']])
    pd.concat(period).to_csv(os.path.join(HERE, 'tables', 'flux_gate_period.tsv'), sep='\t',
                             index=False, float_format='%.4g')
    print(pd.concat(period).round(4).to_string(index=False))

    pages = [('Summary', lambda f, c: summary_page(f, tables, c)),
             ('Limitations', lambda f, c: limits_page(f, c)),
             ('Method', lambda f, c: method_page(f, gl, c)),
             ('Flux profiles', lambda f, c: profile_page(f, tables, c)),
             ('Ratios', lambda f, c: ratio_page(f, tables, gl, c)),
             ('Options', lambda f, c: options_page(f, c))]
    for dark in (False,):   # reports in the light theme
        theme(dark)
        c = colors(dark)
        out = os.path.join(HERE, 'flux_gate_report' + ('_dark' if dark else '') + '.pdf')
        with PdfPages(out) as pdf:
            for n, (topic, draw) in enumerate(pages, 1):
                page(pdf, n, len(pages), topic, c, lambda f: draw(f, c))
        print(out)


if __name__ == '__main__':
    main()
