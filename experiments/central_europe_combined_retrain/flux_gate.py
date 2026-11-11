#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Flux-gate check: how much ice flows into the area below an elevation
contour, from the data and in the model (Aletsch and Rhone of
central_europe_retrain, the reference ELA run).

The gates are the contours of the 2000 surface (usurf_2000) every 100 m.
For each gate h, with R the ice below h, in km3 ice/yr:

- required: the inflow that the data need,
      integral over R of (Hugonnet dh/dt - GLAMOS Ba / 0.91),
  i.e. the emergence implied by dh/dt = SMB + emergence. GLAMOS Ba is the
  mean annual balance 2001-2019 of the 100 m band of the cell.
- model: - integral over R of IGM's divflux, mean over the 32 posterior
  members and the years 2000-2019 (and 2014-2019, the ITS_LIVE years).
- observed: ITS_LIVE 2014-2022 surface velocity (vector) x thickness through
  the faces of the cells on the contour (finite volume), with the surface
  speed scaled to the depth average by 0.8-1.0, for
    * the inverted thickness of FROST (thk_inversion, 2018),
    * the IceBoost product (thk_product),
    * the inverted thickness scaled by the GlaThiDa / inverted ratio of the
      cells within 100 m elevation of the gate (validation only).

The GlaThiDa ratio and the speed ratio model / ITS_LIVE per gate are written
to the table as well.

Run from the repository root:
    python experiments/central_europe_combined_retrain/flux_gate.py --dark
Writes tables/flux_gate.tsv and plots/flux_gate[_dark].png.
"""

import glob
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.visualization import plot_style  # noqa: E402
from frost.visualization.plot_style import COLORS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join('data', 'results', 'central_europe_retrain')
GLAMOS = os.path.join('data', 'raw', 'glamos', 'massbalance_observation_elevationbins.csv')
GLACIERS = [('RGI2000-v7.0-G-11-02596', 'Grosser Aletschgletscher'),
            ('RGI2000-v7.0-G-11-01706', 'Rhonegletscher')]
YEARS = (2001, 2019)       # GLAMOS hydrological years (end year)
DEPTH_AVG = (0.8, 1.0)     # depth-averaged / surface speed
RHO = 0.91                 # w.e. -> ice


def read(path, name):
    with Dataset(path) as d:
        return np.array(d[name][:].filled(np.nan) if np.ma.isMaskedArray(d[name][:])
                        else d[name][:], dtype=float)


def glamos_bands(name):
    with open(GLAMOS, encoding='latin-1') as f:
        lines = [line for line in f if line.startswith(name + ';')]
    g = pd.read_csv(io.StringIO(''.join(lines)), sep=';', header=None,
                    names='name id ds dw de Bw Bs Ba area hmin hmax obs'.split())
    g['year'] = g.de.str[:4].astype(int)
    g = g[(g.year >= YEARS[0]) & (g.year <= YEARS[1])]
    return (pd.to_numeric(g.Ba) / 1000).groupby(pd.to_numeric(g.hmin)).mean()


def gate_inflow(inside, ice, ux, uy, thk, dx):
    """Ice flux (m3/yr) into `inside` through the faces to ice cells
    outside, face values as the mean of the two cells."""
    q = 0.0
    # x faces: cell (i, j) | (i, j+1), normal +x
    a, b = inside[:, :-1], inside[:, 1:]
    both = ice[:, :-1] & ice[:, 1:]
    flux = 0.5 * (thk[:, :-1] + thk[:, 1:]) * 0.5 * (ux[:, :-1] + ux[:, 1:]) * dx
    q += np.sum(flux[b & ~a & both]) - np.sum(flux[a & ~b & both])
    # y faces: cell (i, j) | (i+1, j), normal +y (y ascending)
    a, b = inside[:-1, :], inside[1:, :]
    both = ice[:-1, :] & ice[1:, :]
    flux = 0.5 * (thk[:-1, :] + thk[1:, :]) * 0.5 * (uy[:-1, :] + uy[1:, :]) * dx
    q += np.sum(flux[b & ~a & both]) - np.sum(flux[a & ~b & both])
    return q


def glacier(rgi_id, name):
    base = os.path.join(RUN, rgi_id)
    inp = os.path.join(base, 'Preprocess', 'data', 'input.nc')
    inv = os.path.join(base, 'Preprocess', 'outputs', 'output.nc')
    x = read(inp, 'x')
    dx = abs(x[1] - x[0])
    s = read(inp, 'usurf_2000')
    ice = read(inp, 'thk_2000') > 1
    dhdt = read(inp, 'dhdt')
    ux, uy = read(inp, 'uvelsurfobs'), read(inp, 'vvelsurfobs')
    thkobs = read(inp, 'thkobs')
    thk_inv = read(inv, 'thk_inversion')
    thk_ib = read(inp, 'thk_product')

    ba = glamos_bands(name)
    band = np.floor(s / 100) * 100
    ba_cell = np.full(s.shape, np.nan)
    ba_cell[ice] = pd.Series(band[ice]).map(ba).values
    emergence = dhdt - ba_cell / RHO

    # model: mean -divflux over the members, 2000-2019 and 2014-2019
    div_all, div_late, speed_late = 0.0, 0.0, 0.0
    members = sorted(glob.glob(os.path.join(base, 'Ensemble', 'Member_*', 'outputs', 'output.nc')))
    for path in members:
        with Dataset(path) as d:
            t = np.array(d['time'][:])
            div = np.array(d['divflux'][:])
            vel = np.array(d['velsurf_mag'][:])
        div_all = div_all + div[(t >= 2000) & (t <= 2019)].mean(0)
        div_late = div_late + div[(t >= 2014) & (t <= 2019)].mean(0)
        speed_late = speed_late + vel[(t >= 2014) & (t <= 2020)].mean(0)
    div_all, div_late, speed_late = (v / len(members) for v in (div_all, div_late, speed_late))
    speed_obs = np.hypot(ux, uy)

    rows = []
    area = dx * dx
    for h in np.arange(np.ceil(np.nanmin(s[ice]) / 100) * 100 + 100,
                       np.nanmax(s[ice]), 100):
        inside = ice & (s < h)
        near = ice & (np.abs(s - h) < 100)
        ok = near & np.isfinite(thkobs) & (thk_inv > 0)
        ratio = np.nanmedian(thkobs[ok] / thk_inv[ok]) if ok.sum() >= 5 else np.nan
        row = dict(glacier=name, gate_m=int(h), area_below_km2=inside.sum() * area / 1e6,
                   required=np.nansum(emergence[inside]) * area,
                   model_2000_2019=-np.sum(div_all[inside]) * area,
                   model_2014_2019=-np.sum(div_late[inside]) * area,
                   glathida_cells=int(ok.sum()), glathida_ratio=ratio,
                   speed_ratio=np.nanmedian(speed_late[near] / speed_obs[near]))
        for key, thk in (('obs_inverted', thk_inv), ('obs_iceboost', thk_ib)):
            q = gate_inflow(inside, ice, ux, uy, thk, dx)
            row[key + '_lo'], row[key + '_hi'] = q * DEPTH_AVG[0], q * DEPTH_AVG[1]
        q = gate_inflow(inside, ice, ux, uy, thk_inv * (ratio if np.isfinite(ratio) else np.nan), dx)
        row['obs_glathida_lo'], row['obs_glathida_hi'] = q * DEPTH_AVG[0], q * DEPTH_AVG[1]
        rows.append(row)
    df = pd.DataFrame(rows)
    flux_cols = [c for c in df if c.startswith(('required', 'model', 'obs'))]
    df[flux_cols] = df[flux_cols] / 1e9   # km3 ice/yr
    return df


def plot(tables, style):
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 5.5))
    for ax, df in zip(np.atleast_1d(axes), tables):
        h = df.gate_m
        ax.plot(df.required, h, color=plot_style.fg(), lw=2.2,
                label='required: Hugonnet dh/dt - GLAMOS Ba')
        ax.plot(df.model_2000_2019, h, color=COLORS['blue'], lw=2, label='model, 2000-2019')
        ax.plot(df.model_2014_2019, h, color=COLORS['blue'], lw=1.2, ls='--',
                label='model, 2014-2019')
        for key, color, label in (('obs_inverted', COLORS['red'], 'ITS_LIVE x inverted thk'),
                                  ('obs_iceboost', COLORS['grey'], 'ITS_LIVE x IceBoost'),
                                  ('obs_glathida', COLORS['yellow'],
                                   'ITS_LIVE x inverted thk x GlaThiDa ratio')):
            ax.fill_betweenx(h, df[key + '_lo'], df[key + '_hi'], color=color, alpha=0.35,
                             lw=0, label=label + ' (0.8-1.0)')
        ax.axvline(0, color=plot_style.fg(), lw=0.8)
        ax.set_title(df.glacier.iloc[0])
        ax.set_xlabel('Ice flux into the area below the gate (km$^3$ ice / yr)')
        ax.set_ylabel('Gate: elevation of the 2000 surface (m)')
    np.atleast_1d(axes)[0].legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    return style.savefig(fig, os.path.join(HERE, 'plots', 'flux_gate' + ('_dark' if style.dark else '')))


def main():
    style = plot_style.setup()
    tables = [glacier(*g) for g in GLACIERS]
    out = pd.concat(tables)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    out.to_csv(os.path.join(HERE, 'tables', 'flux_gate.tsv'), sep='\t', index=False,
               float_format='%.4g')
    print(out.round(3).to_string(index=False))
    print(plot(tables, style))


if __name__ == '__main__':
    main()
