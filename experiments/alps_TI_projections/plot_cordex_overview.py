#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Overview of the CORDEX forcing of the selected glaciers: maps of the
end-of-century change in temperature and precipitation (RCP8.5, median of
the runs, 2081-2100 vs 2000-2019) and the Alps-mean annual anomalies per
RCP (median and 5-95 % of the runs). Writes cordex_overview.png and
cordex_summary.csv to data/results/alps_TI_projections/.

Run from the repository root:
    python experiments/alps_TI_projections/plot_cordex_overview.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
from matplotlib.colors import TwoSlopeNorm
from netCDF4 import Dataset

LIST = 'experiments/alps_TI_projections/glacier_selection.csv'
CORDEX_DIR = 'data/raw/cordex_alps'
DEM = 'data/raw/visualization_context/alpsDEM.tif'
OUT_DIR = 'data/results/alps_TI_projections'
SCENARIOS = {1: ('RCP2.6', '#2a78d6'), 2: ('RCP4.5', '#eb6834'),
             3: ('RCP8.5', '#1baf7a')}
REF, END = (2000, 2019), (2081, 2100)


def read_glacier(rgi_id):
    """Annual temp anomaly (K) and precip change (%) vs REF of every
    complete run per scenario, and the summary values of one glacier."""
    with Dataset(os.path.join(CORDEX_DIR, rgi_id[-5:], 'CORDEX_merged.nc')) as ds:
        temp = np.ma.filled(ds['temp'][:].astype(float), np.nan)
        prcp = np.ma.filled(ds['prcp'][:].astype(float), np.nan)
        info = dict(rgi_id=rgi_id, ref_lon=float(ds.ref_pix_lon),
                    ref_lat=float(ds.ref_pix_lat), ref_hgt=float(ds.ref_hgt))
        yr_0 = int(ds.yr_0)
    prcp[np.abs(prcp) > 1e10] = np.nan
    years = yr_0 + np.arange(temp.shape[-1]) // 12
    annual_years = np.unique(years)
    ref = (annual_years >= REF[0]) & (annual_years <= REF[1])
    end = annual_years >= END[0]
    series = {}
    for e in SCENARIOS:
        ok = np.isfinite(temp[e]).all(-1) & np.isfinite(prcp[e]).all(-1)
        t = temp[e][ok].reshape(ok.sum(), -1, 12).mean(-1)
        p = prcp[e][ok].reshape(ok.sum(), -1, 12).mean(-1)
        dt = t - t[:, ref].mean(1, keepdims=True)
        dp = (p / p[:, ref].mean(1, keepdims=True) - 1) * 100
        series[e] = (dt, dp)
        info[f'dT_{e}'] = float(np.median(dt[:, end].mean(1)))
        info[f'dP_{e}'] = float(np.median(dp[:, end].mean(1)))
        if e == 3:
            info['T_ref'] = float(t[:, ref].mean())
            info['P_ref'] = float(p[:, ref].mean())
    return info, annual_years, series


def main():
    glaciers = pd.read_csv(LIST)
    with ProcessPoolExecutor(16) as executor:
        results = list(executor.map(read_glacier, glaciers.rgi_id))
    summary = glaciers[['rgi_id', 'glac_name', 'country', 'cenlat', 'cenlon']].merge(
        pd.DataFrame([r[0] for r in results]), on='rgi_id')
    summary.to_csv(os.path.join(OUT_DIR, 'cordex_summary.csv'), index=False)
    years = results[0][1]

    fig = plt.figure(figsize=(13, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.12,
                            wspace=0.18)

    # maps of the RCP8.5 end-of-century change
    dem = rioxarray.open_rasterio(DEM).squeeze()
    if dem.rio.crs is not None and not dem.rio.crs.is_geographic:
        dem = dem.rio.reproject('EPSG:4326')
    lon_min, lon_max = summary.cenlon.min() - 0.4, summary.cenlon.max() + 0.4
    lat_min, lat_max = summary.cenlat.min() - 0.3, summary.cenlat.max() + 0.3
    dem = dem.where((dem > -100) & (dem < 5000)).sel(
        x=slice(lon_min, lon_max),
        y=slice(lat_max, lat_min) if dem.y[0] > dem.y[-1] else slice(lat_min, lat_max))
    maps = [('dT_3', 'Warming 2081–2100 vs 2000–2019, RCP8.5 (K)',
             dict(cmap='Oranges', vmin=3.4, vmax=4.4)),
            ('dP_3', 'Precipitation change, RCP8.5 (%)',
             dict(cmap='BrBG', norm=TwoSlopeNorm(0, -7, 7)))]
    for i, (column, title, style) in enumerate(maps):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(dem.values, cmap='Greys', alpha=0.5, vmin=-1500, vmax=4500,
                  extent=[float(dem.x.min()), float(dem.x.max()),
                          float(dem.y.min()), float(dem.y.max())],
                  aspect='auto', interpolation='bilinear')
        points = ax.scatter(summary.cenlon, summary.cenlat, c=summary[column],
                            s=22, edgecolor='#fcfcfb', linewidth=0.5, **style)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect(1 / np.cos(np.radians(46.3)))
        ax.set_title(title, fontsize=10, loc='left')
        ax.tick_params(labelsize=8)
        fig.colorbar(points, ax=ax, shrink=0.75, pad=0.02).ax.tick_params(labelsize=8)

    # Alps-mean annual anomalies: run-wise mean over the glaciers
    panels = [(0, 'Temperature anomaly vs 2000–2019 (K)'),
              (1, 'Precipitation change vs 2000–2019 (%)')]
    for i, (k, label) in enumerate(panels):
        ax = fig.add_subplot(grid[1, i])
        for e, (name, color) in SCENARIOS.items():
            runs = np.mean([r[2][e][k] for r in results], axis=0)
            if k == 1:  # 11-yr running mean: yearly precip is mostly noise
                kernel = np.ones(11) / 11
                runs = np.array([np.convolve(r, kernel, mode='same') for r in runs])
                runs[:, :5] = runs[:, -5:] = np.nan
            low, median, high = np.nanpercentile(runs, [5, 50, 95], axis=0)
            ax.fill_between(years, low, high, color=color, alpha=0.18, linewidth=0)
            ax.plot(years, median, color=color, linewidth=2,
                    label=f'{name} (n={runs.shape[0]})')
        ax.axhline(0, color='#a3a29b', linewidth=0.8)
        ax.axvspan(*REF, color='#e4e3dd', alpha=0.5, linewidth=0)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(color='#e4e3dd', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlim(years[0], years[-1])
        if k == 1:
            ax.set_title('11-yr running mean', fontsize=8, loc='right',
                         color='#73726c')
    fig.axes[-1].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle(f'CORDEX forcing of the {len(summary)} glaciers '
                 f'({summary.groupby(["ref_lon", "ref_lat"]).ngroups} distinct '
                 'reference pixels)', fontsize=12)
    out_file = os.path.join(OUT_DIR, 'cordex_overview.png')
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
