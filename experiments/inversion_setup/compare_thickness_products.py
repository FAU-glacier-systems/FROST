#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Global ice thickness products against GlaThiDa, glacier by glacier:
Millan et al. (2022), the consensus of Farinotti et al. (2019) and IceBoost
v2.0 (Maffezzoli et al.). Each survey is moved to the reference date of the
product with the Hugonnet et al. (2021) dh/dt (2000-2020) of its cell,
thk_T = thk_survey - dhdt * (year_survey - T), averaged per cell of the
OGGM glacier grid, and compared with the product on that cell. Only cells
where all three products have ice are scored, so the products share the
same cells.

Reference dates (T): Millan 2017 (velocities 2017-2018; as thk_product_year
in FROST), consensus 2003 (RGI6 outlines of region 11 and their DEMs),
IceBoost 2012 (TanDEM-X EDEM 2010-2014, product tagged 2010-2018).
IceBoost was trained on GlaThiDa, so its scores are not independent.

Inputs: OGGM glacier folders of data/results/central_europe_submit
(gridded_data.nc: millan_ice_thickness, hugonnet_dhdt, glacier_mask;
glathida_data.csv) and data/raw/thickness_products
(download_thickness_products.sh).

Run from the repository root:
    python experiments/inversion_setup/compare_thickness_products.py
        the 25 GLAMOS/WGMS glaciers (central_europe_esmda/glaciers.txt)
    python experiments/inversion_setup/compare_thickness_products.py --all
        every glacier of central_europe_submit with GlaThiDa data (a job)
Writes tables/thickness_products_glaciers.tsv, tables/thickness_products.tsv
and plots/thickness_products.png.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GLACIER_DIRS = os.path.join('data', 'results', 'central_europe_submit')
PRODUCTS_DIR = os.path.join('data', 'raw', 'thickness_products')
CONSENSUS_DIR = os.path.join(PRODUCTS_DIR, 'farinotti2019_consensus', 'RGI60-11')
ICEBOOST_DIR = os.path.join(PRODUCTS_DIR, 'iceboost_v2.0', 'rgi11')
PRODUCT_YEAR = {'millan': 2017, 'consensus': 2003, 'iceboost': 2012}


def oggm_dir(rgi_id):
    return os.path.join(GLACIER_DIRS, rgi_id, 'Preprocess', 'data', rgi_id)


def consensus_index():
    """Bounds (lon/lat) of every consensus GeoTIFF, cached in tables/."""
    cache = os.path.join(HERE, 'tables', 'consensus_bounds.csv')
    if os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for path in sorted(glob.glob(os.path.join(CONSENSUS_DIR, '*.tif'))):
        with rasterio.open(path) as src:
            rows.append(dict(path=path, crs=src.crs.to_string(),
                             **dict(zip(['w', 's', 'e', 'n'], transform_bounds(
                                 src.crs, 'EPSG:4326', *src.bounds)))))
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    table.to_csv(cache, index=False)
    return table


def to_grid(paths, crs, transform, shape):
    """Mosaic of GeoTIFFs (band 1) on the glacier grid, NaN without ice."""
    out = np.full(shape, np.nan)
    for path in paths:
        tile = np.full(shape, np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data[~np.isfinite(data) | (data <= 0)] = np.nan
            reproject(data, tile, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        out = np.fmax(out, tile)
    return out


def glacier_fields(rgi_id, index):
    with Dataset(os.path.join(oggm_dir(rgi_id), 'gridded_data.nc')) as ds:
        crs = ds.pyproj_srs
        x, y = np.array(ds['x'][:]), np.array(ds['y'][:])
        mask = np.array(ds['glacier_mask'][:]) > 0.5
        millan = np.ma.filled(ds['millan_ice_thickness'][:].astype(float), np.nan)
        dhdt = np.ma.filled(ds['hugonnet_dhdt'][:].astype(float), np.nan)
    dx, dy = x[1] - x[0], y[1] - y[0]
    flip = dy > 0  # rasters run from north to south
    if flip:
        millan, dhdt, mask = millan[::-1], dhdt[::-1], mask[::-1]
        y = y[::-1]
    transform = from_origin(x[0] - dx / 2, y[0] + abs(dy) / 2, dx, abs(dy))
    shape = mask.shape

    w, s, e, n = transform_bounds(crs, 'EPSG:4326', *rasterio.transform.array_bounds(
        shape[0], shape[1], transform))
    near = index[(index.e > w) & (index.w < e) & (index.n > s) & (index.s < n)]
    consensus = to_grid(near.path, crs, transform, shape)
    iceboost_file = os.path.join(ICEBOOST_DIR, f'{rgi_id}.tif')
    iceboost = to_grid([iceboost_file], crs, transform, shape) \
        if os.path.exists(iceboost_file) else np.full(shape, np.nan)
    millan[~(millan > 0)] = np.nan
    return dict(x=x, y=y, transform=transform, mask=mask, dhdt=dhdt,
                millan=millan, consensus=consensus, iceboost=iceboost)


def glathida_cells(rgi_id, fields):
    """GlaThiDa points per grid cell: survey year and thickness."""
    points = pd.read_csv(os.path.join(oggm_dir(rgi_id), 'glathida_data.csv'))
    points = points[points.thickness > 0].copy()
    date = pd.to_datetime(points.date, errors='coerce')
    points['year'] = date.dt.year + (date.dt.dayofyear - 1) / 365.25
    points = points[points.year.notna()]
    col, row = ~fields['transform'] * (points.x_proj.to_numpy(), points.y_proj.to_numpy())
    points['ix'], points['iy'] = np.floor(col).astype(int), np.floor(row).astype(int)
    ny, nx = fields['mask'].shape
    points = points[(points.ix >= 0) & (points.ix < nx) & (points.iy >= 0) & (points.iy < ny)]
    points = points[fields['mask'][points.iy, points.ix]]
    points['dhdt'] = fields['dhdt'][points.iy, points.ix]
    return points[np.isfinite(points.dhdt)]


def score(obs, model):
    diff = model - obs
    return dict(n=len(obs), obs_mean=obs.mean(), model_mean=model.mean(),
                bias=diff.mean(), mae=np.abs(diff).mean(),
                rms=np.sqrt((diff ** 2).mean()),
                r=np.corrcoef(obs, model)[0, 1] if len(obs) > 2 else np.nan)


def evaluate(rgi_id, index):
    fields = glacier_fields(rgi_id, index)
    points = glathida_cells(rgi_id, fields)
    if points.empty:
        return [], None
    common = (np.isfinite(fields['millan']) & np.isfinite(fields['consensus'])
              & np.isfinite(fields['iceboost']))
    rows, cells = [], {}
    for product, year in PRODUCT_YEAR.items():
        for correction in ('product_year', 'none'):
            shift = (points.year - year) if correction == 'product_year' else 0.0
            p = points.assign(thk=points.thickness - points.dhdt * shift)
            c = p.groupby(['iy', 'ix']).agg(thk=('thk', 'mean'), year=('year', 'mean'))
            iy, ix = (c.index.get_level_values(k).to_numpy() for k in ('iy', 'ix'))
            keep = common[iy, ix]
            obs, model = c.thk.to_numpy()[keep], fields[product][iy, ix][keep]
            if correction == 'product_year':
                cells[product] = (obs, model)
            rows.append(dict(rgi_id=rgi_id, product=product, correction=correction,
                             survey_year=c.year.to_numpy()[keep].mean() if keep.any() else np.nan,
                             volume_km3=np.nansum(np.where(fields['mask'], fields[product], np.nan))
                             * abs(fields['x'][1] - fields['x'][0]) ** 2 / 1e9,
                             **score(obs, model)))
    return rows, cells


def main(glaciers):
    index = consensus_index()
    rows, pooled = [], {p: ([], []) for p in PRODUCT_YEAR}
    for rgi_id in glaciers:
        if not os.path.exists(os.path.join(oggm_dir(rgi_id), 'glathida_data.csv')):
            continue
        glacier_rows, cells = evaluate(rgi_id, index)
        rows += glacier_rows
        if cells:
            for p, (obs, model) in cells.items():
                pooled[p][0].append(obs); pooled[p][1].append(model)
        print(rgi_id, 'done')
    table = pd.DataFrame(rows)
    table = table[table.n > 0]
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(HERE, 'tables', 'thickness_products_glaciers.tsv'),
                 sep='\t', index=False, float_format='%.4g')

    # pooled over all scored cells, and the median of the glacier scores
    summary = []
    for (product, correction), group in table.groupby(['product', 'correction']):
        n = group.n.to_numpy()
        summary.append(dict(
            product=product, correction=correction, glaciers=len(group), cells=n.sum(),
            bias=np.average(group.bias, weights=n),
            mae=np.average(group.mae, weights=n),
            rms=np.sqrt(np.average(group.rms ** 2, weights=n)),
            median_glacier_bias=group.bias.median(),
            median_glacier_rms=group.rms.median()))
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(HERE, 'tables', 'thickness_products.tsv'),
                   sep='\t', index=False, float_format='%.4g')
    print(summary.to_string(index=False, float_format='%.1f'))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (product, (obs, model)) in zip(axes, pooled.items()):
        obs, model = np.concatenate(obs), np.concatenate(model)
        s = score(obs, model)
        ax.scatter(obs, model, s=3, alpha=0.3)
        lim = [0, max(obs.max(), model.max())]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set_title(f'{product} (at {PRODUCT_YEAR[product]})')
        ax.set_xlabel('GlaThiDa, moved to product year (m)')
        ax.text(0.03, 0.97, f"n {s['n']}\nbias {s['bias']:.0f} m\nRMS {s['rms']:.0f} m\nr {s['r']:.2f}",
                transform=ax.transAxes, va='top')
    axes[0].set_ylabel('product thickness (m)')
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'plots', 'thickness_products.png'), dpi=120,
                bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        glaciers = sorted(os.listdir(GLACIER_DIRS))
    else:
        with open(os.path.join('experiments', 'central_europe_esmda', 'glaciers.txt')) as f:
            glaciers = [line.split()[0] for line in f if line.strip()]
    main(glaciers)
