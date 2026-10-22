#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Ice thickness product of input.nc, the start of the inversion:

- iceboost: IceBoost v2.0 (Maffezzoli et al.), machine learning trained on
  GlaThiDa, one GeoTIFF per RGI7 glacier for all regions (default).
  Zenodo 10.5281/zenodo.17724512. Reference year 2012: its surface is the
  TanDEM-X EDEM (2010-2014).
- consensus: Farinotti et al. (2019), five-model consensus on RGI6
  outlines, mosaicked onto the RGI7 grid from all RGI6 glaciers whose centre
  lies within Lmax of the grid. Reference year: date of the RGI6 outline
  (BgnDate) of the glacier covering most of the grid's ice.
- millan: Millan et al. (2022), velocity based; reference year 2017
  (velocities 2017-2018). OGGM shop writes it for every glacier.

Against date-corrected GlaThiDa on 23 Central European glaciers IceBoost has
bias +8 m and RMS 46 m, the consensus +1 m and 41 m, Millan +69 m and 115 m
(experiments/inversion_setup/STATUS.md); IceBoost was trained on these
points.

All files go to OGGM's download cache: the regional zip of IceBoost (the
glacier's GeoTIFF is extracted next to it), the consensus GeoTIFFs and the
RGI6 statistics table.
"""

import os
import shutil
import zipfile

import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform, Resampling

ICEBOOST_URL = 'https://zenodo.org/records/17724512/files/RGI70G_rgi{region}.zip'
CONSENSUS_URL = 'https://cluster.klima.uni-bremen.de/~fmaussion/icevol/composite'
RGI6_STATS_URL = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62_stats.csv'
PRODUCTS = ('iceboost', 'consensus', 'millan')
PRODUCT_YEAR = {'iceboost': 2012, 'millan': 2017}  # consensus: per glacier


def iceboost_on_grid(rgi_id, x, y, crs):
    """IceBoost thickness of the RGI7 glacier rgi_id on the cell centres
    x, y (ascending) in crs, NaN without ice."""
    from oggm import utils

    region = int(rgi_id.split('-')[3])
    zip_path = utils.file_downloader(ICEBOOST_URL.format(region=region))
    if zip_path is None:
        raise RuntimeError(f'IceBoost download of region {region} failed')

    # the glacier's GeoTIFF, extracted next to the zip (GDAL's /vsizip/
    # failed on Aletsch in a job); temporary name, so parallel tasks never
    # read a half-written file
    tif = os.path.join(os.path.dirname(zip_path), f'rgi{region}', f'{rgi_id}.tif')
    if not os.path.exists(tif):
        os.makedirs(os.path.dirname(tif), exist_ok=True)
        part = f'{tif}.{os.getpid()}.part'
        with zipfile.ZipFile(zip_path) as archive, open(part, 'wb') as out:
            shutil.copyfileobj(archive.open(f'rgi{region}/{rgi_id}.tif'), out)
        os.replace(part, tif)

    dx, dy = x[1] - x[0], y[1] - y[0]
    thk = np.full((len(y), len(x)), np.nan, dtype=np.float32)
    with rasterio.open(tif) as src:
        data = src.read(1).astype(np.float32)
        data[~np.isfinite(data) | (data <= 0)] = np.nan
        reproject(data, thk, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=np.nan,
                  dst_transform=from_origin(x[0] - dx / 2, y[-1] + dy / 2, dx, dy),
                  dst_crs=crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return thk[::-1]  # rasters run north to south, the grid south to north


def consensus_on_grid(rgi_id, x, y, crs):
    """Consensus thickness on the cell centres x, y (ascending) in crs, NaN
    without ice, and its reference year (outline date of the RGI6 glacier
    with the most ice cells on the grid; None without ice)."""
    from oggm import utils

    stats = pd.read_csv(utils.file_downloader(RGI6_STATS_URL),
                        usecols=['RGIId', 'BgnDate', 'CenLon', 'CenLat',
                                 'O1Region', 'Lmax'])
    stats = stats[stats.O1Region.astype(int) == int(rgi_id.split('-')[3])]
    cx, cy = map(np.array, transform('EPSG:4326', crs, stats.CenLon.to_numpy(),
                                     stats.CenLat.to_numpy()))
    reach = stats.Lmax.to_numpy()
    near = stats[(cx > x[0] - reach) & (cx < x[-1] + reach)
                 & (cy > y[0] - reach) & (cy < y[-1] + reach)]

    dx, dy = x[1] - x[0], y[1] - y[0]
    grid = from_origin(x[0] - dx / 2, y[-1] + dy / 2, dx, dy)
    shape = (len(y), len(x))
    thk = np.full(shape, np.nan)
    source = np.full(shape, -1)
    for k, rgi6_id in enumerate(near.RGIId):
        path = utils.file_downloader(
            f'{CONSENSUS_URL}/{rgi6_id[:8]}/{rgi6_id}_thickness.tif')
        if path is None:
            continue
        tile = np.full(shape, np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            data[~np.isfinite(data) | (data <= 0)] = np.nan
            reproject(data, tile, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=grid, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        tile = tile[::-1]  # rasters run north to south, the grid south to north
        larger = np.isfinite(tile) & ~(thk >= tile)
        thk[larger], source[larger] = tile[larger], k
    if not (source >= 0).any():
        return thk, None
    main = near.iloc[np.bincount(source[source >= 0]).argmax()]
    year = int(str(main.BgnDate)[:4])
    if year < 1900:  # -9999999: unknown outline date
        known = near.BgnDate[near.BgnDate > 0].astype(str).str[:4].astype(int)
        if known.empty:
            raise ValueError(f'No outline date for the consensus of {main.RGIId}')
        year = int(known.median())
    return thk, year


def set_thickness_product(input_file, product, rgi_id):
    """Write the thickness product to thk and thkinit of input.nc (and to
    thk_product, if an earlier inversion kept one) and record it in the
    global attributes thk_source and thk_source_year. millan is what OGGM
    shop wrote already."""
    if product not in PRODUCTS:
        raise ValueError(f'thickness_product must be one of {PRODUCTS}, not {product}')
    year = PRODUCT_YEAR.get(product)
    with Dataset(input_file, 'r+') as ds:
        if product != 'millan':
            icemask = np.array(ds['icemask'][:]) > 0.5
            grid = (rgi_id, np.array(ds['x'][:]), np.array(ds['y'][:]), ds.pyproj_srs)
            if product == 'iceboost':
                thk = iceboost_on_grid(*grid)
            else:
                thk, year = consensus_on_grid(*grid)
            if not np.isfinite(thk[icemask]).any():
                raise ValueError(f'{product} has no ice on the grid of {rgi_id}')
            thk = np.where(icemask & np.isfinite(thk), thk, 0.0)
            for name in ('thk', 'thkinit', 'thk_product'):
                if name in ds.variables:
                    ds[name][:] = thk
        ds.setncattr('thk_source', product)
        ds.setncattr('thk_source_year', year)
    print(f'Thickness product of {rgi_id}: {product} ({year})')
    return product, year
