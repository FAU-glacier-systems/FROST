#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Surface velocity observations of input.nc (uvelsurfobs, vvelsurfobs), what
the inversion fits tau_ref to:

- itslive: ITS_LIVE v2 regional mosaics (Gardner et al.), 120 m (default,
  2026-10-22: global, maintained, annual data, plausible slow-ice speeds;
  experiments/thickness_esmda). Year None:
  the composite, a 2014-2022 climatology with its intercept on 1 January
  2018; else the annual mosaic of that year (Central Europe: 2013-2022).
- millan: Millan et al. (2022), 2017-2018, 50 m; OGGM shop writes it for
  every glacier.

The ITS_LIVE mosaics are in one projection per region (Central Europe: UTM
32); vectors are resampled to the glacier grid and rotated into its
projection, so glaciers in other UTM zones get correct components. The
mosaics are downloaded once per region into OGGM's download cache.
"""

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform, Resampling

ITSLIVE_URL = ('https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/'
               '{kind}/cog/ITS_LIVE_velocity_120m_RGI{region:02d}A_{year}_v02_{var}.tif')
PRODUCTS = ('itslive', 'millan')


def itslive_on_grid(rgi_id, x, y, crs, year=None):
    """ITS_LIVE vx, vy (m/yr) on the cell centres x, y (ascending) in crs,
    as components of that projection; NaN without data."""
    from oggm import utils

    region = int(rgi_id.split('-')[3])
    dx, dy = x[1] - x[0], y[1] - y[0]
    grid = from_origin(x[0] - dx / 2, y[-1] + dy / 2, dx, dy)
    fields = {}
    for var in ('vx', 'vy'):
        url = ITSLIVE_URL.format(kind='static' if year is None else 'annual',
                                 region=region, var=var,
                                 year='0000' if year is None else int(year))
        path = utils.file_downloader(url)
        if path is None:
            raise RuntimeError(f'ITS_LIVE download failed: {url}')
        out = np.full((len(y), len(x)), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            reproject(data, out, src_transform=src.transform, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=grid, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            src_crs = src.crs
        fields[var] = out[::-1]  # rasters run north to south, the grid south to north

    # rotate into the grid's projection: move each cell centre by one year
    # of flow in the mosaic's projection and map start and end back
    xx, yy = np.meshgrid(x, y)
    start = np.array(transform(crs, src_crs, xx.ravel(), yy.ravel()))
    end = start + np.array([np.nan_to_num(fields['vx']).ravel(),
                            np.nan_to_num(fields['vy']).ravel()])
    end = np.array(transform(src_crs, crs, end[0], end[1]))
    missing = ~(np.isfinite(fields['vx']) & np.isfinite(fields['vy']))
    vx = np.where(missing, np.nan, (end[0] - xx.ravel()).reshape(xx.shape))
    vy = np.where(missing, np.nan, (end[1] - yy.ravel()).reshape(xx.shape))
    return vx, vy


def set_velocity_product(input_file, product, rgi_id, year=None):
    """Write the velocity product to uvelsurfobs and vvelsurfobs of input.nc
    (NaN without data) and record it in the global attributes
    velocity_source and velocity_source_year. millan is what OGGM shop
    wrote already."""
    if product not in PRODUCTS:
        raise ValueError(f'velocity_product must be one of {PRODUCTS}, not {product}')
    label = '2017-2018' if product == 'millan' else (
        '2014-2022 composite' if year is None else str(year))
    with Dataset(input_file, 'r+') as ds:
        if product == 'itslive':
            vx, vy = itslive_on_grid(rgi_id, np.array(ds['x'][:]), np.array(ds['y'][:]),
                                     ds.pyproj_srs, year)
            icemask = np.array(ds['icemask'][:]) > 0.5
            if not np.isfinite(vx[icemask]).any():
                raise ValueError(f'ITS_LIVE has no velocities on the ice of {rgi_id}')
            ds['uvelsurfobs'][:] = vx
            ds['vvelsurfobs'][:] = vy
        ds.setncattr('velocity_source', product)
        ds.setncattr('velocity_source_year', label)
    print(f'Velocity product of {rgi_id}: {product} ({label})')
    return product, label
