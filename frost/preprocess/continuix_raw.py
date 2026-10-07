#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
ContinuIX EXP02 (raw data: GeoTIFFs and shapefiles in different grids) ->
one netCDF per glacier in the layout of EXP01, on the EXP01 grid of the
glacier, so that continuix.prepare_input reads it like any other experiment.

- DHDT: the full period; G01's two periods are combined weighted by their
  length. Median resampling where the raw grid is finer, so single outlier
  pixels of the 2-4 m products do not bias a cell.
- DEM: the most complete DEM; prepare_input shifts it with DHDT to the
  start of the period.
- VX, VY: mean over the available years, pixels flagged unreliable
  (V-FLAGGED 0) removed.
- THK: the thickness raster where there is one (G01, G04, G06). Where the
  thickness is only measured along GPR profiles (G02, G03, G05, S02), the
  points are averaged per cell into THKOBS and THK is left out: the
  thickness is then inverted.
- ICEMASK: the outline at the start of the dh/dt period (attribute
  icemask_source = 'ICEMASK': used as it is, see continuix.experiment_icemask).

The raw GeoTIFFs and shapefiles are in the CRS of the EXP01 files, so
nothing is reprojected.

Run on a compute node (the G01 rasters are 18000 x 20000 at 2 m; pyproj
on the login node has no PROJ database):
    python -m frost.preprocess.continuix_raw --glacier G03
"""

import argparse
import os
import warnings

import numpy as np
import rasterio
import rasterio.features
import xarray as xr
from pyogrio.raw import read
from rasterio.warp import Resampling, reproject
from shapely import from_wkb

from frost.preprocess.continuix import _crs, _transform, open_experiment

EXP = 'EXP02'

# Files per glacier. dhdt: (file, weight) pairs, weights are the period
# lengths in years; vel: (VX file, VY file) pairs averaged; thkobs: point
# shapefile and its thickness field
RAW = {
    'G01': dict(
        dhdt=[('EXP02_G01_DHDT_2013-2020.tif', 7.0),     # DEMs 201310-202010
              ('EXP02_G01_DHDT_2020-2023.tif', 2.92)],   # DEMs 202010-202309
        dhdt_timestamp='2013-2023',
        dem='EXP02_G01_DEM_201310.tif', dem_timestamp='20131001',
        thk='EXP02_G01_THK_2013.tif',
        vel=[('EXP02_G01_VX_2017-2018.tif', 'EXP02_G01_VY_2017-2018.tif'),
             ('EXP02_G01_VX_2019-2020.tif', 'EXP02_G01_VY_2019-2020.tif'),
             ('EXP02_G01_VX_2020-2021.tif', 'EXP02_G01_VY_2020-2021.tif')],
        outline='EXP02_G01_outline_2013.shp'),
    'G02': dict(
        dhdt=[('EXP02_G02_DHDT_2006-2013.tif', 1.0)],
        dhdt_timestamp='2006-2013',
        # DEM_2006 is zero off the glacier; DEM_2013 is complete
        dem='EXP02_G02_DEM_2013.tif', dem_timestamp='2013',
        thkobs=('EXP02_G02_THK_20120809.shp', 'thickness'),
        vel=[('EXP02_G02_VX_2017-2018.tif', 'EXP02_G02_VY_2017-2018.tif')],
        outline='EXP02_G02_outline_2006.shp'),
    'G03': dict(
        dhdt=[('EXP02_G03_DHDT_2012-2021.tif', 1.0)],
        dhdt_timestamp='2012-2021',
        dem='EXP02_G03_DEM_20170215.tif', dem_timestamp='20170215',
        thkobs=('EXP02_G03_THK_20170215.shp', 'thickness'),
        vel=[('EXP02_G03_VX.tif', 'EXP02_G03_VY.tif')],
        outline='EXP02_G03_outline_20120819.shp'),
    'G04': dict(
        dhdt=[('EXP02_G04_DHDT_2021-2025_2m.tif', 1.0)],
        dhdt_timestamp='2021-2025',
        dem='EXP02_G04_DEM_2021_2m.tif', dem_timestamp='2021',
        thk='EXP02_G04_THK_2021_25m.tif',
        vel=[('EXP02_G04_VX_2022-2023_gappy.tif',
              'EXP02_G04_VY_2022-2023_gappy.tif')],
        # no CRS in the file; same grid as the velocities
        vel_flag='EXP02_G04_V-FLAGGED.tif',
        outline='EXP02_G04_outline_2021.shp'),
    'G05': dict(
        dhdt=[('EXP02_G05_DHDT_20170901-20230823.tif', 1.0)],
        dhdt_timestamp='20170901-20230823',
        dem='EXP02_G05_DEM_20170901.tif', dem_timestamp='20170901',
        thkobs=('EXP02_G05_THK.shp', 'thk'),
        vel=[('EXP02_G05_VX.tif', 'EXP02_G05_VY.tif')],
        outline='EXP02_G05_outline_20170901.shp'),
    'G06': dict(
        dhdt=[('EXP02_G06_DHDT_2006-2017.tif', 1.0)],
        dhdt_timestamp='2006-2017',
        dem='EXP02_G06_DEM_2017.tif', dem_timestamp='2017',
        thk='EXP02_G06_THK_2006.tif',
        vel=[('EXP02_G06_VX_2015-2016.tif', 'EXP02_G06_VY_2015-2016.tif'),
             ('EXP02_G06_VX_2016-2017.tif', 'EXP02_G06_VY_2016-2017.tif')],
        vel_flag='EXP02_G06_V-FLAGGED_2016-2021.tif',
        outline='EXP02_G06_outline_2006.shp'),
}


def _target(grid):
    """Empty destination array (descending rows, as rasterio writes them)
    and transform of the EXP01 grid."""
    return np.full((grid.sizes['y'], grid.sizes['x']), np.nan), _transform(grid)


def _warp(path, grid, crs, src_crs=None, src_transform=None,
          resampling=None):
    """GeoTIFF band 1 on the EXP01 grid: median of the raw pixels per cell
    where the raw grid is finer, bilinear otherwise. Returns ascending y."""
    dst, dst_transform = _target(grid)
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            values[values == src.nodata] = np.nan
        if resampling is None:
            coarsen = abs(src.res[0]) < abs(dst_transform.a)
            resampling = Resampling.med if coarsen else Resampling.bilinear
        reproject(source=values, destination=dst,
                  src_transform=src_transform or src.transform,
                  src_crs=src_crs or src.crs, src_nodata=np.nan,
                  dst_transform=dst_transform, dst_crs=crs,
                  dst_nodata=np.nan, resampling=resampling)
    return dst[::-1]


def _points_to_grid(path, field, grid):
    """Mean of the point values per EXP01 cell, NaN where there is none."""
    meta, _, geometry, fields = read(path)
    values = np.asarray(fields[list(meta['fields']).index(field)],
                        dtype=np.float64)
    points = from_wkb(geometry)
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    xs, ys = grid['x'].values, grid['y'].values
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    col = np.round((x - xs[0]) / dx).astype(int)
    row = np.round((y - ys[0]) / dy).astype(int)
    ok = (np.isfinite(values) & (col >= 0) & (col < xs.size)
          & (row >= 0) & (row < ys.size))
    total = np.zeros((ys.size, xs.size))
    count = np.zeros((ys.size, xs.size))
    np.add.at(total, (row[ok], col[ok]), values[ok])
    np.add.at(count, (row[ok], col[ok]), 1)
    with np.errstate(invalid='ignore'):
        return np.where(count > 0, total / count, np.nan), int(ok.sum())


def _outline_mask(path, grid):
    """Cells whose centre lies in the outline polygons (ascending y)."""
    _, _, geometry, _ = read(path)
    dst, dst_transform = _target(grid)
    mask = rasterio.features.rasterize(
        [(g, 1) for g in from_wkb(geometry)], out_shape=dst.shape,
        transform=dst_transform, fill=0, dtype='uint8')
    return mask[::-1].astype(bool)


def build(raw_dir, data_dir, glacier, out_dir):
    """Write <out_dir>/EXP02/EXP02_<glacier>.nc; returns its path."""
    path = os.path.join(out_dir, EXP, f'{EXP}_{glacier}.nc')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if glacier == 'S02':
        _build_s02(raw_dir, path)
        return path

    spec = RAW[glacier]
    folder = os.path.join(raw_dir, EXP, glacier)
    reference = open_experiment(data_dir, 'EXP01', glacier)
    grid = xr.Dataset(coords={'x': reference['x'], 'y': reference['y']})
    crs = _crs(reference)
    out = xr.Dataset(coords=grid.coords)

    def add(name, values, **attrs):
        out[name] = (('y', 'x'), np.asarray(values, dtype=np.float32))
        out[name].attrs = attrs

    # DHDT: length-weighted mean of the periods, where all have data
    rates = [_warp(os.path.join(folder, f), grid, crs)
             for f, _ in spec['dhdt']]
    weights = np.array([w for _, w in spec['dhdt']])
    dhdt = sum(w * r for w, r in zip(weights, rates)) / weights.sum()
    add('DHDT', dhdt, units='m i.e./yr', timestamp=spec['dhdt_timestamp'],
        description='EXP02 raw dh/dt, median-resampled to the EXP01 grid: '
                    + ', '.join(f for f, _ in spec['dhdt']))

    dem = _warp(os.path.join(folder, spec['dem']), grid, crs)
    # zeros are nodata (G02 DEM_2006 style)
    add('DEM', np.where(dem > 0, dem, np.nan), units='m a.s.l.',
        timestamp=spec['dem_timestamp'], description=spec['dem'])

    # Velocity: mean of the years, flagged pixels removed
    flag = None
    if 'vel_flag' in spec:
        # G04's flag file has no CRS: it is on the grid of the velocities
        with rasterio.open(os.path.join(folder, spec['vel'][0][0])) as vx:
            vel_crs, vel_transform = vx.crs, vx.transform
        flag_path = os.path.join(folder, spec['vel_flag'])
        with rasterio.open(flag_path) as f:
            has_crs = f.crs is not None
        flag = _warp(flag_path, grid, crs,
                     src_crs=None if has_crs else vel_crs,
                     src_transform=None if has_crs else vel_transform,
                     resampling=Resampling.nearest)
    for i, name in enumerate(['VX', 'VY']):
        stack = np.array([_warp(os.path.join(folder, pair[i]), grid, crs)
                          for pair in spec['vel']])
        stack[stack == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN cells
            values = np.nanmean(stack, axis=0)
        if flag is not None:
            values = np.where(flag == 1, values, np.nan)
        add(name, values, units='m/yr',
            description='mean of ' + ', '.join(p[i] for p in spec['vel'])
                        + (f', {spec["vel_flag"]} == 0 removed'
                           if flag is not None else ''))

    if 'thk' in spec:
        add('THK', _warp(os.path.join(folder, spec['thk']), grid, crs),
            units='m i.e.', description=spec['thk'])
    else:
        shapefile, field = spec['thkobs']
        thkobs, count = _points_to_grid(os.path.join(folder, shapefile),
                                        field, grid)
        add('THKOBS', thkobs, units='m i.e.',
            description=f'{shapefile} ({count} points, field {field}), '
                        'mean per cell')

    add('ICEMASK', _outline_mask(os.path.join(folder, spec['outline']), grid),
        description=spec['outline'])
    out['spatial_ref'] = reference['spatial_ref']
    for name in out.data_vars:
        if name != 'spatial_ref':
            out[name].attrs['grid_mapping'] = 'spatial_ref'
    out.attrs = {'title': f'ContinuIX {EXP} {glacier} on the EXP01 grid',
                 'source': f'built by frost.preprocess.continuix_raw from '
                           f'{folder}',
                 'icemask_source': 'ICEMASK'}
    # y order of the ContinuIX files
    if reference.attrs['y_descending']:
        out = out.sortby('y', ascending=False)
    out.to_netcdf(path)
    _summary(out, glacier)
    return path


def _build_s02(raw_dir, path):
    """S02: one netCDF on the EXP01 grid; THK only along profiles."""
    ds = xr.open_dataset(os.path.join(raw_dir, EXP, 'S02',
                                      'EXP02_S02_all.nc')).load()
    ds = ds.rename({'THK': 'THKOBS'}).drop_vars(['BED', 'UNCT_THK'])
    ds.attrs['icemask_source'] = 'ICEMASK'
    ds.attrs['source'] = 'EXP02_S02_all.nc, THK renamed to THKOBS'
    ds.to_netcdf(path)
    _summary(ds, 'S02')


def _summary(ds, glacier):
    ice = np.isfinite(ds['ICEMASK'].values) & (ds['ICEMASK'].values > 0)
    parts = [f'{EXP} {glacier}: {ice.sum()} ice cells']
    for name in ['DHDT', 'VX', 'THK', 'THKOBS']:
        if name in ds:
            v = ds[name].values[ice]
            ok = np.isfinite(v) & (v != 0)
            parts.append(f'{name} {ok.mean():.0%}'
                         + (f' mean {np.mean(v[ok]):.2f}' if ok.any() else ''))
    print(', '.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--glacier', nargs='+',
                        default=list(RAW) + ['S02'])
    parser.add_argument('--raw_dir', default='data/raw/continuix')
    parser.add_argument('--out_dir', default='data/results/continuix/EXP02_input')
    args = parser.parse_args()
    for glacier in args.glacier:
        print('Wrote', build(args.raw_dir, args.raw_dir, glacier, args.out_dir))


if __name__ == '__main__':
    main()
