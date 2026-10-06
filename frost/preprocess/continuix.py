#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
ContinuIX (https://github.com/ContinuIX/ContinuIX-1) input and output for
FROST.

prepare_input: ContinuIX netCDF -> Preprocess/data/input.nc on the model grid
               (replaces the OGGM-shop download)
write_observations: observations.nc for the EnKF from the ContinuIX DHDT
               (replaces the Hugonnet step); run after the inversion
write_submission: EXP##_G##_method##.nc on the original ContinuIX grid
"""

import glob
import json
import os
import re

import numpy as np
import rasterio.transform
import xarray as xr
from netCDF4 import Dataset
from pyproj import CRS
from rasterio.warp import Resampling, reproject
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt

from frost.preprocess.create_observation import write_observation_file

# Ice density used to convert between ice equivalent and water equivalent
# (as IGM's smb_1D_3D)
ICE_DENSITY = 910.0
# Observation period of the synthetic glaciers, which have no timestamps;
# they are in steady state, so only the rates matter
SYNTHETIC_PERIOD = (2000, 2010)
# dh/dt uncertainty (m/yr) where ContinuIX provides none
DEFAULT_DHDT_ERR = 0.2
# Ice-free cells added around the ContinuIX domain, as some glaciers touch it
PAD_CELLS = 5


def meta_path(rgi_id_dir):
    return os.path.join(rgi_id_dir, 'continuix.json')


# ----------------------------------------------------------------------------
# Reading ContinuIX files
# ----------------------------------------------------------------------------

def open_experiment(data_dir, exp, glacier):
    """
    All ContinuIX variables of one experiment and glacier on one grid,
    sorted to ascending x and y.

    EXP16-20 provide the replaced variables in separate files, partly on
    other grids; they are interpolated onto the grid of the base variables.
    """
    exp_dir = os.path.join(data_dir, exp)
    base = os.path.join(exp_dir, f'{exp}_{glacier}_base_vars.nc')
    if os.path.exists(base):
        ds = xr.open_dataset(base).load()
        y = ds['y'].values
        ds.attrs['y_descending'] = int(y[0] > y[-1])
        # _regrid returns ascending y
        ds = ds.sortby(['x', 'y'])
        crs = _crs(ds)
        for path in sorted(glob.glob(os.path.join(exp_dir,
                                                  f'{exp}_{glacier}_glob_*.nc'))):
            other = xr.open_dataset(path).load()
            for name in other.data_vars:
                if name == 'spatial_ref':
                    continue
                ds[name] = (('y', 'x'), _regrid(other, name, _crs(other), ds,
                                                crs, Resampling.bilinear))
                ds[name].attrs = other[name].attrs
    else:
        paths = [os.path.join(exp_dir, f'{exp}_{glacier}.nc'),
                 os.path.join(exp_dir, 'optional', f'{exp}_{glacier}.nc')]
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            raise FileNotFoundError(f'No ContinuIX file for {exp} {glacier} '
                                    f'in {exp_dir}')
        ds = xr.open_dataset(paths[0]).load()
        y = ds['y'].values
        ds.attrs['y_descending'] = int(y[0] > y[-1])
    return ds.sortby(['x', 'y'])


def _crs(ds):
    return CRS.from_wkt(ds['spatial_ref'].attrs['crs_wkt'])


def _transform(ds):
    """Affine transform of a dataset with ascending x and y (flipped below)."""
    x, y = ds['x'].values, ds['y'].values
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    return rasterio.transform.from_origin(x.min() - dx / 2, y.max() + dy / 2,
                                          dx, dy)


def _regrid(src_ds, name, src_crs, dst_ds, dst_crs, resampling):
    """Interpolate src_ds[name] onto the grid of dst_ds (ascending y)."""
    src = np.asarray(src_ds.sortby(['x', 'y'])[name].values, dtype=np.float64)
    dst = np.full((dst_ds.sizes['y'], dst_ds.sizes['x']), np.nan)
    reproject(source=src[::-1], destination=dst,
              src_transform=_transform(src_ds.sortby(['x', 'y'])),
              src_crs=src_crs, src_nodata=np.nan,
              dst_transform=_transform(dst_ds), dst_crs=dst_crs,
              dst_nodata=np.nan, resampling=resampling)
    return dst[::-1]


def _years(text):
    """Decimal years in a ContinuIX timestamp ('2013-2023', '20170901')."""
    years = []
    for token in re.findall(r'\d{8}|\d{4}', str(text)):
        year = int(token[:4])
        if len(token) == 8:
            year += (int(token[4:6]) - 1 + (int(token[6:8]) - 1) / 30) / 12
        if 1900 < year < 2100:
            years.append(year)
    return years


def _uncertainty(ds, name, default):
    """UNCT_<name> field, else the scalar in the 'uncertainty' attribute
    (only if given as a rate), else default."""
    if f'UNCT_{name}' in ds:
        field = ds[f'UNCT_{name}'].values.astype(np.float64)
        return np.where(np.isfinite(field) & (field > 0), field, default)
    text = str(ds[name].attrs.get('uncertainty', ''))
    match = re.search(r'\+-\s*([\d.]+)\s*m\s*/\s*(year|yr)', text)
    value = float(match.group(1)) if match else default
    return np.full(ds[name].shape, value)


def _icemask(ds):
    """Ice mask on the ContinuIX grid: ICEMASK cells (every cell with a year,
    ContinuIX readme) that have thickness, velocity or dh/dt data. S01's
    ICEMASK covers the whole domain, but the ice ends 1 km before it."""
    def has(name):
        return np.isfinite(ds[name].values) & (ds[name].values != 0)
    return has('ICEMASK') & (has('THK') | has('VX') | has('VY') | has('DHDT'))


def _gaps_to_nan(field, icemask):
    """ContinuIX fills missing values with 0; inside the ice these are gaps."""
    field = np.asarray(field, dtype=np.float64).copy()
    field[(field == 0) & icemask] = np.nan
    return field


def _fill(field, mask):
    """Fill NaNs inside mask: linear interpolation, nearest outside the
    convex hull."""
    field = field.copy()
    missing = mask & ~np.isfinite(field)
    valid = np.isfinite(field)
    if missing.any() and valid.any():
        points = np.argwhere(valid)
        targets = np.argwhere(missing)
        values = griddata(points, field[valid], targets, method='linear')
        nearest = griddata(points, field[valid], targets, method='nearest')
        field[missing] = np.where(np.isfinite(values), values, nearest)
    return field


def _fill_nearest(field):
    """Replace all NaNs with the nearest valid value."""
    invalid = ~np.isfinite(field)
    if not invalid.any():
        return field
    indices = distance_transform_edt(invalid, return_distances=False,
                                     return_indices=True)
    return field[tuple(indices)]


# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------

def prepare_input(data_dir, exp, glacier, rgi_id_dir, resolution=50.0):
    """
    Write Preprocess/data/input.nc (IGM input for the inversion) and
    continuix.json (periods, grids) from the ContinuIX files.

    The model grid spacing is max(resolution, ContinuIX spacing), so
    coarse experiments (EXP15, S02) keep their resolution. Fine data are
    block averaged; the ice mask is the cells that are at least half ice.
    The provided THK and BED are not used: the inversion starts from IGM's
    SIA thickness, as in test_default.

    Returns:
        dict  - metadata written to continuix.json
    """
    ds = open_experiment(data_dir, exp, glacier)
    crs = _crs(ds)
    x, y = ds['x'].values, ds['y'].values
    dx_data = float(abs(x[1] - x[0]))
    dx = max(float(resolution), dx_data)

    icemask_data = _icemask(ds)

    # Observation period from the DHDT timestamp; the DEM is shifted to its
    # start with DHDT
    period_years = _years(ds['DHDT'].attrs.get('timestamp', ''))
    if len(period_years) >= 2:
        year_start = int(round(min(period_years)))
        year_end = int(round(max(period_years)))
    else:
        year_start, year_end = SYNTHETIC_PERIOD
    dem_years = _years(ds['DEM'].attrs.get('timestamp', ''))
    year_dem = float(np.mean(dem_years)) if dem_years else float(year_start)

    fields = {
        'usurf': ds['DEM'].values.astype(np.float64),
        'thk': _gaps_to_nan(ds['THK'].values, icemask_data),
        'icemask': icemask_data.astype(np.float64),
        'uvelsurfobs': ds['VX'].values.astype(np.float64),
        'vvelsurfobs': ds['VY'].values.astype(np.float64),
        'dhdt': _gaps_to_nan(ds['DHDT'].values, icemask_data),
        'dhdt_err': _uncertainty(ds, 'DHDT', DEFAULT_DHDT_ERR),
    }
    # Velocity: 0 in both components is missing data
    no_vel = (fields['uvelsurfobs'] == 0) & (fields['vvelsurfobs'] == 0)
    for name in ['uvelsurfobs', 'vvelsurfobs']:
        fields[name][no_vel] = np.nan

    # Model grid: cell centres, ascending, PAD_CELLS ice-free cells around
    # the ContinuIX domain
    x0, x1 = x.min() - dx_data / 2, x.max() + dx_data / 2
    y0, y1 = y.min() - dx_data / 2, y.max() + dx_data / 2
    nx = int(np.ceil((x1 - x0) / dx)) + 2 * PAD_CELLS
    ny = int(np.ceil((y1 - y0) / dx)) + 2 * PAD_CELLS
    x_model = x0 - PAD_CELLS * dx + dx / 2 + dx * np.arange(nx)
    y_model = y0 - PAD_CELLS * dx + dx / 2 + dx * np.arange(ny)
    grid = xr.Dataset(coords={'x': x_model, 'y': y_model})

    resampling = Resampling.average if dx > dx_data else Resampling.bilinear
    source = xr.Dataset({k: (('y', 'x'), v) for k, v in fields.items()},
                        coords={'x': x, 'y': y})
    model = {k: _regrid(source, k, crs, grid, crs, resampling) for k in fields}

    icemask = np.nan_to_num(model['icemask']) >= 0.5
    usurf = _fill_nearest(model['usurf'])
    # observations as provided: NaN where there is no data
    dhdt = np.where(icemask, model['dhdt'], np.nan)
    dhdt_err = np.where(icemask, model['dhdt_err'], np.nan)

    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    os.makedirs(os.path.dirname(input_file), exist_ok=True)
    epsg = crs.to_epsg()
    with Dataset(input_file, 'w') as nc:
        nc.createDimension('x', nx)
        nc.createDimension('y', ny)
        for name, values in [('x', x_model), ('y', y_model)]:
            var = nc.createVariable(name, 'f8', (name,))
            var[:] = values
            var.units = 'm'
        out = {
            'usurf': usurf,
            # replaced by IGM's SIA thickness in the inversion
            'thk': np.where(icemask, _fill(model['thk'], icemask), 0.0),
            'icemask': icemask,
            'uvelsurfobs': model['uvelsurfobs'],
            'vvelsurfobs': model['vvelsurfobs'],
            # keep ContinuIX THK out of the inversion
            'thkobs': np.full((ny, nx), np.nan),
            'usurfobs': usurf,
            'icemaskobs': icemask,
            'dhdt': dhdt,
            'dhdt_err': dhdt_err,
        }
        for name, values in out.items():
            dtype = 'i1' if values.dtype == bool else 'f4'
            var = nc.createVariable(name, dtype, ('y', 'x'))
            var[:] = values.astype(dtype)
        nc.setncattr('epsg', f'EPSG:{epsg}' if epsg else 'EPSG:unknown')
        nc.setncattr('pyproj_srs', crs.to_proj4())
        nc.setncattr('crs_wkt', crs.to_wkt())

    meta = {
        'experiment': exp,
        'glacier': glacier,
        'data_dir': os.path.abspath(data_dir),
        'year_start': year_start,
        'year_end': year_end,
        'year_dem': year_dem,
        'resolution_data': dx_data,
        'resolution_model': dx,
        'resampling': resampling.name,
        'ice_area_km2': float(icemask.sum() * dx ** 2 / 1e6),
        'dhdt_mean': float(np.nanmean(dhdt[icemask])),
        'usurf_median': float(np.median(usurf[icemask])),
        'usurf_min': float(usurf[icemask].min()),
        'usurf_max': float(usurf[icemask].max()),
    }
    with open(meta_path(rgi_id_dir), 'w') as f:
        json.dump(meta, f, indent=4)
    print(f'{exp} {glacier}: {nx}x{ny} cells of {dx:g} m (data {dx_data:g} m), '
          f'{meta["ice_area_km2"]:.1f} km2, dh/dt {year_start}-{year_end} '
          f'{meta["dhdt_mean"]:.2f} m/yr')
    return meta


def smb_prior(meta, gradients_mean, gradients_std):
    """EnKF prior: ELA around the median ice elevation, spread over a third
    of the elevation range; gradients from the config."""
    return ({'ela': meta['usurf_median'], **gradients_mean},
            {'ela': max(100.0, (meta['usurf_max'] - meta['usurf_min']) / 3),
             **gradients_std})


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------

def write_observations(rgi_id_dir):
    """
    observations.nc for the EnKF (create_observation.write_observation_file):
    the ContinuIX DHDT and its error as provided, on the grid and with the
    bed of the inversion result.

    The start surface is the ContinuIX DEM shifted with dh/dt (gaps filled
    for this shift only) from the DEM date to the start of the period.
    """
    with open(meta_path(rgi_id_dir)) as f:
        meta = json.load(f)

    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    with Dataset(input_file) as nc:
        dhdt = np.array(nc['dhdt'][:].filled(np.nan), dtype=np.float64)
        dhdt_err = np.array(nc['dhdt_err'][:].filled(np.nan), dtype=np.float64)
        epsg, pyproj_srs = nc.epsg, nc.pyproj_srs
    output_file = os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'output.nc')
    with Dataset(output_file) as nc:
        x, y = nc['x'][:], nc['y'][:]
        usurf_dem = np.array(nc['usurf'][:], dtype=np.float64)
        topg = np.array(nc['topg'][:], dtype=np.float64)
        icemask = np.array(nc['icemask'][:]) > 0.5
        velsurfobs_mag = np.array(nc['velsurfobs_mag'][:])

    shift = np.nan_to_num(_fill(dhdt, icemask)) \
        * (meta['year_dem'] - meta['year_start'])
    usurf_start = np.maximum(topg, usurf_dem - shift)

    write_observation_file(
        os.path.join(rgi_id_dir, 'observations.nc'), x=x, y=y,
        years=[meta['year_start'], meta['year_end']], usurf=usurf_start,
        topg=topg, icemask=icemask, dhdt=np.where(icemask, dhdt, np.nan),
        dhdt_err=np.where(icemask, dhdt_err, np.nan),
        velsurf_mag=velsurfobs_mag, epsg=epsg, pyproj_srs=pyproj_srs)


# ----------------------------------------------------------------------------
# Submission
# ----------------------------------------------------------------------------

def ela_smb(usurf, ela, abl_grad, acc_grad, accmax=100.0):
    """IGM's smb 'simple' (m ice eq./yr); gradients in m/yr per km."""
    smb = usurf - ela
    smb = smb * np.where(smb < 0, abl_grad, acc_grad) / 1000
    return np.clip(smb, -100, accmax)


def write_submission(rgi_id_dir, path, method_description=''):
    """
    Result file for ContinuIX on the original grid.

    SMB: calibrated ELA model of every final ensemble member evaluated on
         the ContinuIX DEM at the middle of the dh/dt period; mean and
         standard deviation over the ensemble
    FDIV: flux divergence of the forward runs of the final ensemble
         (Ensemble/Member_*/outputs/output.nc), averaged over the period,
         bilinearly interpolated from the model grid
    THK, BED: inverted thickness and the bed that belongs to it, only if
         the inversion changed the thickness
    """
    with open(meta_path(rgi_id_dir)) as f:
        meta = json.load(f)
    with open(os.path.join(rgi_id_dir, 'calibration_results.json')) as f:
        calibration = json.load(f)
    ds = open_experiment(meta['data_dir'], meta['experiment'], meta['glacier'])
    crs = _crs(ds)
    icemask = _icemask(ds)

    # SMB on the original grid
    keys = list(calibration['initial_smb'].keys())
    members = [dict(zip(keys, m)) for m in calibration['final_ensemble']]
    dhdt = np.nan_to_num(_fill(_gaps_to_nan(ds['DHDT'].values, icemask), icemask))
    year_mid = (meta['year_start'] + meta['year_end']) / 2
    usurf_mid = ds['DEM'].values + dhdt * (year_mid - meta['year_dem'])
    smb = np.array([ela_smb(usurf_mid, m['ela'], m['abl_grad'], m['acc_grad'])
                    for m in members])

    # Flux divergence and thickness from the model grid
    fdiv = []
    for output in sorted(glob.glob(os.path.join(rgi_id_dir, 'Ensemble',
                                                'Member_*', 'outputs',
                                                'output.nc'))):
        with Dataset(output) as nc:
            # the first record is written before divflux is computed
            fdiv.append(np.mean(np.array(nc['divflux'][1:]), axis=0))
    if len(fdiv) != len(members):
        raise RuntimeError(f'{len(fdiv)} forward runs for {len(members)} '
                           'ensemble members; rerun the calibration')
    model_file = os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'output.nc')
    with Dataset(model_file) as nc:
        model = xr.Dataset({'thk': (('y', 'x'), np.array(nc['thk'][:]))},
                           coords={'x': nc['x'][:], 'y': nc['y'][:]})
    model['fdiv_mean'] = (('y', 'x'), np.mean(fdiv, axis=0))
    model['fdiv_std'] = (('y', 'x'), np.std(fdiv, axis=0))
    on_grid = {name: _regrid(model, name, crs, ds, crs, Resampling.bilinear)
               for name in ['thk', 'fdiv_mean', 'fdiv_std']}

    def masked(values):
        return np.where(icemask, values, np.nan).astype(np.float32)

    thk = masked(np.maximum(np.nan_to_num(on_grid['thk']), 0))
    variables = {
        'SMB': (masked(smb.mean(axis=0)), 'surface mass balance', 'm i.e./yr'),
        'UNCT_SMB': (masked(smb.std(axis=0)),
                     'uncertainty of SMB (1 sigma of the calibrated ensemble)',
                     'm i.e./yr'),
        'FDIV': (masked(on_grid['fdiv_mean']), 'flux divergence', 'm i.e./yr'),
        'UNCT_FDIV': (masked(on_grid['fdiv_std']),
                      'uncertainty of FDIV (1 sigma of the calibrated ensemble)',
                      'm i.e./yr'),
        'THK': (thk, 'ice thickness (modified: IGM inversion of the surface '
                     'velocity, not the provided THK)', 'm i.e.'),
        'BED': (masked(ds['DEM'].values - thk),
                'basal topography (modified: DEM - THK)', 'm a.s.l.'),
        'DENSITY': (masked(np.full(icemask.shape, ICE_DENSITY)),
                    'density used for ice equivalent', 'kg m-3'),
    }
    if not _thk_inverted(rgi_id_dir):
        del variables['THK'], variables['BED']
    out = xr.Dataset(coords={'x': ds['x'], 'y': ds['y']})
    for name, (values, long_name, units) in variables.items():
        out[name] = (('y', 'x'), values)
        out[name].attrs = {'long_name': long_name, 'units': units,
                           'grid_mapping': 'spatial_ref',
                           'timestamp': f"{meta['year_start']}-{meta['year_end']}"}
    out['spatial_ref'] = ds['spatial_ref']
    parameters = np.array(calibration['final_ensemble'])
    out.attrs = {
        'title': f"FROST results for ContinuIX {meta['experiment']} "
                 f"{meta['glacier']}",
        'description': method_description,
        'model_resolution_m': meta['resolution_model'],
        'period': f"{meta['year_start']}-{meta['year_end']}",
        'smb_parameters': ', '.join(keys),
        'smb_parameters_mean': ', '.join(f'{v:.3f}' for v in parameters.mean(0)),
        'smb_parameters_std': ', '.join(f'{v:.3f}' for v in parameters.std(0)),
    }
    # y order of the ContinuIX file
    if ds.attrs['y_descending']:
        out = out.sortby('y', ascending=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_netcdf(path)
    print('Wrote', path)
    return out


def _thk_inverted(rgi_id_dir):
    """Whether the IGM inversion had the thickness as a control variable."""
    import yaml
    with open(os.path.join(rgi_id_dir, 'Preprocess', 'experiment',
                           'params_inversion.yaml')) as f:
        params = yaml.safe_load(f)
    return 'thk' in [v['name'] for v in
                     params['assimilations']['field_inversion']['variables']]
