#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann, Johannes J. Fürst
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Projection forcing for the TI model: the W5E5 baseline (climate_historical.nc)
up to the last baseline year, then a CORDEX run bias-corrected to W5E5 per
calendar month over the reference period (additive for temperature,
multiplicative for precipitation). Written in the OGGM climate_historical.nc
format that clim_1D_3D reads.
"""

import os

import numpy as np
import xarray as xr
from netCDF4 import Dataset

SCENARIOS = {1: 'rcp_2_6', 2: 'rcp_4_5', 3: 'rcp_8_5'}


def read_cordex(cordex_file):
    """Monthly temp, prcp and the run names of every complete RCP run of a
    CORDEX_merged.nc (experiment, member, time; months from yr_0)."""
    with Dataset(cordex_file) as ds:
        yr_0 = int(ds.yr_0)
        temp = np.ma.filled(ds['temp'][:].astype(float), np.nan)
        prcp = np.ma.filled(ds['prcp'][:].astype(float), np.nan)
        members = [str(m) for m in ds['member'][:]]
    prcp[np.abs(prcp) > 1e10] = np.nan
    years = yr_0 + np.arange(temp.shape[-1]) // 12
    runs = []
    for e, scenario in SCENARIOS.items():
        complete = (np.isfinite(temp[e]).all(-1)
                    & np.isfinite(prcp[e]).all(-1))
        runs += [(scenario, members[m], temp[e, m], prcp[e, m])
                 for m in np.nonzero(complete)[0]]
    return years, runs


def write_forcing(baseline_file, years, temp, prcp, out_file,
                  ref_period=(2000, 2019), year_end=2100):
    """Bias-correct one CORDEX run to the baseline and write baseline years
    followed by the corrected run up to year_end."""
    base = xr.open_dataset(baseline_file)
    b_years = base['time'].dt.year.values
    b_temp = base['temp'].values.astype(float).reshape(-1, 12)
    b_prcp = base['prcp'].values.astype(float).reshape(-1, 12)
    b_std = base['temp_std'].values.astype(float).reshape(-1, 12)
    b_first = int(b_years[0])
    b_last = int(b_years[-1])

    first = int(years[0])
    temp = temp.reshape(-1, 12)
    prcp = prcp.reshape(-1, 12)
    c_rows = slice(ref_period[0] - first, ref_period[1] - first + 1)
    b_rows = slice(ref_period[0] - b_first, ref_period[1] - b_first + 1)
    temp_shift = b_temp[b_rows].mean(0) - temp[c_rows].mean(0)
    prcp_factor = b_prcp[b_rows].mean(0) / prcp[c_rows].mean(0)

    future = slice(b_last + 1 - first, year_end - first + 1)
    out_temp = np.concatenate([b_temp, temp[future] + temp_shift])
    out_prcp = np.concatenate([b_prcp, prcp[future] * prcp_factor])
    # daily temperature spread: baseline climatology of each month
    out_std = np.concatenate([b_std, np.repeat(b_std[b_rows].mean(0)[None],
                                               out_temp.shape[0] - b_std.shape[0], 0)])

    time = xr.date_range(f'{b_first}-01-01', periods=out_temp.size, freq='MS')
    ds = xr.Dataset(
        {'temp': ('time', out_temp.ravel().astype('float32'), base['temp'].attrs),
         'prcp': ('time', out_prcp.ravel().astype('float32'), base['prcp'].attrs),
         'temp_std': ('time', out_std.ravel().astype('float32'), base['temp_std'].attrs)},
        coords={'time': time})
    ds.attrs = dict(base.attrs)
    ds.attrs.update(yr_0=b_first, yr_1=year_end,
                    climate_source=f"{base.attrs.get('climate_source', 'baseline')} "
                                   f"to {b_last}, bias-corrected CORDEX after",
                    bias_correction_period=f'{ref_period[0]}-{ref_period[1]}',
                    temp_shift_monthly=np.round(temp_shift, 3).tolist(),
                    prcp_factor_monthly=np.round(prcp_factor, 3).tolist())
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    ds.to_netcdf(out_file)
    return temp_shift, prcp_factor
