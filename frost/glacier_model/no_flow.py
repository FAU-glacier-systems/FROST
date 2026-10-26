#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Glacier model without ice flow (EnKF ice_flow: none), for glaciers that
barely move: the surface changes by the surface mass balance alone,
dh/dt = smb. In the IGM runs of central_europe_esmda_velyear the flux
divergence of the slow glaciers (ITS_LIVE p99 < 8 m/yr on the ice) was
3-24 % of the SMB (RMS over the bands, mostly ~10 %, 0.1-0.4 m/yr), below
the band data error plus model error; no inversion and no emulator are
needed, and a run takes milliseconds instead of minutes.

Every year the SMB of IGM's smb simple (same formula, clipping and ice-mask
rule) is evaluated on the current surface (elevation feedback) and added to
the thickness, which cannot go below zero: a cell whose ice is gone stops
changing (melt-out). The observed 20-year dh/dt of such a cell is limited the
same way (-H/20), so its information enters through the start thickness
instead of biasing the ablation gradient low (Basodino: a quarter of the
cells melt out at the observed rates, Vedretta Lunga a fifth, Careser a
ninth). Without flow nothing leaves the accumulation area or reaches cells
outside the ice mask.
"""

import os

import numpy as np
from netCDF4 import Dataset


def smb_simple(usurf, icemask, ela, gradabl, gradacc, accmax=100.0):
    """SMB (m ice/yr) of IGM's smb simple for members (leading axis of the
    parameters, gradients per m): (usurf - ela) times the ablation or
    accumulation gradient, clipped to [-100, accmax], and no positive SMB
    outside the ice mask (-10, as IGM)."""
    shape = (-1,) + (1,) * (usurf.ndim - 1)
    ela, gradabl, gradacc = (np.reshape(p, shape) for p in (ela, gradabl, gradacc))
    smb = usurf - ela
    smb = smb * np.where(smb < 0, gradabl, gradacc)
    smb = np.clip(smb, -100.0, accmax)
    return np.where((smb < 0) | icemask, smb, -10.0)


def load_climate(path):
    """Monthly OGGM climate (climate_historical.nc, calendar months from
    January yr_0): temperature and precipitation as (years, 12), the first
    year and the reference height, as IGM's clim_1D_3D reads them."""
    with Dataset(path) as ds:
        temp = np.array(ds['temp'][:], dtype=np.float32).reshape(-1, 12)
        prcp = np.array(ds['prcp'][:], dtype=np.float32).reshape(-1, 12)
        return {'temp': temp, 'prcp': prcp, 'yr_0': int(ds.getncattr('yr_0')),
                'ref_hgt': float(ds.getncattr('ref_hgt'))}


# Constants of the TI processes in igm_wrapper.py (clim_1D_3D, smb_1D_3D)
TEMP_GRADIENT = -0.0065  # K/m
TEMP_ALL_SOLID, TEMP_ALL_LIQUID, TEMP_MELT = 0.0, 2.0, -1.0  # degree C
DAYS_PER_YEAR = 365.242198781
WATER_TO_ICE = 1000.0 / 910.0


def smb_ti(usurf, climate, year, melt_f, prcp_fac, temp_bias):
    """SMB (m ice/yr) of the calendar year `year` for members (leading axis
    of the parameters): IGM's clim_1D_3D + smb_1D_3D, i.e. OGGM's monthly
    TI model. Temperature lapsed from the reference height with -6.5 K/km,
    solid precipitation below 0 degC (linear to none at 2 degC), melt
    melt_f (mm w.e. per degC day) above -1 degC; no ice-mask rule."""
    shape = (-1,) + (1,) * (usurf.ndim - 1)
    melt_f, prcp_fac, temp_bias = (np.reshape(p, shape).astype(np.float32)
                                   for p in (melt_f, prcp_fac, temp_bias))
    index = year - climate['yr_0']
    lapse = TEMP_GRADIENT * (usurf - climate['ref_hgt'])
    smb = np.zeros(np.broadcast_shapes(usurf.shape, melt_f.shape), np.float32)
    for month in range(12):
        temp = climate['temp'][index, month] + temp_bias + lapse
        prcp = climate['prcp'][index, month] * prcp_fac  # kg m-2 per month
        solid = np.clip((TEMP_ALL_LIQUID - temp) / (TEMP_ALL_LIQUID - TEMP_ALL_SOLID), 0, 1)
        melt = melt_f * np.maximum(temp - TEMP_MELT, 0) * DAYS_PER_YEAR / 12 / 1000
        smb += prcp * solid / 1000 - melt  # m w.e.
    return smb * WATER_TO_ICE


def forward(usurf, topg, icemask, ela=None, gradabl=None, gradacc=None,
            year_start=2000, year_end=2020, ti=None):
    """Yearly steps from year_start to year_end for all members at once.

    Args:
        usurf (ndarray)   - start surface, (members, y, x)
        topg (ndarray)    - bed, (y, x)
        icemask (ndarray) - ice of the start (bool), (y, x)
        ela, gradabl, gradacc (ndarray) - ELA SMB parameters per member
                                          (m, per m, per m)
        ti (dict)         - instead of the ELA model: TI parameters per
                            member (melt_f, prcp_fac, temp_bias) and the
                            climate of load_climate; the step from year t
                            to t + 1 uses the climate of year t, as IGM
    Returns:
        years (ndarray)  - year_start..year_end
        usurf, thk, smb (ndarray) - (members, years, y, x); smb[t] is the
                                    SMB of the year ending at years[t]
                                    (smb[0] is that of the first year)
    """
    years = np.arange(year_start, year_end + 1)
    # float32 as IGM: (members, years, y, x) stacks of large glaciers
    # otherwise take several GB
    usurf, topg = usurf.astype(np.float32), topg.astype(np.float32)
    thk = np.maximum(usurf - topg, 0.0)
    surfaces, thicknesses, smbs = [usurf], [thk], []
    for year in years[:-1]:
        if ti is None:
            smb = smb_simple(topg + thk, icemask, ela, gradabl, gradacc)
        else:
            smb = smb_ti(topg + thk, ti['climate'], year, ti['melt_f'], ti['prcp_fac'],
                         ti['temp_bias'])
        thk = np.maximum(thk + smb, 0.0)
        surfaces.append(topg + thk)
        thicknesses.append(thk)
        smbs.append(smb)
    smbs.insert(0, smbs[0])
    return (years, np.stack(surfaces, 1), np.stack(thicknesses, 1),
            np.stack(smbs, 1))


def write_member(path, x, y, years, usurf, thk, smb, topg, icemask):
    """output.nc of one member in the layout of the IGM forward runs (time,
    usurf, thk, smb, topg; velsurf_mag and divflux zero), so that the
    evaluation and the sanity report read it as an IGM member."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with Dataset(path, 'w') as ds:
        ds.createDimension('time', None)
        ds.createDimension('y', len(y))
        ds.createDimension('x', len(x))
        for name, values in [('time', years), ('y', y), ('x', x)]:
            ds.createVariable(name, 'f8', (name,))[:] = values
        zeros = np.zeros_like(thk)
        for name, values in [('usurf', usurf), ('thk', thk), ('smb', smb),
                             ('velsurf_mag', zeros), ('divflux', zeros),
                             ('velbar_mag', zeros)]:
            ds.createVariable(name, 'f4', ('time', 'y', 'x'))[:] = values
        ds.createVariable('topg', 'f4', ('time', 'y', 'x'))[:] = \
            np.repeat(topg[None], len(years), 0)
        ds.createVariable('icemask', 'f4', ('time', 'y', 'x'))[:] = \
            (thk > 0).astype(np.float32)
        ds.setncattr('ice_flow', 'none')
