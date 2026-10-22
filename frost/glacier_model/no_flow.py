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


def forward(usurf, topg, icemask, ela, gradabl, gradacc, year_start, year_end):
    """Yearly steps from year_start to year_end for all members at once.

    Args:
        usurf (ndarray)   - start surface, (members, y, x)
        topg (ndarray)    - bed, (y, x)
        icemask (ndarray) - ice of the start (bool), (y, x)
        ela, gradabl, gradacc (ndarray) - SMB parameters per member
                                          (m, per m, per m)
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
    for _ in years[1:]:
        smb = smb_simple(topg + thk, icemask, ela, gradabl, gradacc)
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
