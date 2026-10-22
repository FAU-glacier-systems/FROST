#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Start geometry for the calibration without ice flow (EnKF ice_flow: none),
in place of the inversion: the thickness product of the download shifted
to 2000 with the Hugonnet dh/dt (igm_inversion.thickness_at_2000), on the
surface of 2000, written to Preprocess/outputs/output.nc with the fields
the inversion writes (usurf, thk, topg, icemask, velsurfobs_mag; velsurf_mag
zero, no tau_ref), so create_observation and the calibration read it as
usual. The thickness matters only where the ice melts out (no_flow.py).
"""

import os

import numpy as np
from netCDF4 import Dataset

from frost.preprocess.igm_inversion import thickness_at_2000
from frost.preprocess.thickness_validation import validate_thickness


def main(rgi_id_dir):
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    thickness_at_2000(input_file)
    output_file = os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'output.nc')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with Dataset(input_file, 'r') as src, Dataset(output_file, 'w') as out:
        # an inversion at the velocity year moved usurf; the 2000 surface
        # is kept as usurf_2000
        usurf = np.array(src['usurf_2000' if 'usurf_2000' in src.variables
                             else 'usurf'][:], dtype=np.float32)
        icemask = np.array(src['icemask'][:], dtype=np.float32)
        thk = np.where(icemask > 0.5, np.array(src['thk'][:], dtype=np.float32), 0)
        u = np.array(src['uvelsurfobs'][:], dtype=np.float32)
        v = np.array(src['vvelsurfobs'][:], dtype=np.float32)
        out.createDimension('y', len(src['y']))
        out.createDimension('x', len(src['x']))
        for name in ('y', 'x'):
            out.createVariable(name, 'f8', (name,))[:] = src[name][:]
        for name, values in [('usurf', usurf), ('thk', thk), ('topg', usurf - thk),
                             ('icemask', icemask),
                             ('velsurfobs_mag', np.hypot(u, v)),
                             ('velsurf_mag', np.zeros_like(usurf))]:
            out.createVariable(name, 'f4', ('y', 'x'))[:] = values
        for name in ('epsg', 'pyproj_srs'):
            out.setncattr(name, src.getncattr(name))
        out.setncattr('ice_flow', 'none')
        dx = abs(float(src['x'][1] - src['x'][0]))
    print(f'No-flow start geometry: {output_file} (ice volume '
          f'{thk.sum() * dx ** 2 * 1e-9:.3f} km3)')
    validate_thickness(rgi_id_dir)
