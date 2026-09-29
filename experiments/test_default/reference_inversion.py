#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
The igm-examples aletsch Part D inversion (params_D_real) on the example's
own input.nc, run through FROST's inversion step with test_default's
params_inversion.yaml. Next to test_default on Aletsch (OGGM-shop data)
this separates the effect of the input data from the inversion setup.

Run from the repository root, on a GPU node:
    python experiments/test_default/reference_inversion.py
    python frost_pipeline.py --config experiments/test_default/pipeline_config.yml \
        --rgi_id RGI2000-v7.0-G-11-02596
"""

import os
import shutil
import sys

from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.preprocess import igm_inversion

EXAMPLE_INPUT = os.path.join(os.path.dirname(os.getcwd()), 'igm-examples',
                             'aletsch', 'data', 'input.nc')
PARAMS = os.path.join('experiments', 'test_default', 'params_inversion.yaml')
RGI_ID_DIR = os.path.join('data', 'results', 'test_default',
                          'igm_examples_aletsch')


def main():
    data_dir = os.path.join(RGI_ID_DIR, 'Preprocess', 'data')
    os.makedirs(data_dir, exist_ok=True)
    input_file = os.path.join(data_dir, 'input.nc')
    shutil.copy(EXAMPLE_INPUT, input_file)
    # The example grid is UTM 32N; FROST's inversion step copies the CRS
    with Dataset(input_file, 'a') as nc:
        nc.setncattr('epsg', 'EPSG:32632')
        nc.setncattr('pyproj_srs', '+proj=utm +zone=32 +datum=WGS84 '
                                   '+units=m +no_defs')
    igm_inversion.main(rgi_id_dir=RGI_ID_DIR, params_inversion_path=PARAMS)


if __name__ == '__main__':
    main()
