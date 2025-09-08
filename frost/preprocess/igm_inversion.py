#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
from pathlib import Path
import os
import shutil
import yaml
import numpy as np
from netCDF4 import Dataset
import netCDF4


def main(rgi_id_dir, params_inversion_path):
    """
    Generates params.json for IGM inversion and runs igm_run.

    Args:
        rgi_id (str) - Glacier RGI ID
    """

    # Check if velocity observations are available

    # Define input and output file names
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')

    flag_velsurfobs = False
    with Dataset(input_file, 'r') as input_dataset:

        # flag_velsurfobs = "uvelsurfobs" in input_dataset.variables and "vvelsurfobs" in input_dataset.variables
        epsg = input_dataset.getncattr('epsg')
        pyproj_srs = input_dataset.getncattr('pyproj_srs')

        for name in ['uvelsurfobs']:
            # Read variable
            variable = input_dataset.variables[name]
            try:
                fillvalue = variable.get_fill_value()
            except:
                fillvalue = netCDF4.default_fillvals['f8']

            # Set values to zero where they are NaN
            original_data_v1 = np.array(variable[:])
            original_data_v2 = np.array(input_dataset.variables['vvelsurfobs'][:])

            # Remove zero values
            modified_data_v1 = np.where(original_data_v1 == 0,
                                        np.nan, original_data_v1)
            modified_data_v2 = np.where(original_data_v2 == 0,
                                        np.nan, original_data_v2)

            # Remove fillvalues
            modified_data_v1 = np.where(np.abs(modified_data_v1) == fillvalue,
                                        np.nan, modified_data_v1)
            modified_data_v2 = np.where(np.abs(modified_data_v2) == fillvalue,
                                        np.nan, modified_data_v2)

            # Compute magnitude of vecotr field
            modified_data = np.sqrt(modified_data_v1 ** 2 + modified_data_v2 ** 2)

            # Check if velocity input is meaningful (nonzero, not NaN, percentile check)
            if np.nansum(np.nansum(modified_data)) != 0 and not (
                    np.isnan(np.nansum(np.nansum(modified_data)))):
                if np.nanpercentile(np.abs(modified_data.flatten()), 99) > 10.0:
                    flag_velsurfobs = True

    # Load base parameters from params_inversion.yaml
    with open(params_inversion_path, 'r') as file:
        inv_params = yaml.safe_load(file)

    # Add velocity observation parameters if available
    if not flag_velsurfobs:
        inv_params['processes']['data_assimilation']['cost_list'] = ['thk', 'icemask', 'usurf']

    # Prepare inversion directory
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    # shutil.rmtree(inversion_dir, ignore_errors=True)
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)
    # Change to inversion directory and save params
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)
    # Write to YAML file
    with open(os.path.join('experiment', 'params_inversion.yaml'), 'w') as file:
        file.write("# @package _global_\n")
        yaml.dump(inv_params, file, sort_keys=False)

    # Run IGM inversion
    # Run the igm_run command
    import sys
    from igm.igm_run import main as igm_main
    sys.argv = ["igm_run", "+experiment=params_inversion"]
    igm_main()
    # subprocess.run(["igm_run", "+experiment=params"], check=True)

    with Dataset(os.path.join('outputs', 'output.nc'), 'a') as output:
        output.setncattr('epsg', epsg)
        output.setncattr('pyproj_srs', pyproj_srs)
    # TODO remove unnecessary files
    latest = max(Path("outputs").glob("*/*"), key=lambda p: p.stat().st_mtime)

    src = os.path.join(latest, 'iceflow-model')
    dst = os.path.join('outputs', 'iceflow-model')

    # Delete destination if it exists
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy source to destination
    shutil.copytree(src, dst)

    os.chdir(original_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates params.json for IGM inversion and runs igm_run.')
    parser.add_argument('--rgi_id_dir', type=str,
                        default="../../results/test_default/glaciers/RGI2000"
                                "-v7.0-G-11-01706/",
                        help='Path to Glacier dir with OGGM_shop output')
    parser.add_argument('--params_inversion_path', type=str,
                        default='../../experiments/test_default/params_inversion.yaml')
    args = parser.parse_args()
    main(args.rgi_id_dir, args.params_inversion_path)
