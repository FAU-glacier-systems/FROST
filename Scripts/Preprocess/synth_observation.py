#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import json
import os
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np


def main(rgi_id):
    print(os.getcwd())
    params_file = os.path.join('..','..','Experiments', rgi_id,
                                 'params_calibration.json')
    with open(params_file, 'r') as file:
        params = json.load(file)

    rgi_dir = os.path.join('..','..','Data', 'Glaciers', rgi_id)
    generate_observations(rgi_dir, params['reference_smb'])


def generate_observations(rgi_dir, smb, year_interval=20):
    # Extract SMB parameters and convert gradients from m/km to m/m
    ela = smb['ela']
    grad_abl = smb['gradabl'] / 1000
    grad_acc = smb['gradacc'] / 1000

    # Define input parameters for the ice flow model (IGM)
    igm_params = {
        "modules_preproc": ["load_ncdf"],
        "modules_process": ["smb_simple", "iceflow", "time", "thk"],
        "modules_postproc": ["write_ncdf", "print_info"],
        "smb_simple_array": [
            ["time", "gradabl", "gradacc", "ela", "accmax"],
            [0, grad_abl, grad_acc, ela, 100],
            [year_interval, grad_abl, grad_acc, ela, 100]
        ],
        "iflo_emulator": "../Inversion/iceflow-model",
        "lncd_input_file": "../Inversion/geology-optimized.nc",
        "wncd_output_file": "observations_synth.nc",
        "wncd_vars_to_save": ["topg", "usurf", "thk", "smb", "velbar_mag",
                              "velsurf_mag", "uvelsurf", "vvelsurf", "wvelsurf",
                              "divflux", "icemask"],
        "time_start": 0,
        "time_save": 20,
        "time_end": year_interval,
        "iflo_retrain_emulator_freq": 0,
    }

    # Create directory for the ensemble member
    igm_dir = os.path.join(rgi_dir, 'SyntheticData')
    os.makedirs(igm_dir, exist_ok=True)

    # Save simulation parameters as JSON
    with open(os.path.join(igm_dir, "params.json"), 'w') as file:
        json.dump(igm_params, file, indent=4, separators=(',', ': '))

    # Run the Iceflow Glacier Model (IGM)
    subprocess.run(["igm_run"], cwd=igm_dir)

    from netCDF4 import Dataset

    # Open the file in append mode
    with Dataset(os.path.join(igm_dir, 'observations_synth.nc'), 'r+') as nc:
        # Original variable name
        original_var = nc.variables['icemask']  # example variable

        # Create a new variable with a different name, same dtype and dimensions
        new_var = nc.createVariable('usurf_err', original_var.datatype,
                                    original_var.dimensions)

        # Copy the data
        new_var[:] = original_var[:]

        # (Optional) Copy attributes
        for attr_name in original_var.ncattrs():
            setattr(new_var, attr_name, getattr(original_var, attr_name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help='RGI ID of the glacier for the model.')
    args = parser.parse_args()
    main(args.rgi_id)
