#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import json
import os
import subprocess
from netCDF4 import Dataset
import numpy as np

# Suppress warnings and optimize TensorFlow execution
os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def forward(member_id, rgi_dir, usurf, smb, year_interval):
    """
    Runs a single forward model simulation for an ensemble member.

    Authors: Oskar Herrmann

    Args:
        member_id (int)       - ID of the ensemble member
        rgi_dir (str)         - Path to the glacier directory
        usurf (ndarray)       - Initial surface elevation array
        smb (dict)            - Surface mass balance parameters:
                                  * 'ela' (float)      - Equilibrium line altitude
                                  * 'gradabl' (float)  - Ablation gradient (per km)
                                  * 'gradacc' (float)  - Accumulation gradient (per km)
        year_interval (int)   - Simulation duration in years

    Returns:
        member_id (int)         - Ensemble member ID
        new_usurf (ndarray)     - Updated surface elevation
        new_smb (ndarray)       - Updated SMB values
    """

    # Extract SMB parameters and convert gradients from m/km to m/m
    ela = smb['ela']
    grad_abl = smb['gradabl'] / 1000
    grad_acc = smb['gradacc'] / 1000

    print(f"Forward ensemble member {member_id} with SMB: {smb}")

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
        "iflo_emulator": "iceflow-model",
        "lncd_input_file": "input.nc",
        "wncd_output_file": "output.nc",
        "wncd_vars_to_save": ["topg","usurf",  "thk", "smb","velbar_mag",
            "velsurf_mag", "uvelsurf","vvelsurf","wvelsurf", "divflux"
        ],
        "time_start": 0,
        "time_end": year_interval,
        "iflo_retrain_emulator_freq": 0,
    }

    # Create directory for the ensemble member
    member_dir = os.path.join(rgi_dir, 'Ensemble', f'Member_{member_id}')

    # Save simulation parameters as JSON
    with open(os.path.join(member_dir, "params.json"), 'w') as file:
        json.dump(igm_params, file, indent=4, separators=(',', ': '))

    # Load input NetCDF file and update elevation values
    input_file = os.path.join(member_dir, "input.nc")
    with Dataset(input_file, 'r+') as input_dataset:
        bedrock = input_dataset.variables['topg'][:]  # Read bedrock elevation
        thickness = usurf - bedrock  # Compute ice thickness

        # Update surface elevation and thickness
        input_dataset.variables['usurf'][:] = usurf
        input_dataset.variables['thk'][:] = thickness

    # Run the Iceflow Glacier Model (IGM)
    subprocess.run(["igm_run"], cwd=member_dir)

    # Read updated results from the output file
    output_file = os.path.join(member_dir, "output.nc")
    with Dataset(output_file, 'r') as new_ds:
        new_usurf = np.array(new_ds['usurf'][-1]) # Final surface elevation
        new_smb = np.mean(np.array(new_ds['smb']), axis=0) # Final SMB values
        init_usurf = np.array(new_ds['usurf'][0])
        new_velsurf_mag = np.array(new_ds['velsurf_mag'][1])
        new_divflux = np.array(new_ds['divflux'][1])

    return member_id, new_usurf, new_smb, init_usurf, new_velsurf_mag, new_divflux
