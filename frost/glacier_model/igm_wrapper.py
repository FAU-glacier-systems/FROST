#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import shutil
import json
import yaml
import os
import subprocess
from netCDF4 import Dataset
import numpy as np

# Suppress warnings and optimize TensorFlow execution

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error


def forward(exp, output1D, output2D_3D, member_id, smb_model, usurf, smb,
            year_start, year_end, workdir, climate_file):
    '''
    Runs a single forward model simulation for an ensemble member.

    Authors: Oskar Herrmann, Johannes J. Fürst

    Args:
        exp(str)              - intended experiment ('Calibration', 'Projection', ...)
        output1D(bool)        - if TRUE: standard time series output from IGM is provided
        output2D_3D(bool)     - if TRUE: 2D/3D output from IGM is provided
        member_id (int)       - ID of the ensemble member
        rgi_dir (str)         - Path to the glacier directory
        smb_model(str)        - chosen SMB model (ELA, TI, ...)
        usurf (ndarray)       - Initial surface elevation array
        smb (dict)            - Surface mass balance parameters:
                                  * 'ela' (float)      - Equilibrium line altitude
                                  * 'gradabl' (float)  - Ablation gradient (per km)
                                  * 'gradacc' (float)  - Accumulation gradient (per km)
        start_year (int)      - Year that simulation starts
        end_year (int)        - Year that simulaiton ends

    Returns:
        member_id (int)         - Ensemble member ID
        new_usurf (ndarray)     - Updated surface elevation
        new_smb (ndarray)       - Updated SMB values
    '''

    # Extract SMB parameters and convert gradients from m/km to m/m
    if str(smb_model) == 'ELA':
        ela = smb['ela']
        abl_grad = smb['abl_grad'] / 1000
        acc_grad = smb['acc_grad'] / 1000
    elif str(smb_model) == 'TI':
        melt_f = smb['melt_f']
        prcp_fac = smb['prcp_fac']
        temp_bias = smb['temp_bias']

    print(f'{exp}')
    print(f'Forward ensemble member {member_id} with SMB: {smb}')

    # Define input parameters for the ice flow model (IGM)
    igm_params = {
        "core": {
            "url_data": "",
        },
        "inputs": {
            "local": {
                "filename": "input.nc"
            }
        },
        "processes": {
            "iceflow": {
                "emulator": {
                    "pretrained": True,
                    "name": 'iceflow-model',
                    "retrain_freq": 0,
                }
            },
            "time": {
                "start": year_start,
                "end": year_end,
                "save": 1.,
            },
        }
    }

    if str(smb_model) == 'TI':

        igm_params['defaults'] = [{"override /inputs": ["local"]}]
        igm_params['defaults'] += [{'override /processes': ["clim_1D_3D", "smb_1D_3D", "iceflow", "time", "thk"]}]

        igm_params['processes']["clim_1D_3D"] = {
            "clim_trend_array": []
        }
        igm_params['processes']['clim_1D_3D']['clim_trend_array'] = [
            ['time', 'delta_temp', 'prec_scal'],
            [1900, 0.0, 1.0],
            [2020, 0.0, 1.0],
        ]
        igm_params['processes']['clim_1D_3D']['ref_period'] = [2000, 2019]
        igm_params['processes']['clim_1D_3D']['temp_bias'] = temp_bias
        igm_params['processes']['clim_1D_3D']['prcp_fac'] = prcp_fac
        igm_params['processes']['clim_1D_3D'][
            'prcp_gradient'] = 0.00035  # https://hess.copernicus.org/articles/24/5355/2020/
        igm_params['processes']['clim_1D_3D']['temp_default_gradient'] = -0.0065
        igm_params['processes']['clim_1D_3D']['update_freq'] = 1
        igm_params['processes']['clim_1D_3D']['file'] = climate_file

        igm_params['processes']["smb_1D_3D"] = {
            "temp_all_solid": 0.0
        }

        igm_params['processes']['smb_1D_3D']['temp_all_liquid'] = 2.0
        igm_params['processes']['smb_1D_3D']['temp_melt'] = -1.0
        igm_params['processes']['smb_1D_3D']['melt_f'] = melt_f
        igm_params['processes']['smb_1D_3D']['wat_density'] = 1000.0
        igm_params['processes']['smb_1D_3D']['ice_density'] = 910.0
        igm_params['processes']['smb_1D_3D']['melt_enhancer'] = 1
        igm_params['processes']['smb_1D_3D']['update_freq'] = 1

    elif str(smb_model) == 'ELA':

        igm_params['defaults'] = [{"override /inputs": ["local"]}]
        igm_params['defaults'] += [{'override /processes': ["smb_simple", "iceflow", "time", "thk"]}]

        igm_params['processes']["smb_simple"] = {
            "array": []
        }
        igm_params['processes']['smb_simple']['array'] = [
            ['time', 'gradabl', 'gradacc', 'ela', 'accmax'],
            [year_start, abl_grad, acc_grad, ela, 100],
            [year_end, abl_grad, acc_grad, ela, 100]
        ]

    if output1D and output2D_3D:
        igm_params['defaults'] += [{'override /outputs': ["write_ts", "write_ncdf"]}]
    elif output1D:
        igm_params['defaults'] += [{'override /outputs': ["write_ts"]}]
    elif output2D_3D:
        igm_params['defaults'] += [{'override /outputs': ["write_ncdf"]}]

    if output1D:
        igm_params['outputs'] = {
            "write_ts": {
                "output_file": []
            }
        }
        igm_params['outputs']['write_ts']['output_file'] = '../../output_ts.nc'
    if output2D_3D:
        igm_params['outputs'] = {
            "write_ncdf": {
                "output_file": [],
                "vars_to_save": [],
            }
        }
        if output1D:
            igm_params['outputs']['write_ts'] = {
                "output_file": []
            }
            igm_params['outputs']['write_ts']['output_file'] = '../../output_ts.nc'

        igm_params['outputs']['write_ncdf']['output_file'] = '../../output.nc'

        if str(smb_model) == 'TI':
            igm_params['outputs']['write_ncdf']['vars_to_save'] = ['topg',
                                                                   'usurf',
                                                                   'thk',
                                                                   'smb',
                                                                   'velbar_mag',
                                                                   'velsurf_mag',
                                                                   'slidingco',
                                                                   'divflux',
                                                                   'mean_temp',
                                                                   'sum_prec',
                                                                   'icemask',
                                                                   'icemask_init'
                                                                   ]
        else:
            igm_params['outputs']['write_ncdf']['vars_to_save'] = ['topg',
                                                                   'usurf',
                                                                   'thk',
                                                                   'smb',
                                                                   'velbar_mag',
                                                                   'velsurf_mag',
                                                                   'divflux',
                                                                   'slidingco',
                                                                   'icemask',
                                                                   'icemask_init'
                                                                   ]

    # Save simulation parameters as JSON

    def sanitize(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # convert np.float32 → float
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    params_path = os.path.join(workdir, 'experiment', 'params.yaml')
    if os.path.exists(params_path):
        os.remove(params_path)
    with open(params_path, 'w') as file:
        file.write('# @package _global_\n')
        yaml.dump(sanitize(igm_params), file)

    # Load input NetCDF file and update elevation values
    print("start igm run here", workdir)
    input_file = os.path.join(workdir, 'data', 'input.nc')
    with Dataset(input_file, 'r+') as input_dataset:
        if not "icemask_init" in input_dataset.variables:
            bedrock = input_dataset.variables['topg'][:]  # Read bedrock elevation
            thickness = usurf - bedrock  # Compute ice thickness

            # Update surface elevation and thickness
            input_dataset.variables['usurf'][:] = usurf
            input_dataset.variables['thk'][:] = thickness

            # Create init icemask
            var_in = input_dataset.variables['icemask']

            var_out = input_dataset.createVariable("icemask_init", var_in.datatype, ("y", "x",))
            var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})
            var_out[:] = input_dataset.variables['icemask'][:]

    # Run the Iceflow Glacier Model (IGM)
    subprocess.run(["igm_run", "+experiment=params"], cwd=workdir)
    # import sys
    # from igm.igm_run import main as igm_main
    # cwd = os.getcwd()
    # print(cwd)
    # os.chdir(workdir)
    # sys.argv = ["igm_run", "+experiment=params"]
    # igm_main()
    # os.chdir(cwd)

    if str(exp) == 'Calibration':
        # Read updated results from the output file
        output_file = os.path.join(workdir, 'outputs', 'output.nc')
        with Dataset(output_file, 'r') as new_ds:
            new_usurf = np.array(new_ds['usurf'][-1])  # Final surface elevation
            new_smb = np.mean(np.array(new_ds['smb']), axis=0)  # Final SMB values
            init_usurf = np.array(new_ds['usurf'][0])
            new_velsurf_mag = np.array(new_ds['velsurf_mag'][0])
            new_divflux = np.array(new_ds['divflux'][1])

        return member_id, new_usurf, new_smb, init_usurf, new_velsurf_mag, new_divflux
    else:
        return

