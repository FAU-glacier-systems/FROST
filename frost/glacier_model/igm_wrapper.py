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
import sys
from igm.igm_run import main as igm_main

# Suppress warnings and optimize TensorFlow execution

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error


def forward(exp, output1D, output2D_3D, member_id, rgi_dir, smb_model, usurf, smb,
            year_start, year_end):
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
        ## USER-DEFINED FUNCTION PATHS HAVE TO BE DEFINED FIRST
        ## - if activated please add '+' to 'override/inputs 
        ##
        ## absolute path option
        #igm_params['defaults'] = [{"//home/hpc/gwgi/gwgi17/projects/2025_eALPS/FROST/frost/igm-user/conf/processes@processes.clim_1D-3D": "clim_1D-3D"}]
        ## relative path option
        #igm_params['defaults'] = [{"/user/conf/processes@processes.clim_1D-3D": "clim_1D-3D"}]

        igm_params['defaults'] = [{"override /inputs": ["local"]}]
        igm_params['defaults'] += [{'override /processes': ["clim_1D_3D", "smb_1D_3D", "iceflow", "time", "thk"]}]

        igm_params['processes']["clim_1D_3D"] = {
                "clim_trend_array": []
            }
        igm_params['processes']['clim_1D_3D']['clim_trend_array']      = [
                                ['time', 'delta_temp', 'prec_scal'],
                                [1900, 0.0, 1.0],
                                [2020, 0.0, 1.0],
                            ]
        igm_params['processes']['clim_1D_3D']['ref_period']            = [2000, 2019]
        igm_params['processes']['clim_1D_3D']['temp_bias']             = temp_bias
        igm_params['processes']['clim_1D_3D']['prcp_fac']              = prcp_fac
        igm_params['processes']['clim_1D_3D']['prcp_gradient']         = 0.00035 # https://hess.copernicus.org/articles/24/5355/2020/
        igm_params['processes']['clim_1D_3D']['temp_default_gradient'] = -0.0065
        igm_params['processes']['clim_1D_3D']['update_freq']     = 1

        igm_params['processes']["smb_1D_3D"] = {
                "temp_all_solid": 0.0
            }

        igm_params['processes']['smb_1D_3D']['temp_all_liquid'] = 2.0
        igm_params['processes']['smb_1D_3D']['temp_melt']       = -1.0
        igm_params['processes']['smb_1D_3D']['melt_f']          = melt_f
        igm_params['processes']['smb_1D_3D']['wat_density']     = 1000.0
        igm_params['processes']['smb_1D_3D']['ice_density']     = 910.0
        igm_params['processes']['smb_1D_3D']['melt_enhancer']   = 1
        igm_params['processes']['smb_1D_3D']['update_freq']     = 1

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


    if output1D and output2D_3D :
        igm_params['defaults'] += [{'override /outputs': ["write_ts", "write_ncdf"]}]
    elif output1D :
        igm_params['defaults'] += [{'override /outputs': ["write_ts"]}]
    elif output2D_3D :
        igm_params['defaults'] += [{'override /outputs': ["write_ncdf"]}]

    if output1D :
        igm_params['outputs'] = {
            "write_ts": {
                "output_file": []
            }
            }
        igm_params['outputs']['write_ts']['output_file'] = '../../output_ts.nc'
    if output2D_3D :
        igm_params['outputs'] = {
            "write_ncdf": {
                "output_file": [],
                "vars_to_save": [],
            }
            }
        if output1D :
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

    # # Define TI specific input parameters (for separate JSON)
    # if str(SMB_model) == 'TI':
    #     oggm_TI_params = {
    #         # 'smb_oggm_TI_bias': 0.0,
    #         'smb_oggm_TI_temp_bias': temp_bias,
    #         'smb_oggm_TI_temp_default_gradient': -0.0065,
    #         'smb_oggm_TI_temp_all_solid': 0.0,
    #         'smb_oggm_TI_temp_all_liquid': 2.0,
    #         'smb_oggm_TI_temp_melt': -1.0,
    #         'smb_oggm_TI_prcp_fac': prcp_fac,
    #         'smb_oggm_TI_prcp_gradient': 0.00035,
    #         # https://hess.copernicus.org/articles/24/5355/2020/
    #         'smb_oggm_TI_melt_f': melt_f
    #     }

    # Create directory for the ensemble member
    if str(exp) == 'Projection':
        member_dir = os.path.join(rgi_dir, 'Projection', f'Member_{member_id}')
    elif str(exp) == 'Calibration':
        member_dir = os.path.join(rgi_dir, 'Ensemble', f'Member_{member_id}')
    else:
        member_dir = os.path.join(rgi_dir, 'Ensemble', f'Member_{member_id}')

    # Save simulation parameters as JSON

    def sanitize(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # convert np.float32 → float
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    params_path = os.path.join(member_dir, 'experiment', 'params.yaml')
    if os.path.exists(params_path):
        os.remove(params_path)
    with open(params_path, 'w') as file:
        file.write('# @package _global_\n')
        yaml.dump(sanitize(igm_params), file)

    # if str(SMB_model) == 'TI':
    #     # Save SMB parameters for simulation as SMB-specific JSON
    #     with (open(os.path.join(member_dir, 'experiment', 'params_smb_TI.yaml'),
    #                'w') as file):
    #         file.write('# @package _global_\n')
    #         yaml.dump(oggm_TI_params, file, indent=4, separators=(',', ': '))
    #
    # Load input NetCDF file and update elevation values
    input_file = os.path.join(member_dir, 'data', 'input.nc')
    with Dataset(input_file, 'r+') as input_dataset:
        if not "icemask_init" in input_dataset.variables :
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
    #
    # if str(SMB_model) == 'TI':
    #     # Copy and assemble TI model routines
    #     src = os.path.join('./Scripts/Process/', 'clim_1D-3D.py')
    #     dst = os.path.join(member_dir, 'clim_1D-3D.py')
    #
    #     # Check the operating system and use the respective command
    #     if os.name == 'nt':  # Windows
    #         cmd = f"copy '{src}' '{dst}'"
    #     else:  # Unix/Linux
    #         cmd = f"cp '{src}' '{dst}'"
    #
    #     # Copy File
    #     os.system(cmd)
    #
    #     src = os.path.join(rgi_dir, 'climate_historical.nc')
    #     dst = os.path.join(member_dir, 'climate_historical.nc')
    #
    #     # Check the operating system and use the respective command
    #     if os.name == 'nt':  # Windows
    #         cmd = f"copy '{src}' '{dst}'"
    #     else:  # Unix/Linux
    #         cmd = f"cp '{src}' '{dst}'"
    #
    #     # Copy File
    #     os.system(cmd)
    #
    #     src = os.path.join('./Scripts/Process/', 'smb_oggm_TI_local.py')
    #     dst = os.path.join(member_dir, 'smb_oggm_TI_local.py')
    #
    #     # Check the operating system and use the respective command
    #     if os.name == 'nt':  # Windows
    #         cmd = f"copy '{src}' '{dst}'"
    #     else:  # Unix/Linux
    #         cmd = f"cp '{src}' '{dst}'"
    #
    #     # Copy File
    #     os.system(cmd)

    # Run the Iceflow Glacier Model (IGM)
    subprocess.run(["igm_run", "+experiment=params"], cwd=member_dir)

    if str(exp) == 'Calibration':
        # Read updated results from the output file
        output_file = os.path.join(member_dir,'outputs','output.nc')
        with Dataset(output_file, 'r') as new_ds:
            new_usurf = np.array(new_ds['usurf'][-1])  # Final surface elevation
            new_smb = np.mean(np.array(new_ds['smb']), axis=0)  # Final SMB values
            init_usurf = np.array(new_ds['usurf'][0])
            new_velsurf_mag = np.array(new_ds['velsurf_mag'][0])
            new_divflux = np.array(new_ds['divflux'][1])

        return member_id, new_usurf, new_smb, init_usurf, new_velsurf_mag, new_divflux
    else:
        return


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier projection experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str,
                        default='RGI2000-v7.0-G-11-01706',
                        help='RGI ID of the glacier for the model.')

    # Add arguments for parameters
    parser.add_argument('--JSON_path', type=str,
                        default='result.json',
                        help='Provide file path to FROST ensemble results (currently TI approach).')

    # Add arguments for parameters
    #
    parser.add_argument('--member_id', type=str,
                        default='mean',
                        help='Provide ensemble ID number or specify <mean> as ensemble parameter average.')

    parser.add_argument('--start_year', type=int,
                        default='2000',
                        help='Start year of forward simulation. Default: 2000')

    parser.add_argument('--end_year', type=int,
                        default='2100',
                        help='End year of forward simulation. Default: 2100')

    parser.add_argument('--flag_3D_output', type=bool,
                        default=False,
                        help='Decide wether 3D output is written.')

    # Parse arguments
    args = parser.parse_args()
    print(args)

    # Projection run setting
    exp = 'Projection'
    output1D = True

    if args.start_year == 2000 :
        output2D_3D = True
    else :
        output2D_3D = False

    if args.flag_3D_output == True:
        output2D_3D = True

    ## Define projection length
    #year_interval = int(args.end_year-args.start_year)

    # Define SMB model
    smb_model = 'TI'

    # Simulation Path
    rgi_id_dir = os.path.join('..', 'data', 'results', 'alps_TI_projections', 'glaciers', args.rgi_id)

    # Define member_id
    member_id = args.member_id

    # Create projection directory
    member_dir = os.path.join(rgi_id_dir, exp, 'Member_' + str(member_id))
    os.makedirs(member_dir, exist_ok=True)

    # Load initial surface elevation
    print(os.getcwd())
    inversion_dir = os.path.join(rgi_id_dir, 'Preprocess','outputs')
    geology_file = os.path.join(inversion_dir, 'output.nc')
    with Dataset(geology_file, 'r') as geology_dataset:
        usurf_init = np.array(geology_dataset['usurf'])

    # Copy input files
    # Copy geology file as the initial input.nc
    os.makedirs(os.path.join(member_dir, 'data'), exist_ok=True)
    if args.start_year == 2000 :
        shutil.copy2(geology_file, os.path.join(member_dir, 'data', 'input.nc'))
    if args.start_year == 2020 :
        if str(member_id) == "mean":
            shutil.copy2(os.path.join(member_dir, 'outputs', 'output_W5E5_'+str(member_id)+'.nc'), os.path.join(member_dir, 'data', 'input.nc'))
        else:
            shutil.copy2(os.path.join(member_dir, 'outputs', 'output_W5E5_best.nc'), os.path.join(member_dir, 'data', 'input.nc'))

    # Copy iceflow-model directory
    member_iceflow_dir = os.path.join(member_dir, 'iceflow-model')
    shutil.rmtree(member_iceflow_dir, ignore_errors=True)
    shutil.copytree(os.path.join(inversion_dir, 'iceflow-model'), member_iceflow_dir)

    # Create experiment folder
    os.makedirs(os.path.join(member_dir, 'experiment'), exist_ok=True)

    # Load glacier-specific parameters
    # calibration version
    # cal_version = 'v01'
    # params_file_path = os.path.join('./Experiments',args.rgi_id,'Ensemble_'+cal_version+'_'+args.rgi_id+'_50.0_6_6_50m',
    #                                    'result.json')
    params_file_path = args.JSON_path

    with open(params_file_path, 'r') as file:
        params = json.load(file)
        final_mean = params['final_mean']
        final_std = params['final_std']

        final_ensemble = params['final_ensemble']

    if str(member_id) == 'mean':
        # Create SMB dictionary
        new_smb = []
        new_smb = {
            'temp_bias': final_mean[2],
            'melt_f': final_mean[0],
            'prcp_fac': final_mean[1]
        }
    else:
        print(final_ensemble)
        new_smb = []
        new_smb = {
            'temp_bias': final_ensemble[int(member_id)][2],
            'melt_f': final_ensemble[int(member_id)][0],
            'prcp_fac': final_ensemble[int(member_id)][1]
        }

    # Saving JSON file
    SMB = {
        "member_id": member_id,
        "temp_bias": new_smb["temp_bias"],
        "melt_f": new_smb["melt_f"],
        "prcp_fac": new_smb["prcp_fac"]
    }

    # Write to file
    outputJSON=os.path.join(rgi_id_dir,"selectedSMBmember.json")
    with open(outputJSON, "w") as f:
        json.dump(SMB, f, indent=4)

    # Load initial surface elevation
    #inversion_dir = os.path.join('.', rgi_id_dir, 'Inversion')
    #geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
    #with Dataset(geology_file, 'r') as geology_dataset:
    #    usurf_init = np.array(geology_dataset['usurf'])

    print('IGM_wrapper.forward')
    print('member_id', member_id)
    print('rgi_id_dir', rgi_id_dir)
    print('usurf', np.shape(usurf_init))
    print('smb', new_smb)
    #print('year_interval', year_interval)
    print('start/end year :', args.start_year, args.end_year)
    member_id = forward(exp, output1D, output2D_3D, member_id, rgi_id_dir, smb_model,
                        usurf_init, new_smb, args.start_year, args.end_year)
