#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import shutil
import os
import subprocess
from netCDF4 import Dataset
import datetime
import numpy as np
import sys
import json
import re
from pathlib import Path


def select_SMB_member(cal_fpath,cal_fname,mod_fpath,mod_fname,obs_fpath, obs_fname, option):
    '''
    Select ensemble member after calibration.

    OPTION: Best performance against observation

    Authors: Johannes J. Fürst

    Args:
        cal_fpath(str)              - file path to calibration file JSON
        cal_fname(str)              - file name of calibration file JSON
        mod_fpath(str)              - file path to model output netCDF
        mod_fname(str)              - file name of model output netCDF
        obs_fpath(str)              - file path to observation netCDF
        obs_fname(str)              - file name of observation netCDF
        option(str)                 - options for SMB selection
                                      default 'best'
        

    Returns: index
    '''

    # Open JSON file with calibrate SMB parameters
    params_file_path = os.path.join(cal_fpath,cal_fname)

    with open(params_file_path, 'r') as file:
        params = json.load(file)
        final_mean = params['final_mean']
        final_std = params['final_std']

        final_ensemble = params['final_ensemble']

    smb_mean = []
    smb_mean = {
        'temp_bias': final_mean[2],
        'melt_f': final_mean[0],
        'prcp_fac': final_mean[1]
    }
   
    #new_smb = []
    #new_smb = {
    #    'temp_bias': final_ensemble[int(member_id)][2],
    #    'melt_f': final_ensemble[int(member_id)][0],
    #    'prcp_fac': final_ensemble[int(member_id)][1]
    #}


    obsfile  = os.path.join(obs_fpath,obs_fname)

    # Open observation NetCDF file
    with Dataset(obsfile) as ds_obs:

        # extract thickness field
        dhdt_all = ds_obs.variables["dhdt"][:]
        dhdt = dhdt_all[-1,:,:]

        # extract thickness field
        usurf_obs = ds_obs.variables["usurf"][:]
        icemask_obs = ds_obs.variables["icemask"][0,:,:]
        icemask_obs[icemask_obs==0]=np.nan

        # compute volume difference
        vol_diff_obs = np.nanmean(np.nanmean(icemask_obs*(usurf_obs[-1,:,:]-usurf_obs[0,:,:])))/20

    if option == "best" :
        # Recursively search for all files named 'output.nc'
        base_dir=Path(mod_fpath)
        #base_dir=os.path.join(mod_fpath)
        files = list(base_dir.rglob("output.nc"))

        mm = 0
        for afile in files :
            with Dataset(afile) as ds_mod:

                # determine member ID
                match = re.search(r"Member_(\d+)", str(afile))
                if match:
                    member_num = int(match.group(1))
                else:
                    #print("No member number found.")
                    member_num = int(-1)

                # extract thickness field
                usurf = ds_mod.variables["usurf"][:]

                # compute volume difference
                value = np.nanmean(np.nanmean(icemask_obs*(usurf[-1,:,:]-usurf[0,:,:])))/20

                # select best member in terms of SMB values
                if mm == 0:
                    vol_diff_mod = value
                    index        = member_num
                else :
                    if np.abs(value-vol_diff_obs) < np.abs(vol_diff_mod-vol_diff_obs):
                        vol_diff_mod = value
                        index        = str(member_num)
                mm += 1
    else:
        index="mean"

    print(index) 

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier projection experiments.')

    # Add arguments for parameters
    parser.add_argument('--cal_fpath', type=str,
                        default='./',
                        help='File path to calibration file JSON.')

    parser.add_argument('--cal_fname', type=str,
                        default='./',
                        help='File name of calibration file JSON.')

    parser.add_argument('--mod_fpath', type=str,
                        default='./',
                        help='File path to model-results netCDF.')

    parser.add_argument('--mod_fname', type=str,
                        default='./',
                        help='File name of model-results netCDF.')

    parser.add_argument('--obs_fpath', type=str,
                        default='./',
                        help='File path to observation netCDF.')
    
    parser.add_argument('--obs_fname', type=str,
                        default='./',
                        help='File name of observation netCDF.')

    parser.add_argument('--option', type=str,
                        default='./',
                        help='Option of SMB selection.')



    # Parse arguments
    args = parser.parse_args()

    INDEX = select_SMB_member(args.cal_fpath, args.cal_fname, args.mod_fpath, args.mod_fname, args.obs_fpath, args.obs_fname, args.option)

