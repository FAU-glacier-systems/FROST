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


def extract_and_save(in_fpath,in_fname,out_fpath,out_fname,experiment_idx,member_idx):
    '''
    Extract single climate forcing from CORDEX netCDF.

    Authors: Johannes J. Fürst

    Args:
        in_fpath(str)              - file path to input netCDF
        in_fname(str)              - file name of input netCDF
        in_fpath(str)              - file path to input netCDF
        in_fname(str)              - file name of input netCDF
        experiment_idx()
        member_idx()
        

    Returns:
    '''

    infile  = os.path.join(in_fpath,in_fname)
    outfile = os.path.join(out_fpath, out_fname)

    with Dataset(infile, "r") as src:
        # Read dimensions
        time_len = len(src.dimensions["time"])

        # Extract slices for the given experiment and member
        time = src.variables["time"][:]
        temp = src.variables["temp"][experiment_idx, member_idx, :]
        temp_std = src.variables["temp_std"][experiment_idx, member_idx, :]
        prcp = src.variables["prcp"][experiment_idx, member_idx, :]
        experiment = src.variables["experiment"][experiment_idx]
        member = src.variables["member"][member_idx]

        with Dataset(os.path.join(out_fpath,"climate_historical_W5E5.nc")) as bc:
            time_W5E5 = bc.variables["time"][:]
            temp_W5E5 = bc.variables["temp"][:]
            prcp_W5E5 = bc.variables["prcp"][:]
            ref_hgt_W5E5 = bc.ref_hgt

        ## BIAS CORRECTION
        ## extract time baselines
        #print(np.asarray(temp_W5E5))
        time_units      = "days since 1801-01-01 00:00:00"
        time_units_W5E5 = time_units
        #time_units_W5E5 = time_W5E5.units
        #time_units      = src.variables["time_ts"].units

        ## reference period
        year_0 = 1970
        year_1 = 1999

        date_ref = np.datetime64("1800-01-01")
        date_0 = np.datetime64("1970-01-01")
        date_1 = np.datetime64("1999-12-31")
        date_ref = datetime.datetime(1800, 1, 1)
        date_0   = datetime.datetime(1970, 1, 1)
        date_1   = datetime.datetime(1999, 12, 31)
        #print('DATE', date_0, date_1)

        days_0 = (date_0 - date_ref).days
        days_1 = (date_1 - date_ref).days
        #print('NORMAL', days_0,days_1)

        mask = (time >= days_0) & (time <= days_1)
        mask_W5E5 = (time_W5E5 >= days_0) & (time_W5E5 <= days_1)
        #print(np.nansum(mask),np.nansum(mask_W5E5))

        # remove temperature bias
        dt_bias = np.nanmean(temp[mask]-temp_W5E5[mask_W5E5])
        temp    = temp - dt_bias

        # remove prcp bias
        prcp_fac = np.nansum(prcp_W5E5[mask_W5E5])/np.nansum(prcp[mask])
        prcp    = prcp*prcp_fac

        # projection period
        date_0   = datetime.datetime(2000, 1, 1)
        date_1   = datetime.datetime(2099, 12, 31)

        days_0 = (date_0 - date_ref).days
        days_1 = (date_1 - date_ref).days

        mask = (time >= days_0) & (time <= days_1)

        #if not np.isnan(np.sum(np.asarray(temp[365:]))) or np.isnan(np.sum(np.asarray(prcp[365:]))) :
        #if np.sum(np.isnan(np.asarray(temp[30*12+1:]))) == 0 and np.sum(np.isnan(np.asarray(prcp[30*12+1:]))) == 0: #433 (2006)
        if np.sum(np.isnan(np.asarray(temp[mask]))) == 0 and np.sum(np.isnan(np.asarray(prcp[mask]))) == 0: #433 (2006)
            # Open new file
            #print('Scenario: ',experiment_idx, ', Model : ', member_idx)
            with Dataset(outfile, "w", format="NETCDF4") as dst:
                # Copy global attributes
                dst.setncatts({attr: getattr(src, attr) for attr in src.ncattrs()})

                # Add new attribute
                #JJFdst.setncattr("ref_hgt", 2275.0)
                dst.setncattr("ref_hgt",ref_hgt_W5E5)

                # Define new dimension
                dst.createDimension("time", time_len)

                # Copy time variable
                time_var_in = src.variables["time"]
                time_var_out = dst.createVariable("time", time_var_in.datatype, ("time",))

                # Copy attributes of time variable
                time_var_out.setncatts({k: time_var_in.getncattr(k) for k in time_var_in.ncattrs()})
                time_var_out[:] = time_var_in[:] #JJF+ 18262

                # Define new variables with only time dimension
                for name, data in [("temp", temp), ("prcp", prcp), ("temp_std", temp_std)]:
                    var_in = src.variables[name]
                    #if name == "r" :
                    #    var_out = dst.createVariable("prcp", var_in.datatype, ("time",))
                    #else :
                    #    var_out = dst.createVariable(name, var_in.datatype, ("time",))
                    var_out = dst.createVariable(name, var_in.datatype, ("time",))
                    # Copy variable attributes
                    var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})
                    var_out[:] = data

                # temp_std
                #var_in  = src.variables["temp"]
                #var_out = dst.createVariable("temp_std", var_in.datatype, ("time",))
                #var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})
                #var_out[:] = 0.0*temp+5.0

                # Set flag if IGM needs to be run
                # (only necessary if climate data is complete)
                IGM_flag = 1
        else:
            IGM_flag = 0


        print(IGM_flag, experiment, member) 

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier projection experiments.')

    # Add arguments for parameters
    parser.add_argument('--in_fpath', type=str,
                        default='./',
                        help='File path of input netCDF.')

    parser.add_argument('--in_fname', type=str,
                        default='./',
                        help='File name of input netCDF.')

    parser.add_argument('--out_fpath', type=str,
                        default='./',
                        help='File path of output netCDF.')

    parser.add_argument('--out_fname', type=str,
                        default='./',
                        help='File name of output netCDF.')

    parser.add_argument('--experiment_idx', type=int,
                        default=0,
                        help='Experiment index.')

    parser.add_argument('--member_idx', type=int,
                        default=0,
                        help='Member index.')

    # Parse arguments
    args = parser.parse_args()

    member_id = extract_and_save(args.in_fpath, args.in_fname, args.out_fpath, args.out_fname, args.experiment_idx, args.member_idx)

