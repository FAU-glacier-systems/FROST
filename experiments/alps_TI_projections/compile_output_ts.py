#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import shutil
import os
import subprocess
from netCDF4 import Dataset
import numpy as np
import sys


def compile_output_ts(counter,blueprint_fpath,blueprint_fname,in_fpath,in_fname,out_fpath,out_fname,experiment_idx,member_idx):
    '''
    Extract single climate forcing from CORDEX netCDF.

    Authors: Johannes J. Fürst

    Args:
        blueprint_fpath(str)       - file path to blueprint netCDF
        blueprint_fname(str)       - file name of blueprint netCDF
        in_fpath(str)              - file path to input netCDF
        in_fname(str)              - file name of input netCDF
        out_fpath(str)             - file path to output netCDF
        out_fname(str)             - file name of putput netCDF
        experiment_idx()
        member_idx()
        

    Returns:
    '''

    bprfile = os.path.join(blueprint_fpath,blueprint_fname)
    infile  = os.path.join(in_fpath,in_fname)
    outfile = os.path.join(out_fpath, out_fname)

    if counter == 0 :
        # open blueprint file for reading
        with Dataset(bprfile, "r") as bpr:

            # remove pre-existing outputfile
            if os.path.isfile(outfile) :
                os.remove(outfile)

            # open output file for writing
            with Dataset(outfile, "w", format="NETCDF4") as dst:

                # open input file holding relevant variables
                with Dataset(infile, "r", format="NETCDF4") as src:
                    # Copy dimensions
                    for name, dim in bpr.dimensions.items():
                        if name == "time":
                            continue
                        dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
                        in_dim = bpr.variables[name]
                        out_dim = dst.createVariable(name, in_dim.datatype, in_dim.dimensions)
                        out_dim.setncatts({k: in_dim.getncattr(k) for k in in_dim.ncattrs()})
                        out_dim[:] = in_dim[:]

                    # Copy dimensions from input file for 'time'
                    if "time" in src.dimensions:
                        dim = src.dimensions["time"]
                        dst.createDimension("time", (len(dim) if not dim.isunlimited() else None))

                    # Copy 'time' variable from input
                    if "time" in src.variables:
                        in_time = src.variables["time"]
                        out_time = dst.createVariable("time", in_time.datatype, in_time.dimensions)
                        out_time.setncatts({k: in_time.getncattr(k) for k in in_time.ncattrs()})
                        out_time[:] = in_time[:]

                    
                    # Copy global attributes
                    dst.setncatts({attr: getattr(bpr, attr) for attr in bpr.ncattrs()})

                    dst.setncattr("number_valid_results", counter)

                    #for name, data in bpr.variables.items():
                    #    # Get reference variable (temp) for shape/dims
                    #    var_in = bpr.variables[name]

                    #    # Create variable
                    #    var_out = dst.createVariable(name, var_in.datatype, var_in.dimensions)

                    #    # copy attributes
                    #    var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})

                    #    # write data
                    #    var_out = data


                    # Reference variable from blueprint to copy dims (but swap time)
                    temp_var = bpr.variables["temp"]
                    new_dims = tuple("time" if d == "time" else d for d in temp_var.dimensions)


                    for src_name, src_var in src.variables.items():

                        if not src_name == "time" :

                            # Create new variable with same datatype + dimensionsI
                            new_var = dst.createVariable(src_name, src_var.datatype, temp_var.dimensions)

                            # Copy attributes
                            new_var.setncatts({k: src_var.getncattr(k) for k in src_var.ncattrs()})

                            # Write data
                            new_var[:] = np.zeros((np.shape(temp_var)[0],np.shape(temp_var)[1],np.shape(src_var)[0]))*np.nan
                            
                            new_var[experiment_idx,member_idx,:] = np.asarray(src_var)

    else:

        if os.path.isfile(outfile) :

            # open output file for writing
            #with Dataset(outfile, "a", format="NETCDF4") as dst:
            dst = Dataset(outfile, "a")

            dst.setncattr("number_valid_results", counter)

            # open input file holding relevant variables
            with Dataset(infile, "r", format="NETCDF4") as src:

                for src_name, src_var in src.variables.items():

                    if not src_name == "time" :

                        # Get pre-existing variable
                        new_var = dst.variables[src_name]

                        # Append at right position
                        new_var[experiment_idx,member_idx,:] = np.asarray(src_var) 


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier projection experiments.')

    # Add arguments for parameters
    parser.add_argument('--counter', type=int,
                        default=0,
                        help='Counting index, determines if output netCDF has to be created (0) or already exists (1). Defaul: 0.')

    parser.add_argument('--blueprint_fpath', type=str,
                        default='./',
                        help='File path of blueprint netCDF.')

    parser.add_argument('--blueprint_fname', type=str,
                        default='./',
                        help='File name of blueprint netCDF.')

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

    member_id = compile_output_ts(args.counter, args.blueprint_fpath, args.blueprint_fname, args.in_fpath, args.in_fname, args.out_fpath, args.out_fname, args.experiment_idx, args.member_idx)

