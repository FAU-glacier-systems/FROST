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


def check_volume(in_fpath,in_fname):
    '''
    Extract single climate forcing from CORDEX netCDF.

    Authors: Johannes J. Fürst

    Args:
        in_fpath(str)              - file path to input netCDF
        in_fname(str)              - file name of input netCDF
        

    Returns: IMG_flag
    '''

    infile  = os.path.join(in_fpath,in_fname)

    with Dataset(infile, "r") as src:
        # Read dimensions
        time_len = len(src.dimensions["time"])

        # Extract slices for the given experiment and member
        time = src.variables["time"][:]
        vol = src.variables["vol"][:]

        if vol[-1] > 0 :
            VOL_flag = 1
        else:
            VOL_flag = 0


        print(VOL_flag) 

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

    # Parse arguments
    args = parser.parse_args()

    VOL_flag = check_volume(args.in_fpath, args.in_fname)

