#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import json
import subprocess
import os
import shutil

# Function to handle the main logic
def main(rgi_id):
    # Define the params to be saved in params.json
    json_file_path = os.path.join('..','..','Experiments', rgi_id,
                                   'params_inversion.json')
    with open(json_file_path, 'r') as file:
        params = json.load(file)

    # Define the path using os.path.join
    inversion_dir = os.path.join('..','..', 'Data', 'Glaciers', rgi_id, 'Inversion')

    # Check if the directory exists, and create it if not
    if  os.path.exists(inversion_dir):
        shutil.rmtree(inversion_dir)
    os.makedirs(inversion_dir)


    # Change directory to the correct location
    os.chdir(inversion_dir)

    # Write the params dictionary to the params.json file
    with open('params.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)

    # Run the igm_run command
    subprocess.run(['igm_run', '--param_file', 'params.json'])


if __name__ == '__main__':
    # Parse command-line argumentsds
    parser = argparse.ArgumentParser(
        description='This script generates params.json for igm inversion and '
                    'runs igm_run.')

    # Add argument for RGI ID
    parser.add_argument('--rgi_id', type=str,
                        default='RGI2000-v7.0-G-11-01706',
                        help='The RGI ID of the glacier to be calibrated '
                             '(default: RGI2000-v7.0-G-11-01706).')


    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.rgi_id)
