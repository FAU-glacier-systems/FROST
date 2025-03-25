#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import json
import subprocess
import os
import shutil

def main(rgi_id):
    """
    Generates params.json for IGM inversion and runs igm_run.

    Args:
        rgi_id (str) - Glacier RGI ID
    """

    # Load parameters
    json_file_path = os.path.join('..', '..', 'Experiments', rgi_id, 'params_inversion.json')
    with open(json_file_path, 'r') as file:
        params = json.load(file)

    # Prepare inversion directory
    inversion_dir = os.path.join('..', '..', 'Data', 'Glaciers', rgi_id, 'Inversion')
    shutil.rmtree(inversion_dir, ignore_errors=True)
    os.makedirs(inversion_dir)

    # Change to inversion directory and save params
    os.chdir(inversion_dir)
    with open('params.json', 'w') as json_file:
        json.dump(params, json_file, indent=4)

    # Run IGM inversion
    subprocess.run(['igm_run', '--param_file', 'params.json'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates params.json for IGM inversion and runs igm_run.')
    parser.add_argument('--rgi_id', type=str, default='RGI2000-v7.0-G-11-01706',
                        help='Glacier RGI ID (default: RGI2000-v7.0-G-11-01706)')
    args = parser.parse_args()
    main(args.rgi_id)
