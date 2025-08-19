#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
from pathlib import Path
import os
import shutil
import yaml


def main(rgi_id_dir, params_inversion_path):
    """
    Generates params.json for IGM inversion and runs igm_run.

    Args:
        rgi_id (str) - Glacier RGI ID
    """

    # Check if velocity observations are available
    flag_velsurfobs = False
    try:
        # Define input and output file names
        input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input_saved.nc')

        # Open the input netCDF file in read mode
        with Dataset(input_file, 'r') as src:


            # try reading velocity data
            var1 = input_dataset.variables['uvelsurfobs']
            var2 = input_dataset.variables['vvelsurfobs']

        flag_velsurfobs = True
    except:
        flag_velsurfobs = False

    # Prepare inversion directory
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    # shutil.rmtree(inversion_dir, ignore_errors=True)
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)

    #shutil.copy(params_inversion_path, exp_dir)

    # Change to inversion directory and save params
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)

    # Set inversion parameters
    # depend on availaibity of velocity observations
    inv_params = dict()
    inv_params["core"] = {
            "url_data": ""
        }
    inv_params["defaults"] = [
            {"override /inputs": ["load_ncdf"]},
            {"override /processes": ["data_assimilation", "iceflow"]},
            {"override /outputs": []}
            ]

    inv_params["inputs"] = {
            "load_ncdf": {"input_file": "input.nc"}
            }

    iceflow_params = dict()
    iceflow_params["physics"] = {
                    "init_slidingco": 0.045
                    }
    iceflow_params["emulator"] = {
                    "save_model": True,
                    "retrain_freq": 1
                    }

    DA_output_params = dict()
    DA_output_params["save_result_in_ncdf"] = "../../output.nc"
    if flag_velsurfobs :
        DA_output_params["vars_to_save"] = [
                            "usurf", "topg", "thk", "slidingco",
                            "velsurf_mag", "velsurfobs_mag", "divflux", "icemask",
                            "arrhenius", "thkobs", "dhdt"
                            ]
    else:
        DA_output_params["vars_to_save"] = [
                            "usurf", "topg", "thk", "slidingco",
                            "velsurf_mag", "divflux", "icemask",
                            "arrhenius", "thkobs", "dhdt"
                            ]
    DA_output_params["plot2d_live"] = False
    DA_output_params["plot2d"] = False

    DA_params = dict()
    DA_params = {"output":DA_output_params}

    DA_params["control_list"] = ["thk"]
    if flag_velsurfobs :
        DA_params["cost_list"] = ["velsurf", "thk", "icemask"]
        DA_params["fitting"] = {
                        "velsurfobs_std": 0.25,
                        "thkobs_std": 10,
                        }
    else:
        DA_params["cost_list"] = ["thk", "icemask"]
        DA_params["fitting"] = {
                        "thkobs_std": 10,
                        }
    DA_params["optimization"] = {
                    "retrain_iceflow_model": True,
                    "nbitmax": 500,
                }
    DA_params["regularization"] = {
                    "thk": 1.0,
                    "smooth_anisotropy_factor": 1.0,
                    "convexity_weight": 1.0,
                }

    inv_params["processes"] = {"iceflow":iceflow_params, "data_assimilation":DA_params}

    inv_params["outputs"] = {}

    # Write to YAML file
    with open(os.path.join('experiment', 'params_inversion.yaml'), 'w') as file:
        file.write("# @package _global_\n")
        yaml.dump(inv_params, file, sort_keys=False)

    # Run IGM inversion
    # Run the igm_run command
    import sys
    from igm.igm_run import main as igm_main
    sys.argv = ["igm_run", "+experiment=params_inversion"]
    igm_main()
    # subprocess.run(["igm_run", "+experiment=params"], check=True)
    # TODO remove unnecessary files
    latest = max(Path("outputs").glob("*/*"), key=lambda p: p.stat().st_mtime)

    src = os.path.join(latest, 'iceflow-model')
    dst = os.path.join('outputs', 'iceflow-model')

    # Delete destination if it exists
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Copy source to destination
    shutil.copytree(src, dst)

    os.chdir(original_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates params.json for IGM inversion and runs igm_run.')
    parser.add_argument('--rgi_id_dir', type=str,
                        default="../../results/test_default/glaciers/RGI2000"
                                "-v7.0-G-11-01706/",
                        help='Path to Glacier dir with OGGM_shop output')
    parser.add_argument('--params_inversion_path', type=str,
                        default='../../experiments/test_default/params_inversion.yaml')
    args = parser.parse_args()
    main(args.rgi_id_dir, args.params_inversion_path)
