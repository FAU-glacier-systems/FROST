#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
from pathlib import Path
import os
import shutil
import yaml


def main(rgi_id_dir):
    """
    Generates params.json for IGM inversion and runs igm_run.

    Args:
        rgi_id (str) - Glacier RGI ID
    """

    # Prepare inversion directory
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    # shutil.rmtree(inversion_dir, ignore_errors=True)
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)

    # Change to inversion directory and save params
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)

    params = {
        "core": {
            "url_data": ""
        },
        "defaults": [
            {"override /inputs": ["load_ncdf"]},
            {"override /processes": ["data_assimilation", "iceflow"]},
            {"override /outputs": []}
        ],
        "inputs": {
            "load_ncdf": {
                "input_file": "input.nc"
            }
        },
        "processes": {
            "iceflow": {
                "physics": {
                    "init_slidingco": 0.045
                },
                "emulator": {
                    "save_model": True
                }
            },
            "data_assimilation": {
                "output": {
                    "save_result_in_ncdf": "../../output.nc",
                    "vars_to_save": [
                        "usurf", "topg", "thk", "slidingco",
                        "velsurf_mag", "velsurfobs_mag", "divflux", "icemask",
                        "arrhenius", "thkobs", "dhdt"
                    ],
                    "plot2d_live": False,
                    "plot2d": False
                },
                "control_list": ["thk", "slidingco", "usurf"],
                "cost_list": ["velsurf", "thk", "icemask", "usurf", "divfluxfcz"],
                "optimization": {
                    "retrain_iceflow_model": True,
                    "nbitmax": 1000,

                },
                "fitting": {
                    "usurfobs_std": 0.3,
                    "velsurfobs_std": 1,
                    "uniformize_thkobs": True,
                    "thkobs_std": 1,
                    "divfluxobs_std": 0.1
                },
                "regularization": {
                    "thk": 1.0,
                    "slidingco": 1.0e6,
                    "smooth_anisotropy_factor": 1.0,
                    "convexity_weight": 0.002,
                }
            }
        },
        "outputs": {}
    }

    # Write to YAML file
    with open(os.path.join('experiment', 'params_inversion.yaml'), 'w') as file:
        file.write("# @package _global_\n")
        yaml.dump(params, file, sort_keys=False)

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
                        default="../Results/Test_default/Glaciers/RGI2000"
                                "-v7.0-G-11-01706/",
                        help='Path to Glacier dir with OGGM_shop output')
    args = parser.parse_args()
    main(args.rgi_id_dir)
