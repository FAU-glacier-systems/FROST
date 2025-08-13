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

    # Prepare inversion directory
    preprocess_dir = os.path.join(rgi_id_dir, 'Preprocess')
    # shutil.rmtree(inversion_dir, ignore_errors=True)
    exp_dir = os.path.join(preprocess_dir, 'experiment')
    os.makedirs(exp_dir, exist_ok=True)

    shutil.copy(params_inversion_path, exp_dir)

    # Change to inversion directory and save params
    original_dir = os.getcwd()
    os.chdir(preprocess_dir)

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
