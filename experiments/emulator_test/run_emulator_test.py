#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Is the start-up speed shock of the forward runs (central_europe_esmda_velyear:
speeds of 2000 up to 24x those of 2018) an error of the frozen iceflow
emulator? Reruns one posterior member (input.nc and SMB of Member_0) of a
glacier 2000-2020 with three ice-flow settings:

    frozen   pretrained network, never retrained (as the calibration runs)
    retrain  the same network, retrained on the glacier: 300 iterations at
             the start, 5 every 5 time steps (IGM defaults)
    solver   no network: the velocity field solved by L-BFGS (500 iterations
             at the start, 20 per time step)

Run from the repository root on a GPU node (one glacier per array task):
    sbatch --array=1-4 experiments/emulator_test/emulator_test_sbatch.sh
or for one glacier:
    python experiments/emulator_test/run_emulator_test.py --rgi_id RGI2000-v7.0-G-11-03664
Writes data/results/emulator_test/<rgi_id>/<variant>/outputs/output.nc and
runtime.txt; compare.py makes the table and figures.
"""

import argparse
import os
import shutil
import subprocess
import time

import yaml

SOURCE = 'data/results/central_europe_esmda_velyear'
TARGET = 'data/results/emulator_test'
MEMBER = 'Member_0'

VARIANTS = {
    'frozen': {'nbit_init': 0, 'retrain_freq': 0},
    'retrain': {'nbit_init': 300, 'nbit': 5, 'retrain_freq': 5},
    'solver': {'mapping': 'identity', 'optimizer': 'lbfgs',
               'nbit_init': 500, 'nbit': 20, 'retrain_freq': 1},
}


def setup(rgi_id, variant):
    member = os.path.join(SOURCE, rgi_id, 'Ensemble', MEMBER)
    workdir = os.path.join(TARGET, rgi_id, variant)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(os.path.join(workdir, 'experiment'))
    shutil.copytree(os.path.join(member, 'data'), os.path.join(workdir, 'data'))
    with open(os.path.join(member, 'experiment', 'params.yaml')) as f:
        params = yaml.safe_load(f)
    unified = params['processes']['iceflow']['unified']
    unified.update(VARIANTS[variant])
    if variant == 'solver':
        unified.pop('network', None)
    with open(os.path.join(workdir, 'experiment', 'params.yaml'), 'w') as f:
        f.write('# @package _global_\n')
        yaml.dump(params, f, sort_keys=False)
    return workdir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rgi_id', required=True)
    parser.add_argument('--variants', nargs='*', default=list(VARIANTS))
    args = parser.parse_args()
    for variant in args.variants:
        workdir = setup(args.rgi_id, variant)
        print(f'{args.rgi_id} {variant}: {workdir}', flush=True)
        start = time.time()
        result = subprocess.run(['igm_run', '+experiment=params'], cwd=workdir)
        with open(os.path.join(workdir, 'runtime.txt'), 'w') as f:
            f.write(f'{time.time() - start:.0f} s, exit code {result.returncode}\n')
        print(f'{args.rgi_id} {variant}: {time.time() - start:.0f} s, exit code '
              f'{result.returncode}', flush=True)


if __name__ == '__main__':
    main()
