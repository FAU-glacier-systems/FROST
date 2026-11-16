#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Emulator retraining rules of IGM 3.3.0 against the solver: one posterior
member (input.nc and SMB of Member_0 of central_europe_combined_retrain, as
experiments/emulator_test, target emulator_test_interval) run 2000-2020 with

    solver        no network: velocities solved by L-BFGS (500 iterations at
                  the start, 20 per time step)
    igm_default   300 at the start, then 5 iterations every 5 time steps
                  (IGM default, adaptive_time none)
    credit        300 at the start, then 5 iterations whenever the
                  accumulated input change reaches kappa 1e-3 (IGM 3.3.0
                  adaptive_time credit, defaults; at the latest every 100 steps)
    credit_x20    as credit with 20 iterations per retraining
    credit_k1e-2  as credit with kappa 1e-2 (fewer retrainings)

The runs of IGM 3.2.1 + the retrain_interval fork (20 every 1 yr of model
time, the v3-dev rule until 2026-11-16) stay in
data/results/emulator_test_interval; report.py compares both.

Run from the repository root on a GPU node (one glacier per array task):
    sbatch --array=1-3 experiments/igm330_retrain/forward_test_sbatch.sh
Writes data/results/igm330_forward_test/<rgi_id>/<variant>/outputs/output.nc,
runtime.txt and (credit variants) credit_log.jsonl.
"""

import argparse
import os
import shutil
import subprocess
import time

import yaml

SOURCE = 'data/results/central_europe_combined_retrain'
TARGET = 'data/results/igm330_forward_test'
OLD_TARGET = 'data/results/emulator_test_interval'  # IGM 3.2.1 + retrain_interval
MEMBER = 'Member_0'

NONE = {'adaptive_time': {'method': 'none'}}
CREDIT = {'retrain_freq': 5, 'adaptive_time': {'method': 'credit'}}
VARIANTS = {
    'solver': {'mapping': 'identity', 'optimizer': 'lbfgs',
               'nbit_init': 500, 'nbit': 20, 'retrain_freq': 1, **NONE},
    'igm_default': {'nbit_init': 300, 'nbit': 5, 'retrain_freq': 5, **NONE},
    'credit': {'nbit_init': 300, 'nbit': 5, **CREDIT},
    'credit_x20': {'nbit_init': 300, 'nbit': 20, **CREDIT},
    'credit_k1e-2': {'nbit_init': 300, 'nbit': 5, 'retrain_freq': 5,
                     'adaptive_time': {'method': 'credit', 'credit': {'kappa': 1e-2}}},
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
    import importlib.metadata
    import importlib.util
    print(f'IGM {importlib.metadata.version("igm-model")}: '
          f'{importlib.util.find_spec("igm").origin}', flush=True)
    for variant in args.variants:
        workdir = setup(args.rgi_id, variant)
        print(f'{args.rgi_id} {variant}: {workdir}', flush=True)
        start = time.time()
        result = subprocess.run(['igm_run', '+experiment=params'], cwd=workdir)
        # credit_log.jsonl lands in the hydra run dir; keep it next to runtime.txt
        for root, _, files in os.walk(workdir):
            if 'credit_log.jsonl' in files and root != workdir:
                shutil.move(os.path.join(root, 'credit_log.jsonl'), workdir)
        with open(os.path.join(workdir, 'runtime.txt'), 'w') as f:
            f.write(f'{time.time() - start:.0f} s, exit code {result.returncode}\n')
        print(f'{args.rgi_id} {variant}: {time.time() - start:.0f} s, exit code '
              f'{result.returncode}', flush=True)


if __name__ == '__main__':
    main()
