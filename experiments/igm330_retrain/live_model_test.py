#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Live IGM model (igm_wrapper.LiveIGM; prototyped here, option 2 of the
efficiency question, STATUS.md):
one process keeps IGM's compiled ice-flow objects (network, optimizer with
its traced minimize, evaluator, discretisation) and runs the members of a
glacier one after the other, without IGM source changes:

    first member   plain in-process igm_run (igm_wrapper.run_igm); the
                   module lists, cfg and state are captured, and the network
                   weights right before the initial training
    next members   new State, inputs/outputs/processes initialised as in
                   igm_run except iceflow: the kept ice-flow objects are put
                   back, network weights reset to the captured start weights,
                   Adam moments zeroed, then IGM's own initial solve
                   (solve_iceflow(init=True)) and evaluate; then IGM's
                   update/finalize loop. cfg = first cfg merged with the
                   member's params.yaml (SMB table, time).

Rhone (as timing_test.py), 8 members with different ELAs, emulator retraining
(igm_default, 300 start iterations), 2-D output. Two modes, one process each:
    python live_model_test.py fresh   every member a fresh in-process igm_run
    python live_model_test.py live    live model
    python live_model_test.py compare differences live vs fresh per member
fresh also reruns member 0 at the end (noise floor of two fresh runs).
    sbatch experiments/igm330_retrain/live_model_test_sbatch.sh
"""

import os
import shutil
import sys
import time

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
import frost.glacier_model.igm_wrapper as W  # noqa: E402

RGI = 'RGI2000-v7.0-G-11-01706'
SRC = os.path.join('data/results/central_europe_combined_retrain', RGI)
TARGET = 'data/results/igm330_live_test'
ELAS = [2950.0, 3000.0, 3050.0, 3100.0, 2900.0, 3150.0, 3020.0, 2980.0]
YEARS = (2000, 2020)


def prepare(name):
    wd = os.path.join(TARGET, name)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, 'experiment'))
    shutil.copytree(os.path.join(SRC, 'Ensemble', 'Member_0', 'data'), os.path.join(wd, 'data'))
    return wd


def run_mode(mode):
    emulator = os.path.abspath(os.path.join(SRC, 'Preprocess', 'outputs', 'emulator.keras'))
    with Dataset(os.path.join(SRC, 'Ensemble', 'Member_0', 'data', 'input.nc')) as ds:
        usurf = np.array(ds['usurf'][:])
    W.LIVE_IGM = mode == 'live'
    members = list(enumerate(ELAS)) + ([(0, ELAS[0])] if mode == 'fresh' else [])
    seen = set()
    for i, ela in members:
        name = f'{mode}_{i}' + ('_rerun' if i in seen else '')
        seen.add(i)
        wd = prepare(name)
        t0 = time.time()
        W.forward('Calibration', False, True, i, 'ELA', usurf,
                  {'ela': ela, 'abl_grad': 7.0, 'acc_grad': 2.0}, *YEARS, wd, None,
                  emulator_path=emulator, max_velbar=1000, in_process=True,
                  emulator_retrain='igm_default')
        print(f'TIMING {mode} {name} ela {ela}: {time.time() - t0:.1f} s', flush=True)


def compare():
    def load(name):
        with Dataset(os.path.join(TARGET, name, 'outputs', 'output.nc')) as ds:
            return {k: np.array(ds[k][:]) for k in ('thk', 'velsurf_mag', 'smb')}
    pairs = [('fresh_0', 'fresh_0_rerun')] + [(f'fresh_{i}', f'live_{i}') for i in range(len(ELAS))]
    for a, b in pairs:
        try:
            A, B = load(a), load(b)
        except FileNotFoundError as e:
            print(f'COMPARE {a} vs {b}: missing output ({e})', flush=True)
            continue
        mask = A['thk'][0] > 1
        d = {k: np.abs(A[k] - B[k]).max() for k in A}
        v_a = A['velsurf_mag'][:, mask].mean()
        v_b = B['velsurf_mag'][:, mask].mean()
        dv = abs(v_a - v_b) / v_a
        vol_a, vol_b = A['thk'][-1].sum(), B['thk'][-1].sum()
        print(f'COMPARE {a} vs {b}: max|dthk| {d["thk"]:.3g} m, max|dvelsurf| '
              f'{d["velsurf_mag"]:.3g} m/yr, max|dsmb| {d["smb"]:.3g}, mean speed '
              f'{100 * dv:.3f} %, final volume {100 * abs(vol_a - vol_b) / vol_a:.4f} %',
              flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'compare':
        compare()
    else:
        run_mode(mode)
