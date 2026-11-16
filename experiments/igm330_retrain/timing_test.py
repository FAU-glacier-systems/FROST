#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Where does the time of one calibration member go? Rhone, Member_0 of
central_europe_combined_retrain, igm_wrapper.forward exactly as the
calibration workers call it (in-process, 2-D output), alone on one GPU.
Each case runs twice in the same process (the second run is what a
calibration worker sees after its first member):

    retrain_20y   emulator retraining, 300 start iterations, 2000-2020 (now)
    retrain_1y    the same, 2000-2001: start-up cost, almost no time loop
    start0_20y    retraining rule, but 0 start iterations, 2000-2020
    start0_1y     the same, 2000-2001
    frozen_20y    frozen emulator (no training at all), 2000-2020
    frozen_1y     the same, 2000-2001

Differences: 300 start iterations = retrain - start0; 19-year time loop =
20y - 1y; build + compile + I/O = frozen_1y. A first frozen 1-year run
absorbs the one-off TensorFlow/CUDA start. On a GPU node:
    sbatch experiments/igm330_retrain/timing_test_sbatch.sh
Writes experiments/igm330_retrain/tables/timing.tsv.
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
TARGET = 'data/results/igm330_timing_test'
SMB = {'ela': 3000.0, 'abl_grad': 7.0, 'acc_grad': 2.0}
CASES = [  # (name, emulator_retrain, start iterations, end year)
    ('warmup', False, 0, 2001),
    ('retrain_20y', 'igm_default', 300, 2020), ('retrain_20y', 'igm_default', 300, 2020),
    ('retrain_1y', 'igm_default', 300, 2001), ('retrain_1y', 'igm_default', 300, 2001),
    ('start0_20y', 'igm_default', 0, 2020), ('start0_20y', 'igm_default', 0, 2020),
    ('start0_1y', 'igm_default', 0, 2001), ('start0_1y', 'igm_default', 0, 2001),
    ('frozen_20y', False, 0, 2020), ('frozen_20y', False, 0, 2020),
    ('frozen_1y', False, 0, 2001), ('frozen_1y', False, 0, 2001),
]


def main():
    emulator = os.path.abspath(os.path.join(SRC, 'Preprocess', 'outputs', 'emulator.keras'))
    with Dataset(os.path.join(SRC, 'Ensemble', 'Member_0', 'data', 'input.nc')) as ds:
        usurf = np.array(ds['usurf'][:])
    rows, seen = [], {}
    for name, rule, start, end in CASES:
        k = seen[name] = seen.get(name, 0) + 1
        wd = os.path.join(TARGET, f'{name}_{k}')
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(os.path.join(wd, 'experiment'))
        shutil.copytree(os.path.join(SRC, 'Ensemble', 'Member_0', 'data'), os.path.join(wd, 'data'))
        W.RETRAIN_START = start
        t0 = time.time()
        W.forward('Calibration', False, True, 0, 'ELA', usurf, SMB, 2000, end, wd,
                  None, emulator_path=emulator, max_velbar=1000, in_process=True,
                  emulator_retrain=rule)
        dt = time.time() - t0
        rows.append((name, k, dt))
        print(f'TIMING {name} run {k}: {dt:.1f} s', flush=True)
    os.makedirs('experiments/igm330_retrain/tables', exist_ok=True)
    with open('experiments/igm330_retrain/tables/timing.tsv', 'w') as f:
        f.write('case\trun\tseconds\n')
        f.writelines(f'{n}\t{k}\t{s:.2f}\n' for n, k, s in rows)
    t = {(n, k): s for n, k, s in rows}
    w = lambda n: t[(n, 2)]  # noqa: E731  warm (second) run
    print(f"SUMMARY warm run: retrain_20y {w('retrain_20y'):.1f} s = "
          f"build/compile/IO {w('frozen_1y'):.1f} + 300 start iterations "
          f"{w('retrain_20y') - w('start0_20y'):.1f} + 19-yr loop with retraining "
          f"{w('start0_20y') - w('start0_1y'):.1f} (frozen loop "
          f"{w('frozen_20y') - w('frozen_1y'):.1f}) + rest "
          f"{w('start0_1y') - w('frozen_1y'):.1f}; first run of a process "
          f"{t[('retrain_20y', 1)]:.1f} s", flush=True)


if __name__ == '__main__':
    main()
