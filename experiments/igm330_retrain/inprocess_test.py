#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
IGM 3.3.0 several times in one process, as the calibration workers do
(igm_wrapper.forward with in_process=True): Member_0 of Rhone, 2000-2003,
three runs per retraining rule. Before igm_wrapper.reset_igm_tf_functions
the second run failed ("Called a function referencing variables which have
been deleted", 2026-11-16). On a GPU node:
    sbatch experiments/igm330_retrain/inprocess_test_sbatch.sh
"""

import os
import shutil
import sys
import time

sys.path.insert(0, os.getcwd())
import frost.glacier_model.igm_wrapper as W  # noqa: E402

RGI = 'RGI2000-v7.0-G-11-01706'
SRC = os.path.join('data/results/central_europe_combined_retrain', RGI)
TARGET = 'data/results/igm330_inprocess_test'


def main():
    failed = 0
    for rule in ['credit', 'igm_default', False]:
        for k in range(3):
            wd = os.path.join(TARGET, str(rule), f'run_{k}')
            shutil.rmtree(wd, ignore_errors=True)
            os.makedirs(os.path.join(wd, 'experiment'))
            shutil.copytree(os.path.join(SRC, 'Ensemble', 'Member_0', 'data'),
                            os.path.join(wd, 'data'))
            start = time.time()
            try:
                W.forward('Calibration', False, True, k, 'ELA', None,
                          {'ela': 3000 + 50 * k, 'abl_grad': 7.0, 'acc_grad': 2.0},
                          2000, 2003, wd, None,
                          emulator_path=os.path.abspath(os.path.join(SRC, 'Preprocess', 'outputs', 'emulator.keras')),
                          max_velbar=1000, in_process=True, emulator_retrain=rule)
                print(f'RESULT {rule} run {k}: ok, {time.time() - start:.1f} s', flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f'RESULT {rule} run {k}: FAILED {type(e).__name__}: {e}', flush=True)
    print('RESULT all ok' if not failed else f'RESULT {failed} failed')


if __name__ == '__main__':
    main()
