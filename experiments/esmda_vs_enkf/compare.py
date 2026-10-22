#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""Table of the EnKF and ES-MDA posteriors of Aletsch and Rhone: prior,
posterior mean and std per parameter, the std ratio EnKF/ES-MDA, and the
misfit chi^2/n of every iteration. Reads only calibration_results.json."""

import json
import os

import numpy as np

GLACIERS = {'RGI2000-v7.0-G-11-02596': 'Aletsch',
            'RGI2000-v7.0-G-11-01706': 'Rhone'}
METHODS = ('enkf', 'esmda')
RESULTS = os.path.join('data', 'results', 'esmda_vs_enkf')


def load(method, rgi_id):
    with open(os.path.join(RESULTS, method, rgi_id,
                           'calibration_results.json')) as f:
        return json.load(f)


def main():
    for rgi_id, name in GLACIERS.items():
        runs = {m: load(m, rgi_id) for m in METHODS}
        keys = list(runs['esmda']['initial_smb'])
        print(f'\n{name} ({rgi_id})')
        print(f"{'parameter':<10} {'prior':>14} {'EnKF':>14} "
              f"{'ES-MDA':>14} {'std EnKF/ES-MDA':>16}")
        for i, key in enumerate(keys):
            prior = (f"{runs['esmda']['initial_smb'][key]:.2f}"
                     f" ± {runs['esmda']['initial_spread'][key]:.2f}")
            post = {m: f"{runs[m]['final_mean'][i]:.2f}"
                       f" ± {runs[m]['final_std'][i]:.2f}" for m in METHODS}
            ratio = runs['enkf']['final_std'][i] / runs['esmda']['final_std'][i]
            print(f"{key:<10} {prior:>14} {post['enkf']:>14} "
                  f"{post['esmda']:>14} {ratio:>16.2f}")
        for m in METHODS:
            corr = np.corrcoef(np.array(runs[m]['final_ensemble']).T)
            pairs = ', '.join(f'{keys[a]}/{keys[b]} {corr[a, b]:+.2f}'
                              for a in range(len(keys))
                              for b in range(a + 1, len(keys)))
            misfit = ' '.join(f"{d['misfit']:.2f}"
                              for d in runs[m]['diagnostics'])
            print(f'{m:<6} correlation: {pairs}')
            print(f'{m:<6} chi^2/n per iteration: {misfit}')


if __name__ == '__main__':
    main()
