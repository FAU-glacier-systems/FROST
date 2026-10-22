#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Maps of inversion variants side by side for one glacier: thickness, modelled
surface speed, modelled minus observed speed and tau_ref, one row per
variant, plus the scores of each variant (scores.json).

Run from the repository root:
    python experiments/inversion_setup/compare_inversions.py \
        --rgi_id RGI2000-v7.0-G-11-02596 --variants central_europe slide_only_tau1e11
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset

RESULTS_DIR = os.path.join('data', 'results', 'inversion_setup')
PLOT_DIR = os.path.join('experiments', 'inversion_setup', 'plots')
SCORES = ['volume_km3', 'thk_mean', 'vel_obs_median', 'vel_median', 'vel_rms',
          'tau_ref_median', 'glathida_bias', 'glathida_rms']


def load(variant, rgi_id):
    folder = os.path.join(RESULTS_DIR, variant, rgi_id)
    with Dataset(os.path.join(folder, 'Preprocess', 'outputs', 'output.nc')) as ds:
        fields = {k: np.array(ds[k][:], float) for k in
                  ('thk', 'velsurf_mag', 'velsurfobs_mag', 'tau_ref', 'icemask')}
    with open(os.path.join(folder, 'scores.json')) as f:
        scores = json.load(f)
    return fields, scores


def main(rgi_id, variants):
    variants = [v for v in variants
                if os.path.exists(os.path.join(RESULTS_DIR, v, rgi_id, 'scores.json'))]
    rows = [(v, *load(v, rgi_id)) for v in variants]
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)),
                             squeeze=False)
    vmax = np.nanmax([np.nanpercentile(f['velsurfobs_mag'][f['icemask'] > 0.5], 99)
                      for _, f, _ in rows])
    for (variant, f, s), ax in zip(rows, axes):
        ice = f['icemask'] > 0.5
        mask = lambda a: np.where(ice, a, np.nan)
        panels = [
            (mask(f['thk']), 'Blues', 0, 600, 'thickness (m)'),
            (mask(f['velsurf_mag']), 'magma', 0, vmax, 'modelled speed (m/yr)'),
            (mask(f['velsurf_mag'] - f['velsurfobs_mag']), 'RdBu_r', -50, 50,
             'modelled - observed speed (m/yr)'),
            (mask(f['tau_ref']), 'viridis', 0, 1, 'tau_ref (MPa)'),
        ]
        for a, (field, cmap, lo, hi, label) in zip(ax, panels):
            im = a.imshow(field, origin='lower', cmap=cmap, vmin=lo, vmax=hi)
            a.set_title(label, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
            fig.colorbar(im, ax=a, fraction=0.046)
        ax[0].set_ylabel(variant, fontsize=12)
        print(f'{variant:>20}: ' + ', '.join(
            f'{k} {s[k]:.3g}' for k in SCORES if s.get(k) is not None))
    fig.suptitle(rgi_id)
    os.makedirs(PLOT_DIR, exist_ok=True)
    out = os.path.join(PLOT_DIR, f'inversion_variants_{rgi_id}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Wrote {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgi_id', required=True)
    parser.add_argument('--variants', nargs='+',
                        default=['central_europe', 'central_europe_local', 'igm_guess',
                                 'slide_only_tau1e11'])
    args = parser.parse_args()
    main(args.rgi_id, args.variants)
