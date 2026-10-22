#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
tau_ref of the inversion with Millan velocities (iceboost/) against
ITS_LIVE velocities (iceboost_itslive/), same IceBoost geometry and
smoothness 1e11: maps of tau_ref, their ratio, the sliding share of the
surface speed and the speed misfit, per glacier of runs.txt.

Run from the repository root after the inversions:
    python experiments/thickness_esmda/compare_tau_ref.py
Writes plots/tau_ref.png and tables/tau_ref.tsv.
"""

import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join('data', 'results', 'thickness_esmda')
RUNS = {'millan': 'iceboost', 'itslive': 'iceboost_itslive'}


def fields(run, rgi_id):
    path = os.path.join(RESULTS, run, rgi_id, 'Preprocess', 'outputs', 'output.nc')
    with Dataset(path) as ds:
        mask = np.array(ds['icemask'][:]) > 0.5
        out = {k: np.where(mask, np.array(ds[k][:], dtype=float), np.nan)
               for k in ('tau_ref', 'velsurf_mag', 'velbase_mag', 'velsurfobs_mag', 'thk')}
        out['dx'] = float(ds['x'][1] - ds['x'][0])
    obs = out['velsurfobs_mag']
    out['velsurfobs_mag'] = np.where(np.isfinite(obs) & (obs > 0), obs, np.nan)
    out['sliding'] = out['velbase_mag'] / np.maximum(out['velsurf_mag'], 1e-3)
    out['misfit'] = out['velsurf_mag'] - out['velsurfobs_mag']
    return out


def main():
    with open(os.path.join(HERE, 'runs.txt')) as f:
        glaciers = dict.fromkeys(tuple(line.split()[1:3]) for line in f if line.strip())
    fig, axes = plt.subplots(len(glaciers), 7, figsize=(26, 5.2 * len(glaciers)),
                             squeeze=False)
    rows = []
    for ax, (rgi_id, name) in zip(axes, glaciers):
        f = {v: fields(run, rgi_id) for v, run in RUNS.items()}
        for v, field in f.items():
            obs = np.isfinite(field['misfit'])
            rows.append(dict(
                glacier=name, velocities=v,
                tau_ref_p10=np.nanpercentile(field['tau_ref'], 10),
                tau_ref_median=np.nanmedian(field['tau_ref']),
                tau_ref_p90=np.nanpercentile(field['tau_ref'], 90),
                at_upper_bound_pct=100 * np.nanmean(field['tau_ref'] > 1.99),
                at_lower_bound_pct=100 * np.nanmean(field['tau_ref'] < 0.051),
                sliding_median=np.nanmedian(field['sliding']),
                obs_speed_median=np.nanmedian(field['velsurfobs_mag']),
                speed_median=np.nanmedian(field['velsurf_mag'][obs]),
                speed_bias=np.mean(field['misfit'][obs]),
                speed_rms=np.sqrt(np.mean(field['misfit'][obs] ** 2))))
        extent = None
        tau_norm = LogNorm(0.05, 2.0)
        for k, v in enumerate(RUNS):
            im = ax[k].imshow(f[v]['tau_ref'], origin='lower', norm=tau_norm, cmap='viridis')
            ax[k].set_title(f'{name}: tau_ref (MPa), {v} velocities')
            plt.colorbar(im, ax=ax[k], shrink=0.8)
        ratio = np.log10(f['itslive']['tau_ref'] / f['millan']['tau_ref'])
        im = ax[2].imshow(ratio, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(0, -1, 1))
        ax[2].set_title('log10 tau_ref itslive / millan')
        plt.colorbar(im, ax=ax[2], shrink=0.8)
        for k, v in enumerate(RUNS):
            im = ax[3 + k].imshow(f[v]['sliding'], origin='lower', vmin=0, vmax=1, cmap='magma')
            ax[3 + k].set_title(f'sliding / surface speed, {v}')
            plt.colorbar(im, ax=ax[3 + k], shrink=0.8)
        vmax = np.nanpercentile(np.abs(f['millan']['misfit']), 98)
        for k, v in enumerate(RUNS):
            a = ax[5 + k]
            im = a.imshow(f[v]['misfit'], origin='lower', cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, -vmax, vmax))
            a.set_title(f'speed model - obs (m/yr), {v}')
            plt.colorbar(im, ax=a, shrink=0.8)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(HERE, 'plots', 'tau_ref.png'), dpi=90)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    table.to_csv(os.path.join(HERE, 'tables', 'tau_ref.tsv'), sep='\t', index=False,
                 float_format='%.4g')
    print(table.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    main()
