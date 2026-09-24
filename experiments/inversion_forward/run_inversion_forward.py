#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
IGM inversion followed by a single forward run, without the ensemble.

Checks the transition from the inverted state to the forward model: at the
first time step the forward run sees the same geometry and tau_ref as the
end of the inversion, so its surface velocities should match the inverted
ones and then only drift slowly. The forward run is done twice, through the
same frost.glacier_model.igm_wrapper.forward the EnKF uses:

    inversion  - with the iceflow network saved by the inversion
                 (pretrained emulator fine-tuned on the glacier)
    pretrained - with the raw pretrained emulator (dahunet_mini), which
                 the forward runs used before; shows the initialisation shock

The SMB is the ELA model with the yearly GLAMOS ELA and gradients
(2000-2019); years without GLAMOS values use the glacier's GLAMOS mean.

Run from the repository root:
    python experiments/inversion_forward/run_inversion_forward.py
or as a GPU job on Alex: sbatch experiments/inversion_forward/run_inversion_forward.sh

Results: data/results/inversion_forward/glaciers/<rgi_id>/Forward/<variant>
and plots/ next to this script.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import frost.glacier_model.igm_wrapper as igm_wrapper
import frost.preprocess.download_data as download_data
import frost.preprocess.igm_inversion as igm_inversion

EXPERIMENT_NAME = 'inversion_forward'
EXPERIMENT_DIR = os.path.join('experiments', EXPERIMENT_NAME)
GLAMOS_TABLE = os.path.join('experiments', 'central_europe', 'validation',
                            'tables', 'GLAMOS_analysis_results.csv')
GLAMOS_YEARS = np.arange(2000, 2020)
VARIANTS = ('inversion', 'pretrained')

# Plot colours: categorical slots 1-2, sequential and diverging ramps
COLORS = {'inversion': '#2a78d6', 'pretrained': '#eb6834'}
LABELS = {'inversion': 'network from the inversion',
          'pretrained': 'raw pretrained network'}
INK, MUTED = '#0b0b0b', '#52514e'


def glamos_smb(rgi_id, year_start, year_end):
    """Yearly ELA and gradients from GLAMOS for year_start..year_end."""
    table = pd.read_csv(GLAMOS_TABLE)
    row = table[table['rgi_id'] == rgi_id]
    if row.empty:
        raise ValueError(f"{rgi_id} is not in {GLAMOS_TABLE}")
    row = row.iloc[0]

    smb = {}
    for key, column, mean_column in [
            ('ela', 'ELAS', 'Mean_ELA'),
            ('abl_grad', 'ablation_gradients', 'Mean_Ablation_Gradient'),
            ('acc_grad', 'accumulation_gradients',
             'Mean_Accumulation_Gradient')]:
        values = np.array(row[column].strip('[]').split(','), dtype=float)
        values = np.where(np.isnan(values), row[mean_column], values)
        yearly = dict(zip(GLAMOS_YEARS, values))
        # Outside 2000-2019 hold the nearest GLAMOS year
        smb[key] = [yearly[np.clip(year, GLAMOS_YEARS[0], GLAMOS_YEARS[-1])]
                    for year in range(year_start, year_end + 1)]
    return smb


def run_forward(rgi_id_dir, variant, smb, year_start, year_end):
    """Forward run from the inverted state; returns its output.nc path."""
    inversion_file = os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                                  'output.nc')
    workdir = os.path.join(rgi_id_dir, 'Forward', variant)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(os.path.join(workdir, 'data'))
    os.makedirs(os.path.join(workdir, 'experiment'))
    shutil.copy2(inversion_file, os.path.join(workdir, 'data', 'input.nc'))

    with Dataset(inversion_file) as ds:
        usurf = np.array(ds['usurf'])

    if variant == 'inversion':
        emulator = os.path.abspath(igm_inversion.emulator_path(rgi_id_dir))
    else:
        emulator = 'dahunet_mini.keras'

    igm_wrapper.forward('Forward', False, True, 0, 'ELA', usurf, smb,
                        year_start, year_end, workdir, None,
                        emulator_path=emulator)
    return os.path.join(workdir, 'outputs', 'output.nc')


def load_results(rgi_id_dir, output_files):
    inversion_file = os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                                  'output.nc')
    with Dataset(inversion_file) as ds:
        inv = {name: np.array(ds[name]) for name in
               ['velsurf_mag', 'velsurfobs_mag', 'icemask', 'thk', 'x']}
    fwd = {}
    for variant, path in output_files.items():
        with Dataset(path) as ds:
            fwd[variant] = {name: np.array(ds[name]) for name in
                            ['time', 'velsurf_mag', 'thk']}
    return inv, fwd


def summarize(inv, fwd):
    """Table of glacier-mean speed and misfit to the inversion per year."""
    mask = inv['icemask'] > 0.5
    obs = mask & np.isfinite(inv['velsurfobs_mag'])
    dx = abs(inv['x'][1] - inv['x'][0])
    rows = []
    for variant, f in fwd.items():
        for i, year in enumerate(f['time']):
            v = f['velsurf_mag'][i]
            rows.append({
                'variant': variant,
                'year': int(year),
                'mean_speed': np.nanmean(v[mask]),
                'rms_vs_inversion': np.sqrt(np.nanmean(
                    (v - inv['velsurf_mag'])[mask] ** 2)),
                'rms_vs_obs': np.sqrt(np.nanmean(
                    (v - inv['velsurfobs_mag'])[obs] ** 2)),
                'volume_km3': np.sum(f['thk'][i]) * dx ** 2 / 1e9,
            })
    table = pd.DataFrame(rows)
    reference = {
        'mean_speed': np.nanmean(inv['velsurf_mag'][mask]),
        'rms_vs_obs': np.sqrt(np.nanmean(
            (inv['velsurf_mag'] - inv['velsurfobs_mag'])[obs] ** 2)),
    }
    return table, reference


def plot_maps(inv, fwd, path):
    mask = inv['icemask'] > 0.5

    def masked(a):
        return np.where(mask, a, np.nan)

    speed_panels = [
        ('Observed', inv['velsurfobs_mag']),
        ('End of inversion', inv['velsurf_mag']),
        (f"Forward {fwd['inversion']['time'][0]:.0f}, "
         f"{LABELS['inversion']}", fwd['inversion']['velsurf_mag'][0]),
    ]
    diff_panels = [
        (f"{LABELS['pretrained'].capitalize()}, "
         f"{fwd['pretrained']['time'][0]:.0f}",
         fwd['pretrained']['velsurf_mag'][0]),
        (f"{LABELS['inversion'].capitalize()}, "
         f"{fwd['inversion']['time'][0]:.0f}",
         fwd['inversion']['velsurf_mag'][0]),
        (f"{LABELS['inversion'].capitalize()}, "
         f"{fwd['inversion']['time'][1]:.0f}",
         fwd['inversion']['velsurf_mag'][1]),
    ]
    vmax = np.nanpercentile(masked(inv['velsurf_mag']), 99)
    diffs = [masked(v - inv['velsurf_mag']) for _, v in diff_panels]
    dmax = max(np.nanpercentile(np.abs(d), 99) for d in diffs)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, v) in zip(axes[0], speed_panels):
        im = ax.imshow(masked(v), origin='lower', cmap='Blues', vmin=0,
                       vmax=vmax)
        ax.set_title(title, fontsize=10, color=INK)
    fig.colorbar(im, ax=axes[0], label='Surface speed (m/yr)', shrink=0.8)
    for ax, (title, _), d in zip(axes[1], diff_panels, diffs):
        im = ax.imshow(d, origin='lower', cmap='RdBu_r',
                       norm=TwoSlopeNorm(0, -dmax, dmax))
        ax.set_title(title, fontsize=10, color=INK)
    fig.colorbar(im, ax=axes[1], shrink=0.8,
                 label='Forward minus inversion (m/yr)')
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_series(table, reference, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, column, ylabel in [
            (axes[0], 'mean_speed', 'Glacier-mean surface speed (m/yr)'),
            (axes[1], 'rms_vs_inversion',
             'RMS difference to the inversion (m/yr)')]:
        for variant in VARIANTS:
            rows = table[table['variant'] == variant]
            ax.plot(rows['year'], rows[column], color=COLORS[variant],
                    linewidth=2, marker='o', markersize=4,
                    label=LABELS[variant])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Year', color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.grid(color='#e6e5e1', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].axhline(reference['mean_speed'], color=MUTED, linewidth=1,
                    linestyle='--', label='end of inversion')
    axes[0].legend(frameon=False, loc='upper right', bbox_to_anchor=(1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rgi_id', default='RGI2000-v7.0-G-11-01706',
                        help='Glacier (default: Rhone)')
    parser.add_argument('--year_start', type=int, default=2000)
    parser.add_argument('--year_end', type=int, default=2020)
    parser.add_argument('--skip_download', action='store_true')
    parser.add_argument('--skip_inversion', action='store_true')
    args = parser.parse_args()

    rgi_id_dir = os.path.join('data', 'results', EXPERIMENT_NAME, 'glaciers',
                              args.rgi_id)
    if not args.skip_download:
        download_data.main(rgi_id=args.rgi_id, rgi_id_dir=rgi_id_dir,
                           smb_model='ELA', target_resolution='None',
                           oggm_shop=True)
    if not args.skip_inversion:
        igm_inversion.main(rgi_id_dir=rgi_id_dir,
                           params_inversion_path=os.path.join(
                               EXPERIMENT_DIR, 'params_inversion.yaml'))

    smb = glamos_smb(args.rgi_id, args.year_start, args.year_end)
    output_files = {variant: run_forward(rgi_id_dir, variant, smb,
                                         args.year_start, args.year_end)
                    for variant in VARIANTS}

    inv, fwd = load_results(rgi_id_dir, output_files)
    table, reference = summarize(inv, fwd)

    plot_dir = os.path.join(EXPERIMENT_DIR, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    table.to_csv(os.path.join(rgi_id_dir, 'forward_summary.csv'), index=False)
    plot_maps(inv, fwd, os.path.join(plot_dir, f'{args.rgi_id}_velocity_maps.png'))
    plot_series(table, reference,
                os.path.join(plot_dir, f'{args.rgi_id}_velocity_series.png'))

    print(f"End of inversion: mean speed {reference['mean_speed']:.1f} m/yr, "
          f"RMS vs obs {reference['rms_vs_obs']:.1f} m/yr")
    print(table.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
