#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""Volume and area of one glacier per RCP: median and 5-95 % of the CORDEX
runs, from <glacier>/Projection/projections.nc."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SCENARIOS = {'rcp_2_6': ('RCP2.6', '#2a78d6'),
             'rcp_4_5': ('RCP4.5', '#eb6834'),
             'rcp_8_5': ('RCP8.5', '#1baf7a')}


def main(rgi_id, experiment_name, name):
    projection_dir = os.path.join('data', 'results', experiment_name, rgi_id,
                                  'Projection')
    ds = xr.open_dataset(os.path.join(projection_dir, 'projections.nc'))
    time = ds.time.values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, var, label in [(axes[0], 'vol', 'Ice volume (km³)'),
                           (axes[1], 'area', 'Glaciated area (km²)')]:
        for scenario, (title, color) in SCENARIOS.items():
            runs = ds[var].values[ds.scenario.values == scenario]
            low, median, high = np.percentile(runs, [5, 50, 95], axis=0)
            ax.fill_between(time, low, high, color=color, alpha=0.18,
                            linewidth=0)
            ax.plot(time, median, color=color, linewidth=2,
                    label=f'{title} (n={len(runs)})')
            ax.annotate(title, (time[-1], median[-1]), xytext=(4, 0),
                        textcoords='offset points', va='center',
                        fontsize=9, color='#3d3d3a', annotation_clip=False)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=-0.03 * np.nanmax(ds[var].values))
        ax.grid(color='#e4e3dd', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(2019.5, color='#a3a29b', linewidth=1, linestyle=':')
    axes[0].text(2010, axes[0].get_ylim()[1] * 0.04, 'W5E5', ha='center',
                 fontsize=8, color='#73726c')
    axes[0].text(2030, axes[0].get_ylim()[1] * 0.04, 'CORDEX', ha='left',
                 fontsize=8, color='#73726c')
    axes[0].legend(frameon=False, fontsize=9, loc='upper right')
    axes[1].set_xlim(time[0], time[-1] + 12)
    fig.suptitle(f'{name}: TI projections, posterior-mean SMB, '
                 'median and 5–95 % of CORDEX runs', fontsize=11)
    fig.tight_layout()
    out_file = os.path.join(projection_dir, 'projections.png')
    fig.savefig(out_file, dpi=150)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgi_id', default='RGI2000-v7.0-G-11-01706')
    parser.add_argument('--experiment_name', default='alps_TI_projections')
    parser.add_argument('--name', default='Rhone')
    args = parser.parse_args()
    main(args.rgi_id, args.experiment_name, args.name)
