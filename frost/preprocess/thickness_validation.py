#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Comparison of the inverted ice thickness with thickness observations
(thkobs in input.nc, GlaThiDa from OGGM shop). The inversion does not use
them, so they validate it independently.

Writes, next to the inversion result in Preprocess/outputs:
    thickness_validation.json  bias, RMS and a table per elevation band
    thickness_validation.png   observed, inverted and their difference
"""

import argparse
import json
import os

import numpy as np
from netCDF4 import Dataset

# Elevation bands of equal ice area
NUM_BANDS = 5


def read_field(dataset, name):
    """Field as float array, NaN for masked and fill values."""
    values = np.ma.filled(dataset[name][:].astype(float), np.nan)
    values[np.abs(values) > 1e35] = np.nan
    return values


def elevation_bands(usurf, ice, observed):
    """Mean observed, inverted and input thickness per band."""
    edges = np.percentile(usurf[ice], np.linspace(0, 100, NUM_BANDS + 1))
    bands = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_band = observed & (usurf >= lower) & (usurf <= upper)
        bands.append({'lower': float(lower), 'upper': float(upper),
                      'n': int(in_band.sum()), 'mask': in_band})
    return bands


def validate_thickness(rgi_id_dir):
    """Compare the inverted thickness with thkobs; returns the statistics,
    or None if input.nc has no thickness observations on the ice."""
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    outputs_dir = os.path.join(rgi_id_dir, 'Preprocess', 'outputs')
    with Dataset(input_file) as nc:
        if 'thkobs' not in nc.variables:
            print('Thickness validation: no thkobs in input.nc')
            return None
        thkobs = read_field(nc, 'thkobs')
        thk_input = read_field(nc, 'thk') if 'thk' in nc.variables else None
    with Dataset(os.path.join(outputs_dir, 'output.nc')) as nc:
        thk = read_field(nc, 'thk')
        usurf = read_field(nc, 'usurf')
        ice = read_field(nc, 'icemask') > 0.5
        x = np.array(nc['x'][:])
        y = np.array(nc['y'][:])

    observed = ice & np.isfinite(thkobs) & (thkobs > 0)
    if not observed.any():
        print('Thickness validation: no thickness observations on the ice')
        return None

    diff = thk[observed] - thkobs[observed]
    stats = {
        'n_observed': int(observed.sum()),
        'coverage': float(observed.sum() / ice.sum()),
        'mean_observed': float(thkobs[observed].mean()),
        'mean_inverted': float(thk[observed].mean()),
        'bias': float(diff.mean()),
        'rms': float(np.sqrt(np.mean(diff ** 2))),
        'mae': float(np.mean(np.abs(diff))),
        'bands': [],
    }
    for band in elevation_bands(usurf, ice, observed):
        mask = band.pop('mask')
        if band['n']:
            band['observed'] = float(thkobs[mask].mean())
            band['inverted'] = float(thk[mask].mean())
            if thk_input is not None:
                band['input'] = float(thk_input[mask].mean())
        stats['bands'].append(band)

    with open(os.path.join(outputs_dir, 'thickness_validation.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    plot(os.path.join(outputs_dir, 'thickness_validation.png'), x, y, thk,
         thkobs, ice, observed, stats)

    print(f"Thickness validation ({stats['n_observed']} cells, "
          f"{stats['coverage']:.0%} of the ice): bias {stats['bias']:+.0f} m, "
          f"RMS {stats['rms']:.0f} m")
    print('  band (m)       n  observed  inverted')
    for band in stats['bands']:
        if band['n']:
            print(f"  {band['lower']:4.0f}-{band['upper']:4.0f} {band['n']:5d} "
                  f"{band['observed']:9.0f} {band['inverted']:9.0f}")
    return stats


def plot(path, x, y, thk, thkobs, ice, observed, stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # crop to the glacier, a few cells around the ice
    rows, cols = np.nonzero(ice)
    r0, r1 = max(rows.min() - 3, 0), min(rows.max() + 4, ice.shape[0])
    c0, c1 = max(cols.min() - 3, 0), min(cols.max() + 4, ice.shape[1])
    crop = np.s_[r0:r1, c0:c1]
    extent = [0, abs(x[c1 - 1] - x[c0]) / 1e3, 0, abs(y[r1 - 1] - y[r0]) / 1e3]
    # first row is the south edge unless y is descending
    origin = 'upper' if y[-1] < y[0] else 'lower'

    vmax = np.nanpercentile(np.concatenate([thk[ice], thkobs[observed]]), 99)
    diff = np.where(observed, thk - thkobs, np.nan)
    dmax = np.nanpercentile(np.abs(diff), 95)
    outline = np.where(ice, 1.0, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    panels = [
        (np.where(observed, thkobs, np.nan), 'Observed thickness (GlaThiDa)',
         'Blues', 0, vmax, 'm'),
        (np.where(ice, thk, np.nan), 'Inverted thickness', 'Blues', 0, vmax, 'm'),
        (diff, 'Inverted - observed', 'RdBu', -dmax, dmax, 'm'),
    ]
    for ax, (field, title, cmap, vmin, vmax_, label) in zip(axes, panels):
        ax.imshow(outline[crop], origin=origin, extent=extent, cmap='Greys',
                  vmin=0, vmax=4, interpolation='nearest')
        im = ax.imshow(field[crop], origin=origin, extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax_, interpolation='nearest')
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel('km')
    axes[0].set_ylabel('km')

    ax = axes[3]
    ax.scatter(thkobs[observed], thk[observed], s=4, alpha=0.4)
    ax.plot([0, vmax], [0, vmax], 'k--', lw=1)
    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel('Observed thickness (m)')
    ax.set_ylabel('Inverted thickness (m)')
    ax.set_title(f"bias {stats['bias']:+.0f} m, RMS {stats['rms']:.0f} m, "
                 f"n = {stats['n_observed']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('rgi_id_dir', help='Glacier folder with Preprocess/')
    validate_thickness(parser.parse_args().rgi_id_dir)
