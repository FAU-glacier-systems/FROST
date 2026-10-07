#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Comparison of the inverted ice thickness with thickness observations
(GlaThiDa from OGGM shop). The inversion does not use them, so they validate
it independently. Every survey is moved to the date of the inverted state
(2000, the start of the calibration) with the observed dh/dt of its cell,
thk_2000 = thk_survey - dhdt * (survey_year - 2000), and averaged per cell.
Without the GlaThiDa table (glathida_data.csv) the gridded thkobs of
input.nc is used as it is.

Writes, next to the inversion result in Preprocess/outputs:
    thickness_validation.json  bias, RMS and a table per elevation band
    thickness_validation.png   observed, inverted and their difference
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Elevation bands of equal ice area
NUM_BANDS = 5


def read_field(dataset, name):
    """Field as float array, NaN for masked and fill values."""
    values = np.ma.filled(dataset[name][:].astype(float), np.nan)
    values[np.abs(values) > 1e35] = np.nan
    return values


def thkobs_at_year(glathida_file, x, y, dhdt, year):
    """GlaThiDa thickness moved to year with the cell's dh/dt and averaged
    per grid cell, and the survey years of the cells (NaN without data)."""
    points = pd.read_csv(glathida_file)
    points = points[points.thickness > 0].copy()
    survey_year = pd.to_datetime(points.date, errors='coerce').dt.year
    points['year'] = survey_year.fillna(year)
    points['ix'] = np.round((points.x_proj - x[0]) / (x[1] - x[0])).astype(int)
    points['iy'] = np.round((points.y_proj - y[0]) / (y[1] - y[0])).astype(int)
    points = points[(points.ix >= 0) & (points.ix < len(x))
                    & (points.iy >= 0) & (points.iy < len(y))]
    points['thk_year'] = (points.thickness - dhdt[points.iy, points.ix]
                          * (points.year - year))
    cells = points.groupby(['iy', 'ix'])[['thk_year', 'year']].mean()
    thkobs = np.full(dhdt.shape, np.nan)
    years = np.full(dhdt.shape, np.nan)
    iy, ix = (cells.index.get_level_values(k).to_numpy() for k in ('iy', 'ix'))
    thkobs[iy, ix] = cells.thk_year.to_numpy()
    years[iy, ix] = cells.year.to_numpy()
    return thkobs, years


def elevation_bands(usurf, ice, observed):
    """Mean observed, inverted and input thickness per band."""
    edges = np.percentile(usurf[ice], np.linspace(0, 100, NUM_BANDS + 1))
    bands = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_band = observed & (usurf >= lower) & (usurf <= upper)
        bands.append({'lower': float(lower), 'upper': float(upper),
                      'n': int(in_band.sum()), 'mask': in_band})
    return bands


def validate_thickness(rgi_id_dir, year=2000):
    """Compare the inverted thickness (the state of year) with the thickness
    observations; returns the statistics, or None if there are none on the
    ice."""
    input_file = os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')
    outputs_dir = os.path.join(rgi_id_dir, 'Preprocess', 'outputs')
    glathida_file = os.path.join(rgi_id_dir, 'Preprocess', 'data',
                                 os.path.basename(os.path.normpath(rgi_id_dir)),
                                 'glathida_data.csv')
    with Dataset(input_file) as nc:
        thkobs = read_field(nc, 'thkobs') if 'thkobs' in nc.variables else None
        thk_input = read_field(nc, 'thk') if 'thk' in nc.variables else None
        dhdt = read_field(nc, 'dhdt') if 'dhdt' in nc.variables else None
    with Dataset(os.path.join(outputs_dir, 'output.nc')) as nc:
        thk = read_field(nc, 'thk')
        usurf = read_field(nc, 'usurf')
        ice = read_field(nc, 'icemask') > 0.5
        x = np.array(nc['x'][:])
        y = np.array(nc['y'][:])

    survey_years = None
    if os.path.exists(glathida_file) and dhdt is not None:
        thkobs, survey_years = thkobs_at_year(
            glathida_file, x, y, np.nan_to_num(dhdt), year)
    elif thkobs is None:
        print('Thickness validation: no thickness observations')
        return None

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
        'year': year if survey_years is not None else None,
        'survey_year_median': (float(np.median(survey_years[observed]))
                               if survey_years is not None else None),
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
        (np.where(observed, thkobs, np.nan),
         'Observed thickness (GlaThiDa'
         + (f", moved to {stats['year']})" if stats['year'] else ')'),
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
