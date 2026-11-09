#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Sanity report of a FROST run (inversion + ES-MDA calibration) as an A4
PDF. Page 1: setup (glaciers, observations, where IGM ran and how it was
configured), a performance summary and four figures over all glaciers
(thickness, velocity in time, dh/dt, SMB). Then one page per topic over
all glaciers (thickness, velocity, dh/dt, SMB profiles, SMB parameters
against GLAMOS / WGMS), the summary table and one page per glacier. Every
plot has a caption; every glacier keeps its colour and marker throughout.

Glacier pages (3 x 2):
1. Thickness of 2000 (the thickness product moved to 2000) with GlaThiDa
   moved to 2000 as dots and cell by cell (inset); speed by elevation band
   in every year of the forward runs against ITS_LIVE.
2. Band profile of dh/dt (observation, calibration target with the model
   error, posterior members) and the posterior SMB against GLAMOS / WGMS,
   both with minus the flux divergence (dh/dt = SMB - flux divergence).
3. Maps of the dh/dt misfit 2000-2020 and of the speed misfit in the
   inversion year (without ice flow: the observed speed).
The bed (high-pass roughness, pits, ice thickness at the outline) and
tau_ref (share at the bounds, small-scale roughness and correlation with
the observed speed, sliding share, velocity misfit of the inversion) are
checked in numbers only (summary.tsv).

Run from the repository root:
    python experiments/central_europe_esmda/sanity_report.py \\
        --results data/results/central_europe_esmda_velyear
Writes <out>/report.pdf, <out>/summary.tsv, <out>/REPORT.md, the per-glacier
figures <out>/<glacier>/glacier.png and <out>/overview.pkl
(data of the overview pages). The text of <out>/findings.md, if present,
goes on the first page; --pdf_only rebuilds only the PDF (e.g. after
editing findings.md); --dark writes a black version (report_dark.pdf,
glacier_dark.png). <out> defaults to experiments/<results name>/
sanity_report; the experiment folder (same name as the results) gives the
band height, model error and tau_ref bounds, and its
plots/evaluation_central_europe/GLAMOS_regional_run.png (3x3 figure of
experiments/central_europe/evaluation.py), if present, is added.
"""

import argparse
import glob
import json
import os
import pickle
import sys
import textwrap
from datetime import date

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from frost.calibration.observation_provider import ObservationProvider
from frost.preprocess.thickness_validation import read_field, thkobs_at_year
from frost.visualization.plot_style import DARK_THEME, MONO, bg, fg
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'validation'))

# --dark: black pages (DARK_THEME), figures *_dark.png, report_dark.pdf
SUFFIX = ''
TABLE_COLOURS = {'cell': 'white', 'header': '#dddddd', 'flag': '#f4b6b6'}

# As the 3x3 figure of experiments/central_europe/evaluation.py: no spines,
# no tick marks, dashed grid behind the data (maps too, where they are empty)
STYLE = {**{f'axes.spines.{side}': False for side in ['top', 'right', 'bottom', 'left']},
         **{f'{axis}tick.{which}.size': 0 for axis in 'xy' for which in ['major', 'minor']},
         'axes.grid': True, 'axes.axisbelow': True, 'grid.linestyle': '--',
         'grid.linewidth': 0.5, 'grid.color': '0.75', 'legend.frameon': False}
plt.rcParams.update(STYLE)

SMOOTHING = 200.0   # m, Gaussian sigma of the high-pass checks
A4 = (8.27, 11.69)  # portrait, inches
DPI = 150

# Flags of the summary table (red) and REPORT.md
FLAGS = {
    'speed_ratio_obs': (lambda v: abs(np.log(v)) > np.log(1.33),
                        'forward speed in the observation year off the observed by > 33 %'),
    'speed_start_ratio': (lambda v: v > 2, 'forward speed of 2000 > 2x that of the observation year (start-up shock)'),
    'dhdt_chi2_data': (lambda v: v > 4, 'modelled band dh/dt outside the data error (chi2/n > 4)'),
    'dead_bands': (lambda v: v > 0, 'bands that melt out in every member'),
    'ela_ref_capped': (lambda v: v > 0.25, 'reference ELA at the glacier top in > 25 % of the years '
                                           '(a lower bound, left out of the ELA scores)'),
}


# ----------------------------------------------------------------- data

def experiment_settings(results):
    """Band height, model error and tau_ref bounds of the experiment of the
    results folder (experiments/<name>)."""
    folder = os.path.join('experiments', os.path.basename(os.path.normpath(results)))
    settings = dict(elevation_step=50, model_error=0.0, tau_bounds=(np.nan, np.nan))
    config = os.path.join(folder, 'pipeline_config.yaml')
    if os.path.exists(config):
        with open(config) as f:
            enkf = yaml.safe_load(f).get('EnKF', {})
        settings['elevation_step'] = enkf.get('elev_band_height', 50)
        settings['model_error'] = enkf.get('model_error', 0.0)
    params = os.path.join(folder, 'params_inversion.yaml')
    if os.path.exists(params):
        with open(params) as f:
            variables = yaml.safe_load(f)['assimilations']['field_inversion']['variables']
        tau = [v for v in variables if v['name'] == 'tau_ref']
        if tau:
            settings['tau_bounds'] = (tau[0].get('lower_bound', np.nan),
                                      tau[0].get('upper_bound', np.nan))
    return folder, settings


def load_glacier(rgi_dir, settings):
    """Inversion result, thickness observations, observation provider and
    the reductions of the posterior members needed by the figures."""
    g = {'rgi_dir': rgi_dir}
    with Dataset(os.path.join(rgi_dir, 'Preprocess', 'outputs', 'output.nc')) as ds:
        # without ice flow (no_flow_setup.py) there is no inversion: no
        # tau_ref, no velocities of the model
        g['no_flow'] = getattr(ds, 'ice_flow', 'igm') == 'none'
        for name in ['usurf', 'thk', 'topg', 'tau_ref', 'velbase_mag',
                     'velsurf_mag', 'velsurfobs_mag']:
            g[name] = read_field(ds, name) if name in ds.variables \
                else np.full(ds['usurf'].shape, np.nan)
        g['ice'] = read_field(ds, 'icemask') > 0.5
        for name in ['usurf_inversion', 'thk_inversion']:
            g[name] = read_field(ds, name) if name in ds.variables \
                else g[name.split('_')[0]]
        g['inversion_year'] = float(getattr(ds, 'inversion_year', 2000.0))
        # no inversion (frost: invert: false, no inversion_year): compare the
        # forward speeds in the year of the velocity product, not in 2000
        g['inverted'] = not (getattr(ds, 'velocity_fit', 'yes') == 'skipped'
                             and 'inversion_year' not in ds.ncattrs())
        if not g['inverted']:
            from frost.preprocess.igm_inversion import velocity_year
            g['inversion_year'] = velocity_year(
                os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc'))
        g['cost_data'] = float(ds['da_cost_data'][:]) if 'da_cost_data' in ds.variables else np.nan
        g['cost_reg'] = float(ds['da_cost_reg'][:]) if 'da_cost_reg' in ds.variables else np.nan
        g['x'] = np.array(ds['x'][:], dtype=float)
        g['y'] = np.array(ds['y'][:], dtype=float)
    g['dx'] = abs(g['x'][1] - g['x'][0])
    g['extent'] = [0, (g['x'][-1] - g['x'][0]) / 1e3, 0, (g['y'][-1] - g['y'][0]) / 1e3]

    rgi_id = os.path.basename(os.path.normpath(rgi_dir))
    with Dataset(os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc')) as ds:
        dhdt_input = np.nan_to_num(read_field(ds, 'dhdt'))
        for name in ['thk_source', 'thk_source_year', 'velocity_source', 'velocity_source_year']:
            g[name] = str(getattr(ds, name, ''))
    # the run the glacier comes from (combined results are links into runs)
    g['experiment'] = os.path.basename(os.path.dirname(os.path.realpath(rgi_dir)))
    glathida = os.path.join(rgi_dir, 'Preprocess', 'data', rgi_id, 'glathida_data.csv')
    g['thkobs'] = thkobs_at_year(glathida, g['x'], g['y'], dhdt_input, 2000)[0] \
        if os.path.exists(glathida) else np.full(g['thk'].shape, np.nan)

    provider = ObservationProvider(rgi_dir, rgi_id, settings['elevation_step'],
                                   None, False, settings['model_error'])
    g['provider'] = provider

    # Speed regions: ice of the inversion geometry with observed speeds,
    # split in thirds by the surface of 2000
    region = (g['thk_inversion'] > 1) & np.isfinite(g['velsurfobs_mag']) \
        & (g['velsurfobs_mag'] > 0)
    edges = np.nanpercentile(g['usurf'][region], [0, 100 / 3, 200 / 3, 100])
    g['region_edges'] = edges
    g['regions'] = {'glacier': region}
    for name, lo, hi in zip(['lower', 'middle', 'upper'], edges[:-1], edges[1:]):
        g['regions'][name] = region & (g['usurf'] >= lo) & (g['usurf'] <= hi)

    members = sorted(glob.glob(os.path.join(rgi_dir, 'Ensemble', 'Member_*',
                                            'outputs', 'output.nc')))
    speeds, band_speed, volumes, band_dhdt, band_smb, band_div, band_gone = [], [], [], [], [], [], []
    # speed maps: mean over the members that still have ice in the cell
    speed_sum = {'start': 0, 'inversion': 0, 'end': 0}
    speed_count = {'start': 0, 'inversion': 0, 'end': 0}
    dhdt_sum = 0
    for path in members:
        with Dataset(path) as ds:
            years = np.array(ds['time'][:], dtype=float)
            usurf = np.array(ds['usurf'][:], dtype=float)
            thk = np.array(ds['thk'][:], dtype=float)
            speed = np.array(ds['velsurf_mag'][:], dtype=float)
            smb = np.array(ds['smb'][1:], dtype=float).mean(0)
            div = np.array(ds['divflux'][1:], dtype=float).mean(0)
        k = int(np.argmin(np.abs(years - g['inversion_year'])))
        moving = thk > 1
        speeds.append([[np.nanmean(speed[t][mask & moving[t]]) if (mask & moving[t]).any()
                        else np.nan for mask in g['regions'].values()]
                       for t in range(len(years))])
        valid = provider.valid
        total = provider.averaging @ np.where(moving, speed, 0.0)[:, valid].T
        count = provider.averaging @ moving[:, valid].T.astype(float)
        band_speed.append((np.where(count > 0.01, total / np.maximum(count, 0.01), np.nan)).T)
        volumes.append(thk.sum((1, 2)) * g['dx'] ** 2 * 1e-9)
        for key, t in [('start', 0), ('inversion', k), ('end', -1)]:
            speed_sum[key] = speed_sum[key] + np.where(moving[t], speed[t], 0.0)
            speed_count[key] = speed_count[key] + moving[t]
        dhdt = (usurf[-1] - usurf[0]) / provider.period
        dhdt_sum = dhdt_sum + dhdt
        band_dhdt.append(provider.band_mean(dhdt))
        band_smb.append(provider.band_mean(smb))
        band_div.append(provider.band_mean(div))
        band_gone.append(provider.band_mean((thk[-1] < 1).astype(float)))
    n = len(members)
    g['n_members'] = n
    g['years'] = years
    g['inv_index'] = k
    g['speeds'] = np.array(speeds)            # member, year, region
    g['band_speed'] = np.array(band_speed)    # member, year, band
    g['volumes'] = np.array(volumes)          # member, year
    g['speed_maps'] = {key: np.where(speed_count[key] > 0, value / np.maximum(speed_count[key], 1),
                                     np.nan) for key, value in speed_sum.items()}
    g['dhdt_model'] = dhdt_sum / n
    g['band_dhdt'] = np.array(band_dhdt)      # member, band
    g['band_smb'] = np.array(band_smb)
    g['band_div'] = np.array(band_div)
    g['band_gone'] = np.array(band_gone)
    g['band_centre'] = provider.band_mean(g['usurf'])
    g['band_weights'] = np.bincount(provider.pixel_band, minlength=provider.num_bins) \
        / provider.valid.sum()
    return g


# -------------------------------------------------------------- checks

def high_pass(field, dx, mask=None):
    """Field minus its Gaussian smoothing (sigma SMOOTHING); with a mask,
    the smoothing is normalised over the masked cells only."""
    sigma = SMOOTHING / dx
    if mask is None:
        return field - gaussian_filter(field, sigma)
    values = np.where(mask, field, 0.0)
    weight = gaussian_filter(mask.astype(float), sigma)
    smooth = gaussian_filter(values, sigma) / np.maximum(weight, 1e-6)
    return np.where(mask, field - smooth, np.nan)


def bed_checks(g):
    """Bed roughness on the ice against the ice-free terrain, single-cell
    pits deeper than 5 m, ice thickness in the outline cells, and the
    thickness against GlaThiDa."""
    bed, ice = g['topg'], g['ice']
    residual = high_pass(bed, g['dx'])
    inner_ice = binary_erosion(ice, iterations=2)
    free = ~binary_dilation(ice, iterations=3)
    padded = np.pad(bed, 1, mode='edge')
    neighbours = np.min([padded[1 + i:padded.shape[0] - 1 + i, 1 + j:padded.shape[1] - 1 + j]
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], axis=0)
    pits = inner_ice & (bed < neighbours - 5)
    margin = ice & ~binary_erosion(ice)
    observed = ice & np.isfinite(g['thkobs']) & (g['thkobs'] > 0)
    rough_ice = np.sqrt(np.nanmean(residual[inner_ice] ** 2))
    rough_free = np.sqrt(np.nanmean(residual[free] ** 2)) if free.any() else np.nan
    diff = g['thk'][observed] - g['thkobs'][observed]
    return {
        'bed_roughness_ratio': rough_ice / rough_free,
        'bed_pits': int(pits.sum()),
        'margin_thk_m': float(np.mean(g['thk'][margin])),
        'thk_cells_glathida': int(observed.sum()),
        'thk_bias_glathida_m': float(diff.mean()) if observed.any() else np.nan,
        'thk_rmse_glathida_m': float(np.sqrt(np.mean(diff ** 2))) if observed.any() else np.nan,
        'thk_r_glathida': float(np.corrcoef(g['thk'][observed], g['thkobs'][observed])[0, 1])
        if observed.sum() > 2 else np.nan,
    }


TAU_KEYS = ['tau_median', 'tau_at_bounds_pct', 'tau_roughness_dec', 'tau_noise_corr',
            'sliding_share_median', 'vel_misfit_rms']
SPEED_KEYS = ['speed_inverted', 'speed_forward_2000', 'speed_forward_obs_year',
              'speed_forward_2020', 'speed_ratio_obs', 'speed_start_ratio',
              'speed_change_2000_2020_pct']


def tau_checks(g, bounds):
    """Share of the ice at the bounds, small-scale roughness of log10
    tau_ref and its correlation with that of the observed speed (tau_ref
    copying observation noise), sliding share, velocity misfit."""
    if g['no_flow']:
        return dict.fromkeys(TAU_KEYS, np.nan)
    ice = (g['thk_inversion'] > 1) & g['ice']
    tau = g['tau_ref']
    lo, hi = bounds
    at_bounds = ice & ((tau <= lo * 1.02) | (tau >= hi * 0.98))
    log_tau_hp = high_pass(np.log10(np.maximum(tau, 1e-6)), g['dx'], ice)
    obs = g['velsurfobs_mag']
    moving = ice & np.isfinite(obs) & (obs > 0.5)
    log_obs_hp = high_pass(np.log10(np.where(moving, obs, 1.0)), g['dx'], moving)
    both = moving & np.isfinite(log_tau_hp) & np.isfinite(log_obs_hp)
    corr = np.corrcoef(log_tau_hp[both], log_obs_hp[both])[0, 1] if both.sum() > 10 else np.nan
    misfit = g['velsurf_mag'] - obs
    sliding = g['velbase_mag'] / np.maximum(g['velsurf_mag'], 1e-3)
    return {
        'tau_median': float(np.median(tau[ice])),
        'tau_at_bounds_pct': 100 * at_bounds.sum() / ice.sum(),
        'tau_roughness_dec': float(np.sqrt(np.nanmean(log_tau_hp[ice] ** 2))),
        'tau_noise_corr': corr,
        'sliding_share_median': float(np.nanmedian(sliding[moving])),
        'vel_misfit_rms': float(np.sqrt(np.nanmean(misfit[moving] ** 2))),
    }


def velocity_checks(g):
    region = g['regions']['glacier']
    observed = np.nanmean(g['velsurfobs_mag'][region])
    obs = g['velsurfobs_mag'][g['ice'] & np.isfinite(g['velsurfobs_mag'])
                              & (g['velsurfobs_mag'] > 0)]
    p99 = float(np.percentile(obs, 99)) if obs.size else np.nan
    if g['no_flow']:
        return {'speed_obs': float(observed), 'velocity_p99_ice': p99,
                'ice_flow': 'none', **dict.fromkeys(SPEED_KEYS, np.nan)}
    series = np.nanmedian(g['speeds'][:, :, 0], axis=0)
    return {
        'speed_obs': float(observed),
        'speed_inverted': float(np.nanmean(g['velsurf_mag'][region])),
        'speed_forward_2000': float(series[0]),
        'speed_forward_obs_year': float(series[g['inv_index']]),
        'speed_forward_2020': float(series[-1]),
        'speed_ratio_obs': float(series[g['inv_index']] / observed),
        'speed_start_ratio': float(series[0] / series[g['inv_index']]),
        'speed_change_2000_2020_pct': float(100 * (series[-1] / series[0] - 1)),
        'velocity_p99_ice': p99, 'ice_flow': 'igm',
    }


def dhdt_checks(g):
    provider = g['provider']
    n = provider.num_bins
    obs = provider.observation[:n]
    model = g['band_dhdt'].mean(0)
    residual = model - obs
    total_cov = provider.covariance[:n, :n]
    dead = (g['band_gone'].min(0) > 0.99) & (g['band_dhdt'].std(0) < 1e-3)
    members = g['band_dhdt'] @ g['band_weights']
    return {
        'dhdt_obs': float(g['band_weights'] @ obs),
        'dhdt_obs_sigma': provider.glacier_mean_sigma(),
        'dhdt_model': float(members.mean()),
        'dhdt_model_std': float(members.std()),
        'dhdt_chi2_data': float(residual @ np.linalg.solve(provider.data_covariance, residual) / n),
        'dhdt_chi2_total': float(residual @ np.linalg.solve(total_cov, residual) / n),
        'bands': n,
        'dead_bands': int(dead.sum()),
        'volume_change_pct': float(100 * (np.median(g['volumes'][:, -1])
                                          / np.median(g['volumes'][:, 0]) - 1)),
    }


def overview_data(g):
    """What the overview pages need of one glacier (kept in overview.pkl)."""
    provider = g['provider']
    observed = g['ice'] & np.isfinite(g['thkobs']) & (g['thkobs'] > 0)
    centre = g['band_centre']
    regions = list(g['regions'])
    return {
        'thk_obs': g['thkobs'][observed], 'thk_model': g['thk'][observed],
        'band_rel_elevation': (centre - centre.min()) / max(np.ptp(centre), 1),
        'band_residual': np.median(g['band_dhdt'], 0) - provider.observation[:provider.num_bins],
        'band_obs': provider.observation[:provider.num_bins],
        'band_model': np.median(g['band_dhdt'], 0),
        'band_weights': g['band_weights'],
        'years': g['years'],
        'speed_series': np.nanmedian(g['speeds'][:, :, 0], axis=0),
        'speed_obs_regions': [np.nanmean(g['velsurfobs_mag'][g['regions'][r]]) for r in regions],
        'speed_forward_regions': [np.nanmedian(g['speeds'][:, g['inv_index'], i])
                                  for i in range(len(regions))],
        'regions': regions,
        'no_flow': g['no_flow'],
    }


# ------------------------------------------------------- glacier pages

def show(ax, g, field, title, cmap='viridis', norm=None, label='', **kwargs):
    # zorder above the grid (0.5): the grid shows where the field is NaN
    image = ax.imshow(field, origin='lower', extent=g['extent'], cmap=cmap,
                      norm=norm, interpolation='nearest', zorder=1, **kwargs)
    ax.contour(g['ice'].astype(float), levels=[0.5], colors=fg(), linewidths=0.5,
               origin='lower', extent=g['extent'], zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('km', fontsize=8)
    ax.set_ylabel('km', fontsize=8)
    ax.tick_params(labelsize=7)
    # colorbar as high as the map, not as the (taller) axes box
    colorbar = plt.colorbar(image, cax=make_axes_locatable(ax).append_axes('right', '4%', pad=0.08))
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.outline.set_visible(False)
    return image


def truncated(name):
    """Colormap without its end that vanishes in the background (the dark
    end on black pages, the light end on white ones)."""
    lo, hi = (0.3, 1.0) if SUFFIX else (0.0, 0.8)
    return matplotlib.colors.ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, 256)))


def gradient(name, n):
    return list(truncated(name)(np.linspace(0, 1, n)))


def masked(field, mask):
    return np.where(mask, field, np.nan)


def small(ax):
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def muted():
    return '0.6' if SUFFIX else '0.35'


def caption(ax, text, offset=30, size=6.2):
    """Caption below the grid cell of the axes (below its tick labels and x
    label), wrapped to the width of the cell: equal-aspect axes shrink inside
    their cell, the caption keeps to the cell."""
    fig = ax.figure
    box = ax.get_subplotspec().get_position(fig) if ax.get_subplotspec() else ax.get_position()
    chars = max(30, int(box.width * fig.get_figwidth() * 72 / (0.6 * size)))
    fig.text(box.x0, box.y0 - offset / 72 / fig.get_figheight(),
             '\n'.join(textwrap.wrap(text, chars)), va='top', ha='left', fontsize=size,
             color=muted())


def page_header(fig, name, rgi_id, subtitle, numbers):
    fig.text(0.06, 0.975, f'{name}  ({rgi_id})', fontsize=13, weight='bold', va='top')
    fig.text(0.06, 0.952, subtitle, fontsize=9.5, va='top')
    for i, line in enumerate(textwrap.wrap(numbers, 104)):
        fig.text(0.06, 0.933 - 0.013 * i, line, fontsize=8, va='top',
                 color='0.6' if SUFFIX else '0.3')


def emptiest_corner(mask, size=0.33):
    """Axes bounds of the inset in the corner of the map with the least ice."""
    ny, nx = mask.shape
    best = None
    for x0 in (0.12, 1 - size - 0.02):
        for y0 in (0.1, 1 - size - 0.02):
            rows = slice(int(y0 * ny), int((y0 + size) * ny) + 1)
            cols = slice(int(x0 * nx), int((x0 + size) * nx) + 1)
            share = mask[rows, cols].mean()
            if best is None or share < best[0]:
                best = (share, [x0, y0, size, size])
    return best[1]


def thickness_panel(ax, g, stats):
    """Thickness map with GlaThiDa dots, model vs GlaThiDa as inset."""
    vmax = np.nanmax(g['thk'][g['ice']])
    show(ax, g, masked(g['thk'], g['ice']),
         'Ice thickness 2000',
         cmap='Blues', label='m', vmin=0, vmax=vmax)
    observed = g['ice'] & np.isfinite(g['thkobs']) & (g['thkobs'] > 0)
    iy, ix = np.nonzero(observed)
    if len(iy):
        ax.scatter(g['extent'][1] * ix / (len(g['x']) - 1), g['extent'][3] * iy / (len(g['y']) - 1),
                   c=g['thkobs'][iy, ix], cmap='Blues', vmin=0, vmax=vmax, s=5,
                   edgecolors=fg(), linewidths=0.2)
        inset = ax.inset_axes(emptiest_corner(g['ice']))
        inset.patch.set_alpha(0.85)
        inset.grid(lw=0.3)
        inset.scatter(g['thkobs'][observed], g['thk'][observed], s=1.5, alpha=0.5, color='C0', lw=0)
        top = max(g['thkobs'][observed].max(), g['thk'][observed].max())
        inset.plot([0, top], [0, top], '--', color=fg(), lw=0.6)
        inset.set_xlim(0, top)
        inset.set_ylim(0, top)
        inset.set_xlabel('GlaThiDa (m)', fontsize=5, labelpad=1)
        inset.set_ylabel('model (m)', fontsize=5, labelpad=1)
        inset.tick_params(labelsize=4.5, length=0, pad=1)
        inset.text(0.05, 0.95, f'r {stats["thk_r_glathida"]:.2f}', transform=inset.transAxes,
                   fontsize=5, va='top')


def speed_map_panel(ax, g):
    """Speed misfit in the inversion year."""
    year = g['years'][g['inv_index']]
    region = g['regions']['glacier']
    difference = masked(g['speed_maps']['inversion'] - g['velsurfobs_mag'], region)
    limit = max(np.nanpercentile(np.abs(difference), 95), 1)
    show(ax, g, difference, f'Speed misfit {year:.0f}', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m/yr')


def dhdt_map_panel(ax, g):
    provider = g['provider']
    valid = provider.valid
    difference = g['dhdt_model'] - provider.dhdt
    limit = max(np.nanpercentile(np.abs(difference[valid]), 95), 0.5)
    show(ax, g, masked(difference, valid), 'dh/dt misfit 2000-2020', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m/yr')


def band_panel(ax, g):
    """Band profile of dh/dt: the observation with its data error, the
    calibration target (with the model error), the posterior members, and
    minus the flux divergence (zero without ice flow)."""
    provider = g['provider']
    n = provider.num_bins
    centre = g['band_centre']
    obs = provider.observation[:n]
    data_sigma = np.sqrt(np.diag(provider.data_covariance))
    total_sigma = np.sqrt(np.diag(provider.covariance[:n, :n]))
    ax.axvline(0, color=fg(), lw=0.6)
    ax.fill_betweenx(centre, obs - 2 * total_sigma, obs + 2 * total_sigma, color='gray', alpha=0.3,
                     lw=0, label='target: obs. +-2 sigma with model error')
    ax.errorbar(obs, centre, xerr=2 * data_sigma, fmt='o', color=fg(), ms=2, lw=0.7,
                label='observed (Hugonnet) +-2 sigma')
    ax.fill_betweenx(centre, np.percentile(g['band_dhdt'], 5, 0), np.percentile(g['band_dhdt'], 95, 0),
                     color='C1', alpha=0.4, lw=0, label='model, members 5-95 %')
    ax.plot(np.median(g['band_dhdt'], 0), centre, 'C1-', label='model, median')
    if not g['no_flow']:
        ax.plot(-g['band_div'].mean(0), centre, 'C4-', lw=1, label='minus flux divergence')
    dead = (g['band_gone'].min(0) > 0.99) & (g['band_dhdt'].std(0) < 1e-3)
    if dead.any():
        ax.plot(np.median(g['band_dhdt'], 0)[dead], centre[dead], 'rx', ms=7,
                label='melted out in every member')
    ax.set_xlabel('dh/dt 2000-2020 (m/yr)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title('Band profile of dh/dt', fontsize=9)
    ax.legend(fontsize=5.5, loc='upper left', frameon=True, framealpha=0.8, facecolor=bg(),
              edgecolor='none')
    small(ax)


def observed_band_speed(g):
    provider = g['provider']
    field = g['velsurfobs_mag']
    values = np.where(np.isfinite(field), field, 0.0)
    count = provider.band_mean(np.isfinite(field).astype(float))
    return provider.band_mean(values) / np.maximum(count, 1e-6)


def band_speed_panel(ax, g):
    """Speed by elevation band, every year of the forward runs, and the
    observed; without ice flow only the observed."""
    centre = g['band_centre']
    if not g['no_flow']:
        band_speed = np.nanmedian(g['band_speed'], 0)           # year, band
        cmap = truncated('magma')
        norm = matplotlib.colors.Normalize(g['years'][0], g['years'][-1])
        for t, values in enumerate(band_speed):
            ax.plot(values, centre, color=cmap(norm(g['years'][t])), lw=0.8)
        colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.8)
        colorbar.locator = matplotlib.ticker.MaxNLocator(integer=True)
        colorbar.update_ticks()
        colorbar.set_label('Year of the forward runs', fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.outline.set_visible(False)
    ax.plot(observed_band_speed(g), centre, '--', color=fg(), lw=1.5, label='ITS_LIVE 2014-2022')
    ax.set_xlabel('Band mean surface speed (m/yr)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title('Speed by elevation band', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    small(ax)


def smb_profile_panel(ax, g):
    """Posterior SMB (all members) against the GLAMOS / WGMS mean annual
    mass balance per elevation bin 2000-2019, with minus the flux divergence
    of the bands (SMB - flux divergence = dh/dt)."""
    from evaluate import member_smb
    from reference_profiles import mass_balance_profiles
    rgi_id = os.path.basename(os.path.normpath(g['rgi_dir']))
    profiles = mass_balance_profiles([rgi_id])
    ax.set_title('SMB against GLAMOS / WGMS', fontsize=9)
    elevation = np.linspace(g['band_centre'].min(), g['band_centre'].max(), 60)
    with open(os.path.join(g['rgi_dir'], 'calibration_results.json')) as f:
        smb = member_smb(json.load(f), elevation,
                         os.path.join(g['rgi_dir'], 'climate_historical.nc'))
    ax.axvline(0, color=fg(), lw=0.6)
    ax.fill_betweenx(elevation, np.percentile(smb, 5, 0), np.percentile(smb, 95, 0), color='C2',
                     alpha=0.4, lw=0, label='posterior SMB, members 5-95 %')
    ax.plot(smb.mean(0), elevation, 'C2-', label='posterior SMB, mean')
    if not g['no_flow']:
        ax.plot(-0.91 * g['band_div'].mean(0), g['band_centre'], 'C4-', lw=1,
                label='minus flux divergence')
    if rgi_id in profiles:
        reference = profiles[rgi_id]
        ax.plot(reference.mb, reference.elevation, 'o', color=fg(), ms=3,
                label=f'{reference.source.iloc[0]} (bins with >= 10 years)')
    else:
        ax.text(0.5, 0.5, 'no reference profile', transform=ax.transAxes, ha='center')
    ax.set_xlabel('m w.e./yr (model: m ice x 0.91)')
    ax.set_ylabel('Elevation (m)')
    ax.legend(fontsize=5.5, loc='upper left', frameon=True, framealpha=0.8, facecolor=bg(),
              edgecolor='none')
    small(ax)


def observed_speed_map_panel(ax, g):
    """ITS_LIVE speed (glaciers without ice flow: no modelled speed)."""
    speed = masked(g['velsurfobs_mag'], g['ice'])
    show(ax, g, speed, 'Observed speed', cmap=truncated('viridis'), label='m/yr', vmin=0,
         vmax=max(np.nanpercentile(speed, 99), 1))


def plot_glacier(g, stats, path, name, rgi_id):
    """One A4 page, 3 x 2: thickness with GlaThiDa and speed by elevation
    band; band profile of dh/dt and SMB against GLAMOS / WGMS (both with
    minus the flux divergence); dh/dt and speed misfit maps. Without ice
    flow the same, with the observed speed only."""
    numbers = (f'Thickness vs GlaThiDa: r {stats["thk_r_glathida"]:.2f}, '
               f'{stats["thk_cells_glathida"]} cells.  ')
    if not g['no_flow']:
        year = g['years'][g['inv_index']]
        numbers += (f'Speed (m/yr): observed {stats["speed_obs"]:.1f}, forward 2000 '
                    f'{stats["speed_forward_2000"]:.1f}, {year:.0f} {stats["speed_forward_obs_year"]:.1f}, '
                    f'2020 {stats["speed_forward_2020"]:.1f}.  ')
    numbers += (f'dh/dt (m/yr): observed {stats["dhdt_obs"]:.2f} +- {stats["dhdt_obs_sigma"]:.2f}, '
                f'model {stats["dhdt_model"]:.2f} +- {stats["dhdt_model_std"]:.2f}; band chi2/n '
                f'{stats["dhdt_chi2_data"]:.1f} (data error), {stats["dhdt_chi2_total"]:.1f} (with model '
                f'error); {stats["dead_bands"]} band{"" if stats["dead_bands"] == 1 else "s"} melted out.  '
                f'Volume {stats["volume_change_pct"]:+.0f} % 2000-2020.')
    fig = plt.figure(figsize=A4)
    # fixed grid (no tight_layout): room below every panel for its caption
    grid = fig.add_gridspec(3, 2, left=0.09, right=0.92, top=0.86, bottom=0.07,
                            hspace=0.62, wspace=0.42)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(3)])
    year = g['years'][g['inv_index']]
    product = f'{PRODUCT_NAMES.get(g["thk_source"], g["thk_source"])} ({g["thk_source_year"]})' \
        if g['thk_source'] else 'the thickness product'
    budget = ('' if g['no_flow'] else
              ' Purple: minus the flux divergence (ensemble mean), dh/dt = SMB - flux divergence.')
    captions = {
        'thickness': f'Ice thickness of 2000: {product} moved to 2000 with the Hugonnet dh/dt (fixed '
                     'in the inversion). Dots: GlaThiDa surveys moved to 2000, validation only. Inset: '
                     'model against GlaThiDa per cell (dashed 1:1).',
        'band_speed': ('ITS_LIVE 2014-2022 band-mean surface speed; without ice flow there is no '
                       'modelled speed.' if g['no_flow'] else
                       'Band-mean surface speed in every year of the forward runs (median of the '
                       'members, colour: year) and ITS_LIVE 2014-2022 (dashed).'),
        'band': 'Band means of dh/dt 2000-2020: observed +-2 sigma (bars: data error), the calibration '
                'target (grey: with the model error), model median and 5-95 % of the members.'
                + (' Without ice flow dh/dt = SMB until the ice melts out.' if g['no_flow'] else budget),
        'smb': 'Posterior SMB (mean and 5-95 % of the members) against the mean annual mass balance '
               'of GLAMOS / WGMS per elevation bin 2000-2019 (bins with >= 10 years).' + budget,
        'dhdt': 'Elevation change 2000-2020, ensemble mean minus Hugonnet et al. (2021). Red: the model '
                'thins too little.',
        'speed': f'Forward surface speed in {year:.0f} (mean of the members that still have ice) minus '
                 'ITS_LIVE 2014-2022. Red: the model is too fast.',
        'observed_speed': f'ITS_LIVE 2014-2022 surface speed; p99 on the ice '
                          f'{stats["velocity_p99_ice"]:.1f} m/yr < 5 m/yr, hence no ice flow.',
    }
    if g['no_flow']:
        page_header(fig, name, rgi_id,
                    f'Without ice flow: thickness 2000 and elevation change 2000-2020 '
                    f'({g["n_members"]} posterior members)',
                    numbers + f'  No ice flow: ITS_LIVE p99 on the ice '
                              f'{stats["velocity_p99_ice"]:.1f} m/yr.')
        last = (observed_speed_map_panel, 'observed_speed')
    else:
        page_header(fig, name, rgi_id,
                    f'Thickness 2000, speed and elevation change 2000-2020 '
                    f'({g["n_members"]} posterior members)', numbers)
        last = (speed_map_panel, 'speed')
    panels = [(thickness_panel, 'thickness'), (band_speed_panel, 'band_speed'), (band_panel, 'band'),
              (smb_profile_panel, 'smb'), (dhdt_map_panel, 'dhdt'), last]
    for i, (ax, (panel, key)) in enumerate(zip(axes.flat, panels)):
        panel(ax, g, stats) if panel is thickness_panel else panel(ax, g)
        caption(ax, f'({"abcdef"[i]}) {captions[key]}')
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------ overview pages

def no_flow(summary):
    """Glaciers calibrated without ice flow (bool per row)."""
    if 'ice_flow' not in summary:
        return np.zeros(len(summary), dtype=bool)
    return (summary.ice_flow == 'none').to_numpy()


def labels(summary):
    """Short names; * marks the glaciers without ice flow."""
    names = list(summary.glacier.str.replace('_', ' ').str.replace('Glacier de ', '')
                 .str.replace('Ghiacciaio d', 'G. d').str[:16])
    return [f'{n} *' if flag else n for n, flag in zip(names, no_flow(summary))]


NO_FLOW_NOTE = '* without ice flow (ITS_LIVE p99 on the ice < 5 m/yr): dh/dt = SMB'

# Glacier identity on every page: matplotlib tab10, then a lighter and a
# darker shade of each colour (30 colours), in the order of glaciers.txt;
# each block of 10 also gets its own marker and line style
MARKERS = ['o', '^', 's']
LINESTYLES = ['-', '--', ':']


def shade(color, amount):
    """Mix a colour with white (amount > 0) or black (amount < 0)."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    target = 1.0 if amount > 0 else 0.0
    return tuple(rgb + abs(amount) * (target - rgb))


def palette():
    base = plt.get_cmap('tab10').colors
    # on black pages a darker shade would vanish: darken less
    darker = -0.2 if SUFFIX else -0.35
    return list(base) + [shade(c, 0.45) for c in base] + [shade(c, darker) for c in base]


# Products in the attributes of input.nc (thk_source, velocity_source)
PRODUCT_NAMES = {'iceboost': 'IceBoost v2.0', 'consensus': 'Farinotti et al. (2019) consensus',
                 'millan': 'Millan et al. (2022)', 'itslive': 'ITS_LIVE v2'}


def glacier_style(summary):
    """Colour (RGBA rows), marker and line style of every glacier, by its
    row in the summary: the same on every page."""
    colours = palette()
    n = len(summary)
    colors = np.array([matplotlib.colors.to_rgba(colours[i % len(colours)]) for i in range(n)])
    group = [(i // 10) % len(MARKERS) for i in range(n)]
    return colors, [MARKERS[k] for k in group], [LINESTYLES[k] for k in group]


def glacier_legend(fig, summary, colors, markers, y=0.012, ncol=5, styles=None):
    """Legend of all glaciers (colour and marker, and line style for pages
    with lines) along the bottom of a page."""
    from matplotlib.lines import Line2D
    styles = styles or ['none'] * len(colors)
    handles = [Line2D([], [], ls=s, lw=1, marker=m, color=c, markersize=4.5)
               for c, m, s in zip(colors, markers, styles)]
    fig.legend(handles, labels(summary), loc='lower center', bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=6, handletextpad=0.3, columnspacing=1.0, frameon=False)


def plain_log_axis(axis, ticks):
    """Log axis with plain tick labels (1, 2, 3 instead of 2x10^0)."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def page_title(fig, title, note=None):
    fig.text(0.06, 0.975, title, fontsize=13, weight='bold', va='top')
    for i, line in enumerate(textwrap.wrap(note or '', 112)):
        fig.text(0.06, 0.952 - 0.012 * i, line, fontsize=7, va='top', color=muted())


def overview_grid(fig, top=0.9, bottom=0.15, hspace=0.42):
    """2 x 2 axes with room below each for its caption (and left of each
    for the glacier names of the bar charts)."""
    grid = fig.add_gridspec(2, 2, left=0.16, right=0.96, top=top, bottom=bottom,
                            hspace=hspace, wspace=0.55)
    return np.array([[fig.add_subplot(grid[r, c]) for c in range(2)] for r in range(2)])


def flowing(summary, data, colors, markers, styles):
    """The glaciers with ice flow; colours, markers, styles stay with them."""
    keep = ~no_flow(summary)
    return (summary[keep].reset_index(drop=True), [d for d, k in zip(data, keep) if k],
            colors[keep], [m for m, k in zip(markers, keep) if k],
            [s for s, k in zip(styles, keep) if k])


def stats_box(ax, text):
    ax.text(0.97, 0.03, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
            bbox=dict(facecolor=bg(), edgecolor='none', alpha=0.8, pad=1.5))


def name_bars(ax, summary, values, colors, xlabel):
    """Horizontal bars, one per glacier, in the colour of the glacier."""
    ax.barh(range(len(summary)), values, color=colors, height=0.7)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(labels(summary), fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', visible=False)
    small(ax)
    ax.tick_params(axis='y', labelsize=6)


# ----------------------------------------------- panels over all glaciers

def thickness_scatter(ax, data, colors):
    """Model thickness against GlaThiDa per cell, all glaciers."""
    obs = np.concatenate([d['thk_obs'] for d in data])
    model = np.concatenate([d['thk_model'] for d in data])
    ax.set_title('Thickness against GlaThiDa', fontsize=9)
    if not len(obs):
        ax.text(0.5, 0.5, 'no GlaThiDa cells', transform=ax.transAxes, ha='center')
        return
    for d, c in zip(data, colors):
        ax.scatter(d['thk_obs'], d['thk_model'], s=1.5, color=c, alpha=0.5, lw=0)
    top = max(obs.max(), model.max())
    ax.plot([0, top], [0, top], '--', color=fg(), lw=0.8)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect('equal')
    ax.set_xlabel('GlaThiDa moved to 2000 (m)')
    ax.set_ylabel('Model 2000 (m)')
    stats_box(ax, f'r {np.corrcoef(obs, model)[0, 1]:.2f}\n'
                  f'RMSE {np.sqrt(np.mean((model - obs) ** 2)):.0f} m\nn {len(obs)}')
    small(ax)


def speed_time(ax, summary, data, colors, styles):
    """Glacier-mean forward speed relative to the observed, in time."""
    for d, c, style in zip(data, colors, styles):
        ax.plot(d['years'], d['speed_series'] / d['speed_obs_regions'][0], color=c, lw=1, ls=style)
    ax.axhline(1, color=fg(), lw=0.8)
    ax.axvspan(2014, 2022, color='gray', alpha=0.15, lw=0)
    ax.set_yscale('log')
    plain_log_axis(ax.yaxis, [0.5, 0.75, 1, 1.5, 2, 3, 5, 10])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Year')
    ax.set_ylabel('Forward / observed speed')
    ax.set_title('Glacier speed in time', fontsize=9)
    small(ax)


def dhdt_scatter(ax, summary, colors, markers):
    """Glacier-mean dh/dt, model against Hugonnet."""
    ax.errorbar(summary.dhdt_obs, summary.dhdt_model, xerr=2 * summary.dhdt_obs_sigma,
                yerr=2 * summary.dhdt_model_std, fmt='none', ecolor='0.5', lw=0.5, alpha=0.4)
    flat = no_flow(summary)
    for i, row in enumerate(summary.itertuples()):
        ax.scatter(row.dhdt_obs, row.dhdt_model, marker=markers[i], s=22, zorder=3, lw=1,
                   facecolors='none' if flat[i] else colors[i], edgecolors=colors[i])
    low = min(summary.dhdt_obs.min(), summary.dhdt_model.min()) - 0.2
    high = max(summary.dhdt_obs.max(), summary.dhdt_model.max()) + 0.2
    ax.plot([low, high], [low, high], '--', color=fg(), lw=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal')
    error = summary.dhdt_model - summary.dhdt_obs
    stats_box(ax, f'r {np.corrcoef(summary.dhdt_obs, summary.dhdt_model)[0, 1]:.2f}\n'
                  f'MAE {error.abs().mean():.2f} m/yr\nn {len(summary)}')
    ax.set_xlabel('Hugonnet 2000-2020 (m/yr)')
    ax.set_ylabel('Model 2000-2020 (m/yr)')
    ax.set_title('Glacier-mean elevation change', fontsize=9)
    small(ax)


def smb_scatter(ax, summary, profiles, colors, markers):
    """Posterior SMB against the reference per elevation bin, all glaciers."""
    model, reference = [], []
    for i, row in enumerate(summary.itertuples()):
        p = profiles.get(row.rgi_id)
        if p is None:
            continue
        ax.scatter(p['reference'], p['mean'], marker=markers[i], s=8, color=colors[i], lw=0,
                   alpha=0.8)
        model.append(p['mean'])
        reference.append(p['reference'])
    ax.set_title('SMB against GLAMOS / WGMS', fontsize=9)
    if not model:
        ax.text(0.5, 0.5, 'no reference profiles', transform=ax.transAxes, ha='center')
        return
    model, reference = np.concatenate(model), np.concatenate(reference)
    low, high = min(model.min(), reference.min()) - 0.3, max(model.max(), reference.max()) + 0.3
    ax.plot([low, high], [low, high], '--', color=fg(), lw=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal')
    stats_box(ax, f'r {np.corrcoef(reference, model)[0, 1]:.2f}\n'
                  f'RMSE {np.sqrt(np.mean((model - reference) ** 2)):.2f}\nn {len(model)} bins')
    ax.set_xlabel('GLAMOS / WGMS (m w.e./yr)')
    ax.set_ylabel('FROST posterior (m w.e./yr)')
    small(ax)


# ------------------------------------------------------------ title page

def run_configuration(experiment, flow=True):
    """EnKF block of pipeline_config.yaml and params_inversion.yaml of
    experiments/<experiment> (empty where missing). Experiments with both
    kinds of glaciers (central_europe_TI) have pipeline_config_igm.yaml and
    pipeline_config_noflow.yaml instead."""
    folder = os.path.join('experiments', experiment)
    config = os.path.join(folder, 'pipeline_config.yaml')
    if not os.path.exists(config):
        config = os.path.join(folder, f'pipeline_config_{"igm" if flow else "noflow"}.yaml')
    loaded = []
    for path in (config, os.path.join(folder, 'params_inversion.yaml')):
        content = {}
        if os.path.exists(path):
            with open(path) as f:
                content = yaml.safe_load(f) or {}
        loaded.append(content)
    return loaded[0].get('EnKF', {}) or {}, loaded[1]


def igm_version(results, summary):
    for row in summary.itertuples():
        path = os.path.join(results, row.rgi_id, 'Preprocess', 'outputs', 'oggm_shop', 'version.txt')
        if os.path.exists(path):
            with open(path) as f:
                fields = dict(line.split(': ', 1) for line in f.read().splitlines() if ': ' in line)
            return fields.get('version', '')
    return ''


def calibration_text(enkf):
    return (f'{str(enkf.get("method", "esmda")).upper()}, {enkf.get("ensemble_size", "?")} members, '
            f'{enkf.get("iterations", "?")} iterations, {enkf.get("elev_band_height", 50)} m bands, '
            f'model error {enkf.get("model_error", 0)} m/yr')


def igm_text(enkf, inversion):
    """Inversion, emulator and calibration settings of an IGM run."""
    frost = inversion.get('frost') or {}
    iceflow = inversion.get('processes', {}).get('iceflow', {})
    unified = iceflow.get('unified', {})
    field_inversion = inversion.get('assimilations', {}).get('field_inversion', {})
    tau = [v for v in field_inversion.get('variables', []) if v.get('name') == 'tau_ref']
    lam = [r.get('lam') for r in field_inversion.get('objective', {}).get('regularization', [])
           if r.get('name') == 'tau_ref']
    year = frost.get('inversion_year')
    year = 'the velocity year' if year == 'velocity' else year or 2000
    if frost.get('invert', True) is False:
        tau_ref = iceflow.get('physics', {}).get('sliding', {}).get('tau_ref', '?')
        lines = [f'Inversion: none, uniform tau_ref {tau_ref} MPa']
    else:
        bounds = f', {tau[0].get("lower_bound")}-{tau[0].get("upper_bound")} MPa' if tau else ''
        lines = [f'Inversion: tau_ref ({tau[0].get("transform", "") if tau else ""}{bounds}) '
                 f'fitted to the surface speed on the geometry of {year}'
                 + (f', smoothness {lam[0]:.0e}' if lam else '') + '; thickness fixed']
    network = os.path.basename(str(unified.get('network', {}).get('pretrained_path', 'network')))
    nbit = unified.get('nbit_init', 0) or 0
    lines.append(f'Emulator: {network}, ' + (f'retrained {nbit} iterations before the inversion'
                                            if nbit else 'frozen in the inversion') + ', '
                 + ('retrained in the forward runs (300 iterations at the start; 20 every step, '
                    'or 5 every 5 steps where 5 steps < 0.9 yr)'
                    if enkf.get('emulator_retrain') else 'frozen in the forward runs'))
    lines.append(f'Calibration: {calibration_text(enkf)}'
                 + (f', speed cap {enkf["max_velbar"]} m/yr' if enkf.get('max_velbar') else ''))
    return lines


def setup_items(summary, results):
    """(kind, text) of the setup: 'h' heading, 't' bullet; the glaciers and
    observations (left column) and the configuration (right column)."""
    flat = no_flow(summary)
    plain = [n.rstrip(' *') for n in labels(summary)]
    left = [('h', 'Glaciers')]
    area = f', {summary.area_km2.sum():.0f} km2' if 'area_km2' in summary else ''
    left.append(('t', f'{len(summary)} glaciers with GLAMOS / WGMS mass balance{area}'))
    if (~flat).any():
        left.append(('t', f'IGM ice flow on {(~flat).sum()}'
                     + (' (ITS_LIVE p99 on the ice >= 5 m/yr)' if flat.any() else '') + ': '
                     + ', '.join(n for n, f in zip(plain, flat) if not f)))
    if flat.any():
        left.append(('t', f'No ice flow on {flat.sum()} (dh/dt = SMB until the ice melts out): '
                     + ', '.join(n for n, f in zip(plain, flat) if f)))
    left.append(('h', 'Observations'))
    enkf = run_configuration(summary.experiment.iloc[0])[0] if 'experiment' in summary else {}
    left.append(('t', 'Calibrated to: Hugonnet et al. (2021) dh/dt 2000-2020 in '
                      f'{enkf.get("elev_band_height", 50)} m elevation bands'))
    if 'velocity_source' in summary and (~flat).any():
        products = sorted({f'{PRODUCT_NAMES.get(r.velocity_source, r.velocity_source)} '
                           f'{r.velocity_source_year}' for r in summary[~flat].itertuples()})
        left.append(('t', f'Inversion: surface speed of {", ".join(products)}'))
    if 'thk_source' in summary:
        products = sorted({f'{PRODUCT_NAMES.get(r.thk_source, r.thk_source)} ({r.thk_source_year})'
                           for r in summary.itertuples()})
        left.append(('t', f'Thickness: {", ".join(products)} moved to 2000 with the dh/dt'))
    left.append(('t', 'Validation only: GlaThiDa (moved to 2000), GLAMOS / WGMS SMB profiles, '
                      'ELA and gradients'))
    right = []
    if 'experiment' in summary:
        version = igm_version(results, summary[~flat]) if (~flat).any() else ''
        for experiment in summary[~flat].experiment.unique():
            enkf, inversion = run_configuration(experiment)
            right.append(('h', f'With ice flow: {experiment}'))
            right += [('t', line) for line in
                      ([f'IGM {version}'] if version else []) + igm_text(enkf, inversion)]
        for experiment in summary[flat].experiment.unique():
            enkf, _ = run_configuration(experiment, flow=False)
            right.append(('h', f'Without ice flow: {experiment}'))
            right.append(('t', f'Calibration: {calibration_text(enkf)}'))
        prior, std = enkf.get('smb_prior_mean', {}), enkf.get('smb_prior_std', {})
        if prior and 'melt_f' in prior or enkf.get('prior_from_oggm'):
            left.append(('h', 'SMB model'))
            left.append(('t', 'OGGM temperature-index model on the monthly W5E5 climate '
                              '(melt factor, precipitation factor, temperature bias); prior '
                              + ('mean from OGGM\'s calibration of each glacier'
                                 if enkf.get('prior_from_oggm') else
                                 f'melt_f {prior.get("melt_f")}, prcp_fac {prior.get("prcp_fac")}, '
                                 f'temp_bias {prior.get("temp_bias")}')
                              + f', std {std.get("melt_f")} mm w.e./(degC day), '
                                f'{std.get("prcp_fac")}, {std.get("temp_bias")} degC'))
        elif prior:
            left.append(('h', 'SMB model'))
            left.append(('t', 'ELA, ablation and accumulation gradient, constant 2000-2020; prior '
                              f'ELA {prior.get("ela")} +- {std.get("ela")} m, gradients '
                              f'{prior.get("abl_grad")} +- {std.get("abl_grad")} and '
                              f'{prior.get("acc_grad")} +- {std.get("acc_grad")} m/yr per km'))
    return left, right


def text_column(fig, x, y, items, chars=64, size=6.4):
    """Headings and bullets from y down; returns the y below the last line."""
    for kind, text in items:
        if kind == 'h':
            y -= 0.003
            fig.text(x, y, text, fontsize=8, weight='bold', va='top')
            y -= 0.0145
            continue
        for i, line in enumerate(textwrap.wrap(text, chars - 2)):
            fig.text(x, y, ('• ' if i == 0 else '  ') + line, fontsize=size, va='top')
            y -= 0.0105 * size / 6.4
        y -= 0.0015
    return y


def performance(summary, data, profiles, folder):
    """The numbers of the performance summary (title page)."""
    flat = no_flow(summary)
    m = {'glaciers': len(summary), 'igm': int((~flat).sum())}
    obs = np.concatenate([d['thk_obs'] for d in data])
    model = np.concatenate([d['thk_model'] for d in data])
    if len(obs):
        m.update(thk_r=np.corrcoef(obs, model)[0, 1], thk_rmse=np.sqrt(np.mean((model - obs) ** 2)),
                 thk_cells=len(obs), thk_glaciers=sum(len(d['thk_obs']) > 0 for d in data))
    moving = summary[~flat]
    if len(moving):
        m.update(year=moving.inversion_year.iloc[0], speed_ratio=moving.speed_ratio_obs.median(),
                 speed_inside=int((np.abs(np.log(moving.speed_ratio_obs)) <= np.log(1.33)).sum()),
                 start_ratio=moving.speed_start_ratio.median())
    error = summary.dhdt_model - summary.dhdt_obs
    m.update(dhdt_r=np.corrcoef(summary.dhdt_obs, summary.dhdt_model)[0, 1], dhdt_mae=error.abs().mean(),
             chi2=summary.dhdt_chi2_data.median(), chi2_flagged=int((summary.dhdt_chi2_data > 4).sum()))
    if profiles:
        m.update(smb_n=len(profiles), smb_rmse=np.mean([p['rmse'] for p in profiles.values()]),
                 smb_bias=np.mean([p['bias'] for p in profiles.values()]),
                 smb_inside=100 * np.mean([p['inside'] for p in profiles.values()]))
    path = os.path.join(folder, 'tables', 'glaciers.tsv')
    if os.path.exists(path):
        table = pd.read_csv(path, sep='\t')
        table = table[(table.run == 'new') & table.rgi_id.isin(summary.rgi_id)
                      & (table.ela_censored <= 0.25) & np.isfinite(table.ela_ref)]
        if len(table) > 2:
            m.update(ela_r=np.corrcoef(table.ela_ref, table.ela)[0, 1], ela_n=len(table),
                     ela_mae=np.mean(np.abs(table.ela - table.ela_ref)))
    return m


def performance_items(m):
    items = [('h', 'Performance')]
    if 'thk_r' in m:
        items.append(('t', f'Thickness: r {m["thk_r"]:.2f}, RMSE {m["thk_rmse"]:.0f} m on '
                           f'{m["thk_cells"]} GlaThiDa cells of {m["thk_glaciers"]} glaciers'))
    if 'speed_ratio' in m:
        items.append(('t', f'Velocity ({m["igm"]} glaciers with ice flow): forward speed in '
                           f'{m["year"]:g} median {m["speed_ratio"]:.2f} x ITS_LIVE, '
                           f'{m["speed_inside"]} within +-33 %; 2000 / {m["year"]:g} median '
                           f'{m["start_ratio"]:.2f}'))
    items.append(('t', f'dh/dt: glacier means r {m["dhdt_r"]:.2f}, MAE {m["dhdt_mae"]:.2f} m/yr; '
                       f'band chi2/n median {m["chi2"]:.1f} (data error), {m["chi2_flagged"]} '
                       'glaciers > 4'))
    if 'smb_rmse' in m:
        items.append(('t', f'SMB profiles ({m["smb_n"]} glaciers): RMSE {m["smb_rmse"]:.2f}, bias '
                           f'{m["smb_bias"]:+.2f} m w.e./yr; {m["smb_inside"]:.0f} % of the bins '
                           'within +-2 std of the members'))
    if 'ela_r' in m:
        items.append(('t', f'ELA against GLAMOS / WGMS: r {m["ela_r"]:.2f}, MAE {m["ela_mae"]:.0f} m '
                           f'on {m["ela_n"]} references (not capped at the glacier top)'))
    return items


def metric_tiles(fig, m, top):
    """The key numbers as tiles, two rows of four; returns the y below."""
    tiles = [(f'{m["glaciers"]}', 'glaciers', f'{m["igm"]} with IGM ice flow')]
    if 'thk_r' in m:
        tiles.append((f'{m["thk_r"]:.2f}', 'thickness r', f'vs GlaThiDa, RMSE {m["thk_rmse"]:.0f} m'))
    if 'speed_ratio' in m:
        tiles.append((f'{m["speed_ratio"]:.2f}x', f'speed {m["year"]:g} / ITS_LIVE',
                      f'median, {m["speed_inside"]} of {m["igm"]} within +-33 %'))
        tiles.append((f'{m["start_ratio"]:.2f}x', 'start-up 2000 / ' + f'{m["year"]:g}',
                      'median forward speed ratio'))
    tiles.append((f'{m["dhdt_mae"]:.2f}', 'dh/dt MAE (m/yr)', f'glacier means, r {m["dhdt_r"]:.2f}'))
    tiles.append((f'{m["chi2"]:.1f}', 'band chi2 / n', f'median, {m["chi2_flagged"]} glaciers > 4'))
    if 'smb_rmse' in m:
        tiles.append((f'{m["smb_rmse"]:.2f}', 'SMB RMSE (m w.e./yr)',
                      f'vs GLAMOS / WGMS, {m["smb_inside"]:.0f} % in +-2 std'))
    if 'ela_r' in m:
        tiles.append((f'{m["ela_mae"]:.0f} m', 'ELA MAE', f'r {m["ela_r"]:.2f}, {m["ela_n"]} references'))
    width, height, gap = 0.21, 0.07, 0.0133
    face = '#161616' if SUFFIX else '#f0f0f0'
    for i, (value, label, note) in enumerate(tiles[:8]):
        x = 0.06 + (i % 4) * (width + gap)
        y = top - (i // 4) * (height + gap)
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x, y - height), width, height, boxstyle='round,pad=0,rounding_size=0.006',
            transform=fig.transFigure, facecolor=face, edgecolor='none', zorder=0))
        fig.text(x + 0.012, y - 0.008, label, fontsize=6.5, va='top', color=muted())
        fig.text(x + 0.012, y - 0.024, value, fontsize=17, weight='bold', va='top')
        fig.text(x + 0.012, y - height + 0.007, note, fontsize=5.6, va='bottom', color=muted())
    rows = (min(len(tiles), 8) + 3) // 4
    return top - rows * (height + gap)


def title_page(summary, data, profiles, results, folder):
    fig = plt.figure(figsize=A4)
    experiment = os.path.basename(os.path.normpath(results))
    fig.text(0.06, 0.975, 'FROST calibration report', fontsize=16, weight='bold', va='top')
    fig.text(0.06, 0.95, f'{experiment}  ·  {results}  ·  {date.today().isoformat()}', fontsize=7.5,
             va='top', color=muted())
    m = performance(summary, data, profiles, folder)
    y = metric_tiles(fig, m, 0.93) - 0.015
    left, right = setup_items(summary, results)
    text_column(fig, 0.06, y, left, chars=66, size=6.8)
    text_column(fig, 0.53, y, right + performance_items(m), chars=66, size=6.8)
    return fig


# ----------------------------------------------------------- topic pages

def thickness_page(summary, data):
    fig = plt.figure(figsize=A4)
    page_title(fig, 'Thickness', 'Thickness of 2000: the thickness product moved to 2000 with the '
                                 'Hugonnet dh/dt, fixed in the inversion. GlaThiDa surveys are moved '
                                 'to 2000 with their date and the dh/dt and used for validation only.')
    axes = overview_grid(fig)
    colors, markers, _ = glacier_style(summary)
    thickness_scatter(axes[0, 0], data, colors)
    caption(axes[0, 0], '(a) Model against GlaThiDa per cell, all glaciers in their colours '
                        '(dashed 1:1).')

    ax = axes[0, 1]
    name_bars(ax, summary, summary.thk_r_glathida.fillna(0), colors, 'r with GlaThiDa')
    for i, row in enumerate(summary.itertuples()):
        ax.text(max(row.thk_r_glathida, 0) + 0.02 if pd.notna(row.thk_r_glathida) else 0.02, i,
                f'{row.thk_cells_glathida}' if row.thk_cells_glathida else 'no data',
                va='center', fontsize=5.5, color=muted())
    ax.set_xlim(0, 1.15)
    ax.set_title('Correlation per glacier', fontsize=9)
    caption(ax, '(b) Correlation of model and GlaThiDa thickness on the cells with surveys; the '
                'number is the count of cells.')

    ax = axes[1, 0]
    name_bars(ax, summary, summary.thk_rmse_glathida_m.fillna(0), colors, 'RMSE (m)')
    ax.set_title('RMSE per glacier', fontsize=9)
    caption(ax, '(c) Root mean square difference of model and GlaThiDa thickness on the cells '
                'with surveys.')

    ax = axes[1, 1]
    ax.set_title('Volume against area', fontsize=9)
    if {'area_km2', 'volume_2000_km3'} <= set(summary.columns):
        for i, row in enumerate(summary.itertuples()):
            ax.scatter(row.area_km2, row.volume_2000_km3, marker=markers[i], color=colors[i], s=22,
                       zorder=3)
        area = np.geomspace(summary.area_km2.min() / 1.5, summary.area_km2.max() * 1.5, 50)
        ax.plot(area, 0.034 * area ** 1.375, '--', color=fg(), lw=0.8, label='V = 0.034 A^1.375')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=6.5, loc='upper left')
        ax.set_xlabel('Area (km2)')
        ax.set_ylabel('Ice volume 2000 (km3)')
        small(ax)
    caption(ax, '(d) Ice volume of 2000 against the glacier area, with volume-area scaling '
                '(Bahr et al. 1997): covers the glaciers without surveys too.')
    glacier_legend(fig, summary, colors, markers, y=0.01)
    return fig


def velocity_page(summary, data):
    fig = plt.figure(figsize=A4)
    colors, markers, styles = glacier_style(summary)
    n_flat = no_flow(summary).sum()
    summary, data, colors, markers, styles = flowing(summary, data, colors, markers, styles)
    year = summary.inversion_year.iloc[0]
    inverted = summary.get('inverted', pd.Series(True, index=summary.index)).all()
    page_title(fig, 'Velocity', f'{len(summary)} glaciers with ice flow' + (
        f'; the {n_flat} without (ITS_LIVE p99 on the ice < 5 m/yr) are left out' if n_flat else '')
        + f'. Forward runs of the posterior members 2000-2020, compared with ITS_LIVE 2014-2022 in '
          f'{year:g}, the {"inversion" if inverted else "velocity"} year.')
    grid = fig.add_gridspec(2, 2, left=0.16, right=0.96, top=0.9, bottom=0.17, hspace=0.42,
                            wspace=0.55)
    names = labels(summary)

    ax = fig.add_subplot(grid[0, 0])
    for i, row in enumerate(summary.itertuples()):
        ax.scatter(row.speed_obs, row.speed_forward_obs_year, color=colors[i], marker=markers[i],
                   s=24, zorder=3)
    low, high = 0.2, 1.5 * max(summary.speed_obs.max(), summary.speed_forward_obs_year.max())
    ax.plot([low, high], [low, high], '--', color=fg(), lw=0.8)
    ax.fill_between([low, high], [low / 1.33, high / 1.33], [low * 1.33, high * 1.33],
                    color='gray', alpha=0.2, lw=0)
    for row, name in zip(summary.itertuples(), names):
        if abs(np.log(row.speed_ratio_obs)) > np.log(1.5):
            ax.annotate(name, (row.speed_obs, row.speed_forward_obs_year), fontsize=5.5,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    plain_log_axis(ax.xaxis, ticks)
    plain_log_axis(ax.yaxis, ticks)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect('equal')
    ax.set_xlabel('ITS_LIVE 2014-2022 (m/yr)')
    ax.set_ylabel(f'Forward {year:g} (m/yr)')
    ax.set_title(f'Glacier-mean speed in {year:g}', fontsize=9)
    small(ax)
    caption(ax, f'(a) Forward speed in {year:g} (median of the members) against ITS_LIVE, glacier '
                'means over the ice with observed speeds. Grey: +-33 %; names: off by more than 1.5x.')

    ax = fig.add_subplot(grid[0, 1])
    name_bars(ax, summary, summary.speed_start_ratio, colors, f'Forward 2000 / forward {year:g}')
    ax.axvline(1, color=fg(), lw=0.6)
    ax.axvline(2, color='r', ls='--', lw=0.8)
    ax.set_xscale('log')
    plain_log_axis(ax.xaxis, [1, 1.5, 2, 3, 5, 8])
    ax.set_title('Start-up of the forward runs', fontsize=9)
    caption(ax, f'(b) Forward speed of 2000 against that of {year:g}. Red dashed: 2x, the flag of a '
                'start-up shock (ice flow out of balance with the geometry of 2000).')

    ax = fig.add_subplot(grid[1, :])
    speed_time(ax, summary, data, colors, styles)
    caption(ax, '(c) Glacier-mean forward speed (median of the members) relative to ITS_LIVE, every '
                'year; grey: observation period 2014-2022. Colour, marker and line style of each '
                'glacier as in the legend (tab10, then lighter shades with dashed lines from the '
                '11th glacier on, darker shades with dotted lines from the 21st).')
    glacier_legend(fig, summary, colors, markers, y=0.01, styles=styles)
    return fig


def dhdt_page(summary, data):
    fig = plt.figure(figsize=A4)
    page_title(fig, 'Elevation change', 'Hugonnet et al. (2021) dh/dt 2000-2020, the calibration '
                                        'target, in elevation bands. ' + NO_FLOW_NOTE + ' (open markers).')
    axes = overview_grid(fig)
    colors, markers, styles = glacier_style(summary)
    dhdt_scatter(axes[0, 0], summary, colors, markers)
    caption(axes[0, 0], '(a) Glacier-mean dh/dt 2000-2020, model (+-2 std of the members) against '
                        'Hugonnet (+-2 sigma).')

    ax = axes[0, 1]
    grid = np.linspace(0, 1, 21)
    profiles = []
    for d, c, style in zip(data, colors, styles):
        ax.plot(d['band_residual'], d['band_rel_elevation'], color=c, lw=0.8, alpha=0.8, ls=style)
        profiles.append(np.interp(grid, d['band_rel_elevation'], d['band_residual']))
    ax.plot(np.median(profiles, 0), grid, '-', color=fg(), lw=2.2)
    ax.axvline(0, color=fg(), lw=0.6)
    ax.set_xlim(-3, 3)
    ax.set_xlabel('Band dh/dt, model - observed (m/yr)')
    ax.set_ylabel('Relative elevation (0 tongue, 1 top)')
    ax.set_title('Band misfit over the elevation', fontsize=9)
    small(ax)
    caption(ax, '(b) Band dh/dt misfit (median of the members) over the relative elevation; thick: '
                'median of the glaciers. > 0: the model thins too little.')

    ax = axes[1, 0]
    name_bars(ax, summary, summary.dhdt_chi2_data, colors, 'Band chi2 / n (data error)')
    ax.axvline(1, color=fg(), lw=0.6)
    ax.axvline(4, color='r', ls='--', lw=0.8)
    ax.set_xscale('log')
    plain_log_axis(ax.xaxis, [0.1, 0.3, 1, 2, 4, 10, 30])
    ax.set_title('Band misfit per glacier', fontsize=9)
    caption(ax, '(c) Chi2 per band of the model median against the observation error of the data '
                '(without the model error). 1: within the error; red dashed: 4, flagged.')

    ax = axes[1, 1]
    observed, modelled = [], []
    flat = no_flow(summary)
    for i, d in enumerate(data):
        if 'band_obs' not in d:
            continue
        size = 4 + 60 * d['band_weights']
        ax.scatter(d['band_obs'], d['band_model'], s=size, marker=markers[i], lw=0.5,
                   facecolors='none' if flat[i] else colors[i], edgecolors=colors[i], alpha=0.8)
        observed.append(d['band_obs'])
        modelled.append(d['band_model'])
    ax.set_title('Bands, all glaciers', fontsize=9)
    if observed:
        observed, modelled = np.concatenate(observed), np.concatenate(modelled)
        low = min(observed.min(), modelled.min()) - 0.3
        high = max(observed.max(), modelled.max()) + 0.3
        ax.plot([low, high], [low, high], '--', color=fg(), lw=0.8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect('equal')
        stats_box(ax, f'r {np.corrcoef(observed, modelled)[0, 1]:.2f}\nRMSE '
                      f'{np.sqrt(np.mean((modelled - observed) ** 2)):.2f} m/yr\nn {len(observed)} bands')
    ax.set_xlabel('Hugonnet band dh/dt (m/yr)')
    ax.set_ylabel('Model band dh/dt (m/yr)')
    small(ax)
    caption(ax, '(d) Band dh/dt 2000-2020 of every glacier, model median against Hugonnet; marker '
                'size: area share of the band.')
    glacier_legend(fig, summary, colors, markers, y=0.01)
    return fig


def smb_profiles(summary, results):
    """Posterior SMB (mean, std of the members) on the elevation bins of the
    GLAMOS / WGMS profile of every glacier that has one (evaluate.py,
    reference_profiles.py)."""
    from evaluate import member_smb
    from reference_profiles import mass_balance_profiles
    references = mass_balance_profiles(list(summary.rgi_id))
    profiles = {}
    for row in summary.itertuples():
        path = os.path.join(results, row.rgi_id, 'calibration_results.json')
        if row.rgi_id not in references or not os.path.exists(path):
            continue
        reference = references[row.rgi_id]
        elevation = reference.elevation.to_numpy(dtype=float)
        with open(path) as f:
            smb = member_smb(json.load(f), elevation,
                             os.path.join(results, row.rgi_id, 'climate_historical.nc'))
        mean, std, mb = smb.mean(0), smb.std(0), reference.mb.to_numpy(dtype=float)
        diff = mean - mb
        profiles[row.rgi_id] = dict(elevation=elevation, mean=mean, std=std, reference=mb,
                                    source=reference.source.iloc[0],
                                    rmse=float(np.sqrt(np.mean(diff ** 2))), bias=float(diff.mean()),
                                    inside=float(np.mean(np.abs(diff) <= 2 * std)))
    return profiles


def profile_page(summary, profiles):
    """One SMB profile panel per glacier against GLAMOS / WGMS."""
    rows = int(np.ceil(len(summary) / 5))
    fig = plt.figure(figsize=A4)
    page_title(fig, 'SMB profiles', 'Dots: mean annual mass balance per elevation bin 2000-2019 '
                                    '(GLAMOS / WGMS, bins with >= 10 years). Line and shade: posterior '
                                    'SMB, mean +- 2 std of the members (m ice x 0.91). x: m w.e./yr, '
                                    'y: elevation (m). Compares the SMB directly, without the ELA and '
                                    'gradient definitions of the references.')
    grid = fig.add_gridspec(rows, 5, left=0.07, right=0.97, top=0.9, bottom=0.08,
                            hspace=0.75, wspace=0.45)
    colors, _, _ = glacier_style(summary)
    for i, (row, name, color) in enumerate(zip(summary.itertuples(), labels(summary), colors)):
        ax = fig.add_subplot(grid[i // 5, i % 5])
        p = profiles.get(row.rgi_id)
        if p is None:
            ax.axis('off')
            ax.set_title(f'{name}\nno reference', fontsize=6.5, color=color)
            continue
        ax.axvline(0, color=fg(), lw=0.5)
        ax.fill_betweenx(p['elevation'], p['mean'] - 2 * p['std'], p['mean'] + 2 * p['std'],
                         color=color, alpha=0.3, lw=0)
        ax.plot(p['mean'], p['elevation'], '-', color=color, lw=1.2)
        ax.plot(p['reference'], p['elevation'], 'o', color=fg(), ms=2)
        ax.set_title(f'{name}', fontsize=6.5, color=color)
        # upper left: empty in every profile (high up the SMB is positive)
        ax.text(0.03, 0.97, f'{p["source"]}\nRMSE {p["rmse"]:.2f}', transform=ax.transAxes,
                fontsize=5, color=muted(), va='top')
        ax.tick_params(labelsize=5, length=0, pad=1)
    if profiles:
        rmse = np.array([p['rmse'] for p in profiles.values()])
        bias = np.array([p['bias'] for p in profiles.values()])
        inside = np.array([p['inside'] for p in profiles.values()])
        text = (f'{len(profiles)} glaciers: RMSE {rmse.mean():.2f}, bias {bias.mean():+.2f} m w.e./yr, '
                f'{100 * inside.mean():.0f} % of the bins within +-2 std.')
        fig.text(0.06, 0.03, text, fontsize=7, color=muted())
    return fig


def reference_page(summary, folder):
    """Posterior ELA and gradients against GLAMOS / WGMS (evaluate.py
    tables), and the references' own consistency (geodetic against
    glaciological mass balance). None without the tables."""
    paths = [os.path.join(folder, 'tables', name) for name in ('glaciers.tsv', 'aggregated_results.csv')]
    if not all(os.path.exists(path) for path in paths):
        return None
    table = pd.read_csv(paths[0], sep='\t')
    table = table[table.run == 'new'].set_index('rgi_id')
    aggregated = pd.read_csv(paths[1]).set_index('rgi_id')
    colors, markers, _ = glacier_style(summary)
    fig = plt.figure(figsize=A4)
    page_title(fig, 'SMB parameters against GLAMOS / WGMS',
               'Posterior mean +- 1 std; gradients in m w.e. (model m ice x 0.91). Open markers: '
               'reference ELA at the glacier top in > 25 % of the years (a lower bound), not in r, '
               'bias, MAE. * without ice flow.')
    axes = overview_grid(fig)
    panels = [('ela', 'ELA (m)', 1.0), ('abl', 'Ablation gradient (m w.e./yr per km)', 0.91),
              ('acc', 'Accumulation gradient (m w.e./yr per km)', 0.91)]
    texts = ['(a) Equilibrium line altitude, posterior against the mean of the reference '
             '2000-2019 (dashed 1:1).',
             '(b) Ablation gradient of the posterior against that fitted to the reference profile.',
             '(c) Accumulation gradient of the posterior against that fitted to the reference profile.']
    for ax, (p, label, factor), text in zip(axes.flat, panels, texts):
        xs, ys, every = [], [], []
        for i, row in enumerate(summary.itertuples()):
            if row.rgi_id not in table.index:
                continue
            g = table.loc[row.rgi_id]
            x, y, std = g[f'{p}_ref'], factor * g[p], factor * g[f'{p}_std']
            if not np.isfinite(x):
                continue
            capped = p == 'ela' and g.ela_censored > 0.25
            ax.errorbar(x, y, yerr=std, fmt='none', ecolor=colors[i], alpha=0.45, lw=0.8)
            ax.scatter(x, y, marker=markers[i], s=24, lw=1, zorder=3, edgecolors=colors[i],
                       facecolors='none' if capped else colors[i])
            every += [x, y]
            if not capped:
                xs.append(x)
                ys.append(y)
        xs, ys = np.array(xs), np.array(ys)
        lim = [min(every), max(every)]
        pad = 0.08 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, '--', color=fg(), lw=0.8)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        fmt = '.0f' if p == 'ela' else '.2f'
        stats_box(ax, f'r {np.corrcoef(xs, ys)[0, 1]:.2f}\nbias {np.mean(ys - xs):+{fmt}}\n'
                      f'MAE {np.mean(np.abs(ys - xs)):{fmt}}\nn {len(xs)}')
        ax.set_xlabel(f'GLAMOS / WGMS {label}')
        ax.set_ylabel(f'FROST {label}')
        ax.set_title(label.split(' (')[0], fontsize=9)
        small(ax)
        caption(ax, text)
    ax = axes[1, 1]
    xs, ys = [], []
    for i, row in enumerate(summary.itertuples()):
        if row.rgi_id not in aggregated.index:
            continue
        x, y = aggregated.loc[row.rgi_id, 'dhdt_mean'], aggregated.loc[row.rgi_id, 'annual_mass_balance']
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, marker=markers[i], color=colors[i], s=24, zorder=3)
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    lim = [min(xs.min(), ys.min()) - 0.15, 0.1]
    ax.plot(lim, lim, '--', color=fg(), lw=0.8, label='1:1')
    ax.plot(lim, [0.85 * v for v in lim], ':', color=fg(), lw=0.8, label='x 0.85 (density)')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(fontsize=6.5, loc='upper left')
    stats_box(ax, f'r {np.corrcoef(xs, ys)[0, 1]:.2f}\nMAE {np.mean(np.abs(ys - xs)):.2f}\nn {len(xs)}')
    ax.set_xlabel('Hugonnet dh/dt 2000-2020 (m/yr)')
    ax.set_ylabel('GLAMOS / WGMS balance (m w.e./yr)')
    ax.set_title('The references themselves', fontsize=9)
    small(ax)
    caption(ax, '(d) Geodetic (Hugonnet) against glaciological (GLAMOS / WGMS) mass balance '
                '2000-2020: the consistency of the references, no FROST result.')
    glacier_legend(fig, summary, colors, markers, y=0.01)
    return fig


TABLE = [('glacier', 'Glacier', None),
         ('ice_flow', 'Ice\nflow', '{}'),
         ('thk_r_glathida', 'Thk r', '{:.2f}'),
         ('speed_obs', 'Speed\nobserved', '{:.1f}'),
         ('speed_ratio_obs', 'Forward/\nobserved', '{:.2f}'),
         ('speed_start_ratio', '2000/\nobs. year', '{:.1f}'),
         ('dhdt_obs', 'dh/dt\nobserved', '{:.2f}'),
         ('dhdt_model', 'dh/dt\nmodel', '{:.2f}'),
         ('dhdt_chi2_data', 'Band\nchi2/n', '{:.1f}'),
         ('smb_rmse', 'SMB\nRMSE', '{:.2f}'),
         ('dead_bands', 'Melted\nbands', '{:.0f}')]


def table_page(summary):
    names = labels(summary)
    colors, _, _ = glacier_style(summary)
    cells, colours = [], []
    for name, (_, row) in zip(names, summary.iterrows()):
        line, colour = [], []
        for column, _, fmt in TABLE:
            if fmt is None:
                line.append(name)
                colour.append(TABLE_COLOURS['cell'])
                continue
            value = row.get(column, np.nan)
            line.append(fmt.format(value) if pd.notna(value) else '-')
            if column == 'ice_flow':
                line[-1] = {'igm': 'IGM', 'none': 'none'}.get(value, '-')
            flagged = column in FLAGS and pd.notna(value) and FLAGS[column][0](value)
            colour.append(TABLE_COLOURS['flag'] if flagged else TABLE_COLOURS['cell'])
        cells.append(line)
        colours.append(colour)
    fig = plt.figure(figsize=A4)
    page_title(fig, 'Summary table')
    ax = fig.add_axes([0.045, 0.25, 0.91, 0.68])
    ax.axis('off')
    widths = [0.2] + [0.8 / (len(TABLE) - 1)] * (len(TABLE) - 1)
    table = ax.table(cellText=cells, colLabels=[c[1] for c in TABLE], cellColours=colours,
                     colWidths=widths, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.3)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor(muted())
        if r == 0:
            cell.set_height(cell.get_height() * 2)
            cell.set_facecolor(TABLE_COLOURS['header'])
        # the glacier's own colour for its name, as on every other page
        cell.get_text().set_color(colors[r - 1] if r > 0 and c == 0 else fg())
        cell.get_text().set_family(MONO)  # numbers line up in the columns
        if c == 0:
            cell.get_text().set_ha('left')
            cell.get_text().set_weight('bold' if r > 0 else 'normal')
    notes = ['Speeds in m/yr (glacier means over the ice with observed speeds), dh/dt in m/yr '
             '(2000-2020), SMB RMSE in m w.e./yr against the GLAMOS / WGMS profile. Red: flagged. '
             + NO_FLOW_NOTE + '; no forward speeds.'] + \
        [f'- {text}' for _, text in FLAGS.values()] + \
        ['Bed and tau_ref, in numbers only (summary.tsv): ' + bed_tau_note(summary)]
    # notes right below the table, whatever its number of rows
    fig.canvas.draw()
    y = fig.transFigure.inverted().transform(
        table.get_window_extent(fig.canvas.get_renderer()))[0, 1] - 0.02
    for note in notes:
        for line in textwrap.wrap(note, 115):
            fig.text(0.06, y, line, fontsize=7, va='top', color=muted())
            y -= 0.014
    return fig


def bed_tau_note(summary):
    return (f'bed roughness on the ice {summary.bed_roughness_ratio.min():.1f}-'
            f'{summary.bed_roughness_ratio.max():.1f} x that of the ice-free terrain, '
            f'{summary.bed_pits.min()}-{summary.bed_pits.max()} single-cell pits per glacier; '
            f'tau_ref at its bounds on {summary.tau_at_bounds_pct.max():.1f} % of the ice at most, '
            f'small-scale correlation with the observed speed {summary.tau_noise_corr.min():.2f} to '
            f'{summary.tau_noise_corr.max():.2f} (no sign of fitting observation noise).')


def text_pages(title, paragraphs):
    """A4 pages of wrapped text: '## ' starts a heading, '- ' a bullet,
    '1. ' a numbered item; continues on a new page when one is full."""
    pages = [plt.figure(figsize=A4)]
    pages[0].text(0.08, 0.96, title, fontsize=15, weight='bold', va='top')
    y = 0.925
    for paragraph in paragraphs:
        heading = paragraph.startswith('## ')
        numbered = paragraph[:3].rstrip().rstrip('.').isdigit() and paragraph[1:3] == '. '
        bullet = paragraph.startswith('- ')
        body = paragraph[3:] if heading else paragraph[2:] if bullet else paragraph
        lines = [body] if heading else textwrap.wrap(body, 98 if (bullet or numbered) else 102) or ['']
        needed = 0.03 if heading else 0.0135 * len(lines) + 0.005
        if y - needed < 0.03:
            pages.append(plt.figure(figsize=A4))
            y = 0.95
        if heading:
            y -= 0.006
            pages[-1].text(0.08, y, body, fontsize=10, weight='bold', va='top')
            y -= 0.021
            continue
        for i, line in enumerate(lines):
            prefix = ('• ' if i == 0 else '  ') if bullet else ''
            indent = 0.012 if bullet else 0.024 if numbered and i else 0.0
            pages[-1].text(0.08 + indent, y, prefix + line, fontsize=8.2, va='top')
            y -= 0.0135
        y -= 0.005
    return pages


def findings_paragraphs(out):
    path = os.path.join(out, 'findings.md')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        blocks = [b for b in f.read().split('\n\n') if b.strip()]
    paragraphs = []
    for block in blocks:
        block = block.strip()
        if block.startswith('## '):
            paragraphs.append(block.splitlines()[0])
        elif block.startswith('# '):
            continue  # the page title
        elif block.startswith('- '):
            paragraphs += ['- ' + item.strip().replace('\n', ' ')
                           for item in block[2:].split('\n- ')]
        elif block[:2].rstrip('.').isdigit() and block[1:3] == '. ':
            items, item = [], ''
            for line in block.splitlines():
                if line[:2].rstrip('.').isdigit() and line[1:3] == '. ':
                    if item:
                        items.append(item)
                    item = line.strip()
                else:
                    item += ' ' + line.strip()
            paragraphs += items + [item]
        else:
            paragraphs.append(block.replace('\n', ' '))
    return paragraphs


def write_pdf(summary, data, out, results, settings, folder):
    path = os.path.join(out, f'report{SUFFIX}.pdf')
    if 'ela_ref_capped' not in summary:
        from reference_profiles import ela_censored_share
        share = ela_censored_share(list(summary.rgi_id))
        summary = summary.assign(ela_ref_capped=[share[r] for r in summary.rgi_id])
    profiles = smb_profiles(summary, results)
    summary = summary.assign(smb_rmse=[profiles[r]['rmse'] if r in profiles else np.nan
                                       for r in summary.rgi_id])
    flat = no_flow(summary)
    # (topic, figure); glacier pages are the PNGs
    pages = [('Overview', title_page(summary, data, profiles, results, folder))]
    findings = findings_paragraphs(out)
    if findings:
        pages += [('Findings', fig) for fig in text_pages('Findings', findings)]
    pages.append(('Thickness', thickness_page(summary, data)))
    if not flat.all():
        pages.append(('Velocity', velocity_page(summary, data)))
    pages.append(('Elevation change', dhdt_page(summary, data)))
    pages.append(('SMB profiles', profile_page(summary, profiles)))
    reference = reference_page(summary, folder)
    if reference is not None:
        pages.append(('SMB parameters', reference))
    pages.append(('Summary table', table_page(summary)))
    glaciers = [(row.glacier.replace('_', ' '), row.glacier) for row in summary.itertuples()]
    total = len(pages) + len(glaciers)
    experiment = os.path.basename(os.path.normpath(results))

    def finish(pdf, fig, topic, number, **kwargs):
        fig.text(0.035, 0.993, f'FROST calibration report  ·  {experiment}', fontsize=6.5,
                 va='top', color=muted())
        fig.text(0.965, 0.993, topic, fontsize=6.5, va='top', ha='right', color=muted())
        fig.text(0.5, 0.004, f'{number} / {total}', fontsize=7, ha='center', va='bottom',
                 color=muted())
        pdf.savefig(fig, **kwargs)
        plt.close(fig)

    with PdfPages(path) as pdf:
        for number, (topic, fig) in enumerate(pages, start=1):
            finish(pdf, fig, topic, number)
        for number, (name, folder_name) in enumerate(glaciers, start=len(pages) + 1):
            fig = plt.figure(figsize=A4)
            # the glacier page, slightly reduced: side margins, room for header and footer
            ax = fig.add_axes([0.035, 0.02, 0.93, 0.955])
            ax.imshow(plt.imread(os.path.join(out, folder_name, f'glacier{SUFFIX}.png')))
            ax.axis('off')
            finish(pdf, fig, name, number, dpi=DPI)
    return path


def write_markdown(summary, out, results):
    lines = [f'# Sanity report: {results}', '',
             'The full report is report.pdf (A4). Flags:', '']
    lines += [f'- **{column}**: {text}' for column, (_, text) in FLAGS.items()]
    lines += ['', '| Glacier | Thk r | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | '
              'dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |',
              '|---|---|---|---|---|---|---|---|']
    for _, row in summary.iterrows():
        flags = [column for column, (test, _) in FLAGS.items()
                 if pd.notna(row[column]) and test(row[column])]
        figures = f'[glacier]({row.glacier}/glacier.png)'
        lines.append(
            f'| {row.glacier} | {row.thk_r_glathida:.2f} | {row.speed_obs:.1f} / '
            f'{row.speed_forward_2000:.1f} / {row.speed_forward_obs_year:.1f} / '
            f'{row.speed_forward_2020:.1f} | {row.dhdt_obs:.2f} / {row.dhdt_model:.2f} | '
            f'{row.dhdt_chi2_data:.1f} | {row.dead_bands} | {", ".join(flags) or "-"} | {figures} |')
    with open(os.path.join(out, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', required=True, help='results folder of the run')
    parser.add_argument('--out', help='report folder (default experiments/<name>/sanity_report)')
    parser.add_argument('--glaciers', nargs='*',
                        help='RGI ids or glacier names (default: all of glaciers.txt)')
    parser.add_argument('--pdf_only', action='store_true',
                        help='only rebuild report.pdf from the figures, summary.tsv and '
                             'overview.pkl of a previous run (e.g. after editing findings.md)')
    parser.add_argument('--dark', action='store_true',
                        help='black pages: report_dark.pdf and *_dark.png (and the dark 3x3 '
                             'figure from plots/evaluation_central_europe_dark, if there)')
    args = parser.parse_args()
    if args.dark:
        global SUFFIX
        SUFFIX = '_dark'
        plt.rcParams.update(DARK_THEME)
        plt.rcParams['grid.color'] = '0.35'
        TABLE_COLOURS.update(cell='black', header='#333333', flag='#7a2424')

    folder, settings = experiment_settings(args.results)
    out = args.out or os.path.join(folder, 'sanity_report')
    os.makedirs(out, exist_ok=True)
    if args.pdf_only:
        summary = pd.read_csv(os.path.join(out, 'summary.tsv'), sep='\t')
        with open(os.path.join(out, 'overview.pkl'), 'rb') as f:
            data = pickle.load(f)
        print(f'PDF: {write_pdf(summary, data, out, args.results, settings, folder)}')
        return

    glaciers_file = os.path.join(folder, 'glaciers.txt')
    if not os.path.exists(glaciers_file):
        glaciers_file = os.path.join(HERE, 'glaciers.txt')
    with open(glaciers_file) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    if args.glaciers:
        glaciers = [g for g in glaciers if g[0] in args.glaciers or g[1] in args.glaciers]

    rows, data = [], []
    for rgi_id, name in glaciers:
        rgi_dir = os.path.join(args.results, rgi_id)
        if not os.path.exists(os.path.join(rgi_dir, 'calibration_results.json')):
            print(f'{name}: no results, skipped')
            continue
        print(f'{name} ...', flush=True)
        g = load_glacier(rgi_dir, settings)
        stats = {'rgi_id': rgi_id, 'glacier': name, 'members': g['n_members'],
                 'inversion_year': g['inversion_year'], 'inverted': g['inverted'],
                 'experiment': g['experiment'],
                 **{key: g[key] for key in ['thk_source', 'thk_source_year', 'velocity_source',
                                            'velocity_source_year']},
                 'area_km2': float(g['ice'].sum() * g['dx'] ** 2 * 1e-6),
                 'volume_2000_km3': float(np.median(g['volumes'][:, 0]))}
        stats.update(bed_checks(g))
        stats.update(tau_checks(g, settings['tau_bounds']))
        stats.update(velocity_checks(g))
        stats.update(dhdt_checks(g))
        target = os.path.join(out, name)
        os.makedirs(target, exist_ok=True)
        title = name.replace('_', ' ')
        plot_glacier(g, stats, os.path.join(target, f'glacier{SUFFIX}.png'), title, rgi_id)
        rows.append(stats)
        data.append(overview_data(g))

    summary = pd.DataFrame(rows)
    from reference_profiles import ela_censored_share
    share = ela_censored_share(list(summary.rgi_id))
    summary['ela_ref_capped'] = [share[r] for r in summary.rgi_id]
    summary.to_csv(os.path.join(out, 'summary.tsv'), sep='\t', index=False, float_format='%.4g')
    with open(os.path.join(out, 'overview.pkl'), 'wb') as f:
        pickle.dump(data, f)
    write_markdown(summary, out, args.results)
    print(f'PDF: {write_pdf(summary, data, out, args.results, settings, folder)}')


if __name__ == '__main__':
    main()
