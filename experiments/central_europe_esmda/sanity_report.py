#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Sanity report of a FROST run (inversion + ES-MDA calibration) as an A4
PDF: statistics over all glaciers first, then one page per glacier.

1. Thickness: the thickness of 2000 (the thickness product moved to 2000)
   on a map with GlaThiDa moved to 2000 as dots, and cell by cell (inset).
2. Velocity in time: the inversion fits the observed speeds on the geometry
   of the inversion year (the year of the velocity product). Do the forward
   runs of the posterior ensemble reproduce them in that year (map of the
   difference), and how do the speeds change 2000-2020? The yearly mean
   speed of every member (whole glacier and the lower, middle and upper
   third by the elevation of 2000) and the speed by elevation band in
   every year.
3. dh/dt: map of model - observed 2000-2020, the band profile with the
   observation error, the ensemble spread, the SMB and the flux divergence
   (dh/dt = smb - divflux).
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
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from frost.calibration.observation_provider import ObservationProvider
from frost.preprocess.thickness_validation import read_field, thkobs_at_year
from frost.visualization.plot_style import DARK_THEME, bg, fg
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'validation'))

# --dark: black pages (DARK_THEME), figures *_dark.png, report_dark.pdf
SUFFIX = ''
TABLE_COLOURS = {'cell': 'white', 'header': '#dddddd', 'flag': '#f4b6b6'}

# As the 3x3 figure of experiments/central_europe/evaluation.py: no spines,
# dashed grid behind the data
STYLE = {**{f'axes.spines.{side}': False for side in ['top', 'right', 'bottom', 'left']},
         'axes.grid': True, 'axes.axisbelow': True, 'grid.linestyle': '--',
         'grid.linewidth': 0.5, 'grid.color': '0.75', 'legend.frameon': False}
plt.rcParams.update(STYLE)

SMOOTHING = 200.0   # m, Gaussian sigma of the high-pass checks
A4 = (8.27, 11.69)  # portrait, inches
DPI = 150

# Flags of the summary table (red) and REPORT.md
FLAGS = {
    'thk_bias_glathida_m': (lambda v: abs(v) > 30, 'thickness off GlaThiDa by > 30 m on average'),
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
        g['cost_data'] = float(ds['da_cost_data'][:]) if 'da_cost_data' in ds.variables else np.nan
        g['cost_reg'] = float(ds['da_cost_reg'][:]) if 'da_cost_reg' in ds.variables else np.nan
        g['x'] = np.array(ds['x'][:], dtype=float)
        g['y'] = np.array(ds['y'][:], dtype=float)
    g['dx'] = abs(g['x'][1] - g['x'][0])
    g['extent'] = [0, (g['x'][-1] - g['x'][0]) / 1e3, 0, (g['y'][-1] - g['y'][0]) / 1e3]

    rgi_id = os.path.basename(os.path.normpath(rgi_dir))
    with Dataset(os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc')) as ds:
        dhdt_input = np.nan_to_num(read_field(ds, 'dhdt'))
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
    image = ax.imshow(field, origin='lower', extent=g['extent'], cmap=cmap,
                      norm=norm, interpolation='nearest', **kwargs)
    ax.contour(g['ice'].astype(float), levels=[0.5], colors=fg(), linewidths=0.5,
               origin='lower', extent=g['extent'])
    ax.grid(False)
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
         'Thickness 2000 (dots: GlaThiDa moved to 2000)\ninset: model vs GlaThiDa per cell',
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
        inset.tick_params(labelsize=4.5, length=1.5, pad=1)
        inset.text(0.05, 0.95, f'r {stats["thk_r_glathida"]:.2f}\nbias '
                   f'{stats["thk_bias_glathida_m"]:+.0f} m', transform=inset.transAxes,
                   fontsize=5, va='top')


def speed_map_panel(ax, g):
    """Speed misfit in the inversion year."""
    year = g['years'][g['inv_index']]
    region = g['regions']['glacier']
    difference = masked(g['speed_maps']['inversion'] - g['velsurfobs_mag'], region)
    limit = max(np.nanpercentile(np.abs(difference), 95), 1)
    show(ax, g, difference, f'Speed: forward {year:.0f} - observed 2014-2022\n'
         '(red: too fast; mean of the members with ice)', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m/yr')


def dhdt_map_panel(ax, g):
    provider = g['provider']
    valid = provider.valid
    difference = g['dhdt_model'] - provider.dhdt
    limit = max(np.nanpercentile(np.abs(difference[valid]), 95), 0.5)
    show(ax, g, masked(difference, valid), 'dh/dt 2000-2020: model - Hugonnet\n'
         '(red: model thins too little; ensemble mean)', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m/yr')


def band_panel(ax, g):
    """Band profile of dh/dt with its budget (without ice flow: dh/dt =
    SMB, the flux divergence is zero and left out)."""
    provider = g['provider']
    n = provider.num_bins
    centre = g['band_centre']
    obs = provider.observation[:n]
    data_sigma = np.sqrt(np.diag(provider.data_covariance))
    total_sigma = np.sqrt(np.diag(provider.covariance[:n, :n]))
    ax.axvline(0, color=fg(), lw=0.6)
    ax.fill_betweenx(centre, obs - 2 * total_sigma, obs + 2 * total_sigma, color='gray', alpha=0.3,
                     lw=0, label='obs. +-2 sigma (with model error)')
    ax.errorbar(obs, centre, xerr=2 * data_sigma, fmt='o', color=fg(), ms=2, lw=0.7,
                label='obs. dh/dt +-2 sigma (data)')
    ax.fill_betweenx(centre, np.percentile(g['band_dhdt'], 5, 0), np.percentile(g['band_dhdt'], 95, 0),
                     color='C1', alpha=0.4, lw=0, label='model dh/dt, members 5-95 %')
    ax.plot(np.median(g['band_dhdt'], 0), centre, 'C1-', label='model dh/dt, median')
    ax.plot(g['band_smb'].mean(0), centre, 'C2-', lw=1,
            label='SMB (where the ice lasts, = dh/dt)' if g['no_flow'] else 'SMB')
    if not g['no_flow']:
        ax.plot(-g['band_div'].mean(0), centre, 'C4-', lw=1, label='minus flux divergence')
    dead = (g['band_gone'].min(0) > 0.99) & (g['band_dhdt'].std(0) < 1e-3)
    if dead.any():
        ax.plot(np.median(g['band_dhdt'], 0)[dead], centre[dead], 'rx', ms=7,
                label='melted out in every member')
    ax.set_xlabel('m ice/yr (2000-2020 mean; SMB' + ('' if g['no_flow'] else ' and flux')
                  + ': ensemble mean)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title('Band profile of the elevation change\n'
                 + ('without ice flow: dh/dt = SMB, limited\nby the ice where it melts out' if g['no_flow']
                    else 'and its budget: dh/dt = SMB - flux divergence'), fontsize=9)
    ax.legend(fontsize=5.5, loc='best', frameon=True, framealpha=0.8, facecolor=bg(), edgecolor='none')
    small(ax)


def speed_time_panel(ax, g):
    """Speed in time by thirds of the elevation."""
    year = g['years'][g['inv_index']]
    colors = dict(zip(['lower', 'middle', 'upper'], gradient('viridis', 3)), glacier=fg())
    edges = g['region_edges']
    for r, (key, mask) in enumerate(g['regions'].items()):
        series = g['speeds'][:, :, r]
        label = 'whole glacier' if key == 'glacier' else \
            f'{key} ({edges[r - 1]:.0f}-{edges[r]:.0f} m)'
        ax.fill_between(g['years'], np.nanpercentile(series, 5, 0), np.nanpercentile(series, 95, 0),
                        color=colors[key], alpha=0.2, lw=0)
        ax.plot(g['years'], np.nanmedian(series, 0), color=colors[key], label=label)
        ax.hlines(np.nanmean(g['velsurfobs_mag'][mask]), 2014, 2022, color=colors[key],
                  lw=4, alpha=0.5)
    ax.axvline(year, color=fg(), lw=0.6, ls=':')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f'Year (dotted: inversion year {year:.0f})')
    ax.set_ylabel('Mean surface speed (m/yr)')
    ax.set_title('Speed of the members (line: median, shade: 5-95 %)\n'
                 'thick bars: observed 2014-2022', fontsize=9)
    ax.legend(fontsize=6)
    small(ax)


def band_speed_panel(ax, g):
    """Speed by elevation band, every year."""
    provider = g['provider']
    centre = g['band_centre']
    band_speed = np.nanmedian(g['band_speed'], 0)           # year, band
    cmap = truncated('magma')
    norm = matplotlib.colors.Normalize(g['years'][0], g['years'][-1])
    for t, values in enumerate(band_speed):
        ax.plot(values, centre, color=cmap(norm(g['years'][t])), lw=0.8)
    field = g['velsurfobs_mag']
    values = np.where(np.isfinite(field), field, 0.0)
    count = provider.band_mean(np.isfinite(field).astype(float))
    ax.plot(provider.band_mean(values) / np.maximum(count, 1e-6), centre, '--', color=fg(),
            lw=1.5, label='observed 2014-2022')
    colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), ax=ax, shrink=0.8)
    colorbar.locator = matplotlib.ticker.MaxNLocator(integer=True)
    colorbar.update_ticks()
    colorbar.set_label('Year of the forward runs', fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.outline.set_visible(False)
    ax.set_xlabel('Band mean surface speed (m/yr)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title('Speed by elevation band, every year\n(median of the members)', fontsize=9)
    ax.legend(fontsize=6, loc='upper right')
    small(ax)


def smb_profile_panel(ax, g):
    """Posterior SMB (all members) against the GLAMOS / WGMS mean annual
    mass balance per elevation bin 2000-2019."""
    from evaluate import member_smb
    from reference_profiles import mass_balance_profiles
    rgi_id = os.path.basename(os.path.normpath(g['rgi_dir']))
    profiles = mass_balance_profiles([rgi_id])
    ax.set_title('SMB profile against the reference\n(mean annual mass balance 2000-2019)', fontsize=9)
    if rgi_id not in profiles:
        ax.text(0.5, 0.5, 'no reference profile', transform=ax.transAxes, ha='center')
        return
    reference = profiles[rgi_id]
    elevation = np.linspace(g['band_centre'].min(), g['band_centre'].max(), 60)
    with open(os.path.join(g['rgi_dir'], 'calibration_results.json')) as f:
        smb = member_smb(json.load(f), elevation)
    ax.axvline(0, color=fg(), lw=0.6)
    ax.fill_betweenx(elevation, np.percentile(smb, 5, 0), np.percentile(smb, 95, 0), color='C1',
                     alpha=0.4, lw=0, label='posterior, members 5-95 %')
    ax.plot(smb.mean(0), elevation, 'C1-', label='posterior mean')
    ax.plot(reference.mb, reference.elevation, 'o', color=fg(), ms=3,
            label=f'{reference.source.iloc[0]} (bins with >= 10 years)')
    ax.set_xlabel('Annual mass balance (m w.e./yr; model: m ice x 0.91)')
    ax.set_ylabel('Elevation (m)')
    ax.legend(fontsize=6, loc='best', frameon=True, framealpha=0.8, facecolor=bg(), edgecolor='none')
    small(ax)


def plot_glacier(g, stats, path, name, rgi_id):
    """One A4 page. With ice flow: thickness with GlaThiDa, speed and dh/dt
    misfit maps, band mass budget, speed in time and by elevation band.
    Without ice flow: thickness, dh/dt misfit, band profile and the SMB
    profile against the reference (no speeds)."""
    numbers = (f'Thickness vs GlaThiDa: bias {stats["thk_bias_glathida_m"]:+.0f} m, '
               f'r {stats["thk_r_glathida"]:.2f}, {stats["thk_cells_glathida"]} cells.  ')
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
    if g['no_flow']:
        # maps lower than the profiles, so their colorbars stay below the titles
        fig, axes = plt.subplots(2, 2, figsize=A4, gridspec_kw={'height_ratios': [1, 1.4]})
        page_header(fig, name, rgi_id,
                    f'Without ice flow: thickness 2000 and elevation change 2000-2020 '
                    f'({g["n_members"]} posterior members)',
                    numbers + f'  No ice flow: ITS_LIVE p99 on the ice '
                              f'{stats["velocity_p99_ice"]:.1f} m/yr.')
        thickness_panel(axes[0, 0], g, stats)
        dhdt_map_panel(axes[0, 1], g)
        band_panel(axes[1, 0], g)
        smb_profile_panel(axes[1, 1], g)
    else:
        fig, axes = plt.subplots(3, 2, figsize=A4)
        page_header(fig, name, rgi_id,
                    f'Thickness 2000, speed and elevation change 2000-2020 '
                    f'({g["n_members"]} posterior members)', numbers)
        thickness_panel(axes[0, 0], g, stats)
        speed_map_panel(axes[0, 1], g)
        dhdt_map_panel(axes[1, 0], g)
        band_panel(axes[1, 1], g)
        speed_time_panel(axes[2, 0], g)
        band_speed_panel(axes[2, 1], g)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
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


NO_FLOW_NOTE = '* without ice flow (ITS_LIVE p99 on the ice < 10 m/yr): dh/dt = SMB'

# Glacier identity on every page: 8 categorical hues (CVD-checked palette,
# light and dark steps) x 3 marker shapes / line styles, in the order of
# glaciers.txt, so each of up to 24 glaciers has its own colour-marker pair
PALETTE = {'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
                     '#6250d6', '#e34948'],
           'dark': ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300',
                    '#9085e9', '#e66767']}
MARKERS = ['o', '^', 's']
LINESTYLES = ['-', '--', ':']


def glacier_style(summary):
    """Colour (RGBA rows), marker and line style of every glacier."""
    palette = PALETTE['dark' if SUFFIX else 'light']
    n = len(summary)
    colors = np.array([matplotlib.colors.to_rgba(palette[i % len(palette)]) for i in range(n)])
    group = [(i // len(palette)) % len(MARKERS) for i in range(n)]
    return colors, [MARKERS[k] for k in group], [LINESTYLES[k] for k in group]


def glacier_legend(fig, summary, colors, markers, y=0.01, ncol=4):
    """Legend of all glaciers (colour and marker) along the bottom of a page."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls='none', marker=m, color=c, markersize=5)
               for c, m in zip(colors, markers)]
    fig.legend(handles, labels(summary), loc='lower center', bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=6.5, handletextpad=0.3, columnspacing=1.2, frameon=False)



def overview_thickness_dhdt(summary, data):
    fig, axes = plt.subplots(2, 2, figsize=A4)
    fig.text(0.06, 0.975, 'All glaciers: thickness and elevation change', fontsize=13,
             weight='bold', va='top')
    if no_flow(summary).any():
        fig.text(0.06, 0.95, NO_FLOW_NOTE + ' (open markers)', fontsize=7.5, va='top')
    names = labels(summary)
    colors, markers, styles = glacier_style(summary)

    ax = axes[0, 0]
    obs = np.concatenate([d['thk_obs'] for d in data])
    model = np.concatenate([d['thk_model'] for d in data])
    for d, c in zip(data, colors):
        ax.scatter(d['thk_obs'], d['thk_model'], s=2, color=c, alpha=0.6)
    top = max(obs.max(), model.max())
    ax.plot([0, top], [0, top], '--', color=fg(), lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel('GlaThiDa moved to 2000 (m)')
    ax.set_ylabel('Model 2000 (m)')
    ax.set_title(f'Thickness per cell, all glaciers (dashed 1:1)\nr {np.corrcoef(obs, model)[0, 1]:.2f}, bias '
                 f'{np.mean(model - obs):+.0f} m, RMSE {np.sqrt(np.mean((model - obs) ** 2)):.0f} m, '
                 f'n {len(obs)}', fontsize=9)
    small(ax)

    ax = axes[0, 1]
    bias = summary.thk_bias_glathida_m.to_numpy()
    ax.barh(range(len(summary)), bias, color=colors)
    ax.axvline(0, color=fg(), lw=0.6)
    for x in (-30, 30):
        ax.axvline(x, color='r', ls='--', lw=0.8, label='flag: |bias| > 30 m' if x > 0 else None)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean thickness bias vs GlaThiDa (m)')
    ax.set_title('Mean thickness bias per glacier\n(model - GlaThiDa on the cells with data)', fontsize=9)
    small(ax)

    ax = axes[1, 0]
    ax.errorbar(summary.dhdt_obs, summary.dhdt_model, xerr=2 * summary.dhdt_obs_sigma,
                yerr=2 * summary.dhdt_model_std, fmt='none', ecolor='0.6', lw=0.7,
                label='+-2 sigma (obs.) / +-2 std (members)')
    flat = no_flow(summary)
    for i, row in enumerate(summary.itertuples()):
        ax.scatter(row.dhdt_obs, row.dhdt_model, marker=markers[i], s=26, zorder=3, lw=1,
                   facecolors='none' if flat[i] else colors[i], edgecolors=colors[i])
    low = min(summary.dhdt_obs.min(), summary.dhdt_model.min()) - 0.2
    ax.plot([low, 0], [low, 0], '--', color=fg(), lw=0.8, label='1:1')
    ax.legend(fontsize=7, loc='upper left')
    error = summary.dhdt_model - summary.dhdt_obs
    ax.set_xlabel('Observed (m/yr, +-2 sigma)')
    ax.set_ylabel('Modelled (m/yr, +-2 std of the members)')
    ax.set_title(f'Glacier-mean dh/dt 2000-2020\nr '
                 f'{np.corrcoef(summary.dhdt_obs, summary.dhdt_model)[0, 1]:.2f}, '
                 f'MAE {error.abs().mean():.2f} m/yr', fontsize=9)
    small(ax)

    ax = axes[1, 1]
    grid = np.linspace(0, 1, 21)
    profiles = []
    for d, c, style in zip(data, colors, styles):
        ax.plot(d['band_residual'], d['band_rel_elevation'], color=c, lw=0.8, alpha=0.8, ls=style)
        profiles.append(np.interp(grid, d['band_rel_elevation'], d['band_residual']))
    ax.plot(np.median(profiles, 0), grid, '-', color=fg(), lw=2.5, label='median of the glaciers')
    ax.axvline(0, color=fg(), lw=0.6)
    ax.set_xlim(-3, 3)
    ax.set_xlabel('Band dh/dt, model - observed (m/yr)')
    ax.set_ylabel('Relative elevation (0 tongue, 1 top)')
    ax.set_title('Band dh/dt misfit over the elevation\n(thin lines: glaciers; > 0: thins too little)',
                 fontsize=9)
    ax.legend(fontsize=7)
    small(ax)
    glacier_legend(fig, summary, colors, markers, y=0.025)
    fig.tight_layout(rect=[0.03, 0.12, 0.97, 0.94])
    return fig


def plain_log_axis(axis, ticks):
    """Log axis with plain tick labels (1, 2, 3 instead of 2x10^0)."""
    axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def overview_velocity(summary, data):
    fig, axes = plt.subplots(2, 2, figsize=A4)
    fig.text(0.06, 0.975, 'All glaciers: velocity in time', fontsize=13, weight='bold', va='top')
    colors, markers, styles = glacier_style(summary)
    keep = ~no_flow(summary)
    if not keep.all():
        fig.text(0.06, 0.95, f'{keep.sum()} glaciers with ice flow; the {(~keep).sum()} without '
                 '(ITS_LIVE p99 on the ice < 10 m/yr) are left out', fontsize=7.5, va='top')
    # colours stay with their glacier (as on the other pages)
    summary, data, colors = summary[keep].reset_index(drop=True), \
        [d for d, k in zip(data, keep) if k], colors[keep]
    markers = [m for m, k in zip(markers, keep) if k]
    styles = [st for st, k in zip(styles, keep) if k]
    names = labels(summary)
    year = int(summary.inversion_year.iloc[0])

    ax = axes[0, 0]
    for i, row in enumerate(summary.itertuples()):
        ax.scatter(row.speed_obs, row.speed_forward_obs_year, color=colors[i], marker=markers[i],
                   s=26, zorder=3)
    low, high = 0.2, 1.5 * max(summary.speed_obs.max(), summary.speed_forward_obs_year.max())
    ax.plot([low, high], [low, high], '--', color=fg(), lw=0.8)
    ax.fill_between([low, high], [low / 1.33, high / 1.33], [low * 1.33, high * 1.33],
                    color='gray', alpha=0.2, lw=0, label='+-33 % (flag outside)')
    ax.legend(fontsize=7, loc='upper left')
    for _, row, name in zip(range(len(summary)), summary.itertuples(), names):
        if abs(np.log(row.speed_ratio_obs)) > np.log(1.5):
            ax.annotate(name, (row.speed_obs, row.speed_forward_obs_year), fontsize=6,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Observed 2014-2022 (m/yr)')
    ax.set_ylabel(f'Forward {year} (m/yr)')
    ax.set_title(f'Glacier-mean speed in {year}\n(names: off by > 1.5x)', fontsize=9)
    small(ax)

    ax = axes[0, 1]
    for d, c, style in zip(data, colors, styles):
        ax.plot(d['years'], d['speed_series'] / d['speed_obs_regions'][0], color=c, lw=1, ls=style)
    ax.axhline(1, color=fg(), lw=0.8)
    ax.axvspan(2014, 2022, color='gray', alpha=0.15, lw=0, label='observation period')
    ax.legend(fontsize=7)
    ax.set_yscale('log')
    plain_log_axis(ax.yaxis, [0.5, 0.75, 1, 1.5, 2, 3, 5, 10])
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Year')
    ax.set_ylabel('Forward speed / observed speed')
    ax.set_title('Glacier-mean speed (median of the members)\nrelative to the observed', fontsize=9)
    small(ax)

    ax = axes[1, 0]
    ratio = summary.speed_start_ratio.to_numpy()
    ax.barh(range(len(summary)), ratio, color=colors)
    ax.axvline(1, color=fg(), lw=0.6)
    ax.axvline(2, color='r', ls='--', lw=0.8, label='flag: > 2x (start-up shock)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xscale('log')
    plain_log_axis(ax.xaxis, [1, 1.5, 2, 3, 5, 8])
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f'Forward speed 2000 / forward speed {year}')
    ax.set_title(f'Start-up: forward speed of 2000\nagainst that of {year}', fontsize=9)
    small(ax)

    ax = axes[1, 1]
    thirds = dict(zip(['lower', 'middle', 'upper'], zip(gradient('viridis', 3), ['v', 'o', '^'])))
    for i, d in enumerate(data):
        for r, region in enumerate(d['regions']):
            if region in thirds:
                ratio = d['speed_forward_regions'][r] / d['speed_obs_regions'][r]
                color, marker = thirds[region]
                ax.scatter(ratio, i, color=color, marker=marker, s=14,
                           label=f'{region} third' if i == 0 else None)
    ax.axvline(1, color=fg(), lw=0.6)
    ax.axvspan(1 / 1.33, 1.33, color='gray', alpha=0.2, lw=0, label='+-33 %')
    ax.set_xscale('log')
    plain_log_axis(ax.xaxis, [0.5, 0.75, 1, 1.33, 2, 3])
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f'Forward {year} / observed')
    ax.set_title(f'Speed by thirds of the elevation of 2000\n(forward {year} / observed 2014-2022)',
                 fontsize=9)
    ax.legend(fontsize=7)
    small(ax)
    glacier_legend(fig, summary, colors, markers, y=0.025)
    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.94])
    return fig


TABLE = [('glacier', 'Glacier', None),
         ('thk_bias_glathida_m', 'Thk bias\nvs GlaThiDa', '{:+.0f}'),
         ('thk_r_glathida', 'Thk r', '{:.2f}'),
         ('margin_thk_m', 'Thk at\noutline', '{:.0f}'),
         ('speed_obs', 'Speed\nobserved', '{:.1f}'),
         ('speed_forward_2000', 'Forward\n2000', '{:.1f}'),
         ('speed_forward_obs_year', 'Forward\nobs. year', '{:.1f}'),
         ('speed_ratio_obs', 'Forward/\nobserved', '{:.2f}'),
         ('speed_start_ratio', '2000/\nobs. year', '{:.1f}'),
         ('dhdt_obs', 'dh/dt\nobserved', '{:.2f}'),
         ('dhdt_model', 'dh/dt\nmodel', '{:.2f}'),
         ('dhdt_chi2_data', 'chi2/n\n(data)', '{:.1f}'),
         ('dead_bands', 'Melted\nbands', '{:.0f}'),
         ('ela_ref_capped', 'ELA ref\ncapped', '{:.0%}'),
         ('ice_flow', 'Ice\nflow', '{}')]


def table_page(summary):
    names = labels(summary)
    cells, colours = [], []
    for name, (_, row) in zip(names, summary.iterrows()):
        line, colour = [], []
        for column, _, fmt in TABLE:
            value = row[column]
            if fmt is None:
                line.append(name)
                colour.append(TABLE_COLOURS['cell'])
                continue
            line.append(fmt.format(value) if pd.notna(value) else '-')
            if column == 'ice_flow':
                line[-1] = {'igm': 'IGM', 'none': 'none'}.get(value, '-')
            flagged = column in FLAGS and pd.notna(value) and FLAGS[column][0](value)
            colour.append(TABLE_COLOURS['flag'] if flagged else TABLE_COLOURS['cell'])
        cells.append(line)
        colours.append(colour)
    fig = plt.figure(figsize=A4)
    fig.text(0.06, 0.975, 'All glaciers: summary', fontsize=13, weight='bold', va='top')
    ax = fig.add_axes([0.045, 0.25, 0.91, 0.68])
    ax.axis('off')
    widths = [0.17] + [0.83 / (len(TABLE) - 1)] * (len(TABLE) - 1)
    table = ax.table(cellText=cells, colLabels=[c[1] for c in TABLE], cellColours=colours,
                     colWidths=widths, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.3)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_height(cell.get_height() * 2)
            cell.set_facecolor(TABLE_COLOURS['header'])
        cell.set_edgecolor(fg())
        cell.get_text().set_color(fg())
        if c == 0:
            cell.get_text().set_ha('left')
    notes = ['Thickness in m, speeds in m/yr (glacier means over the ice with observed speeds), '
             'dh/dt in m/yr (2000-2020). Red: flagged. ' + NO_FLOW_NOTE + '; no forward speeds.'] + \
        [f'- {text}' for _, text in FLAGS.values()] + \
        ['Bed and tau_ref, in numbers only (summary.tsv): '
         + bed_tau_note(summary)]
    y = 0.22
    for note in notes:
        for line in textwrap.wrap(note, 115):
            fig.text(0.06, y, line, fontsize=7.5, va='top')
            y -= 0.015
    return fig


def bed_tau_note(summary):
    return (f'bed roughness on the ice {summary.bed_roughness_ratio.min():.1f}-'
            f'{summary.bed_roughness_ratio.max():.1f} x that of the ice-free terrain, '
            f'{summary.bed_pits.min()}-{summary.bed_pits.max()} single-cell pits per glacier; '
            f'tau_ref at its bounds on {summary.tau_at_bounds_pct.max():.1f} % of the ice at most, '
            f'small-scale correlation with the observed speed {summary.tau_noise_corr.min():.2f} to '
            f'{summary.tau_noise_corr.max():.2f} (no sign of fitting observation noise).')


def profile_page(summary, results):
    """SMB profile of the posterior (mean +- 2 std of the members) against
    the GLAMOS / WGMS mean annual mass balance per elevation bin 2000-2019,
    one panel per glacier (evaluate.py, reference_profiles.py)."""
    from evaluate import member_smb
    from reference_profiles import mass_balance_profiles
    profiles = mass_balance_profiles(list(summary.rgi_id))
    fig, axes = plt.subplots(5, 5, figsize=A4)
    fig.text(0.06, 0.975, 'All glaciers: SMB profile against GLAMOS / WGMS', fontsize=13,
             weight='bold', va='top')
    stats = []
    colors, _, _ = glacier_style(summary)
    for ax, row, name, color in zip(axes.flat, summary.itertuples(), labels(summary), colors):
        path = os.path.join(results, row.rgi_id, 'calibration_results.json')
        if row.rgi_id not in profiles or not os.path.exists(path):
            ax.axis('off')
            continue
        reference = profiles[row.rgi_id]
        elevation = reference.elevation.to_numpy(dtype=float)
        with open(path) as f:
            smb = member_smb(json.load(f), elevation)
        mean, std = smb.mean(0), smb.std(0)
        diff = mean - reference.mb.to_numpy()
        stats.append((np.sqrt(np.mean(diff ** 2)), diff.mean(), np.mean(np.abs(diff) <= 2 * std)))
        ax.axvline(0, color=fg(), lw=0.5)
        ax.fill_betweenx(elevation, mean - 2 * std, mean + 2 * std, color=color, alpha=0.3, lw=0)
        ax.plot(mean, elevation, '-', color=color, lw=1.2)
        ax.plot(reference.mb, elevation, 'o', color=fg(), ms=2)
        ax.set_title(f'{name}\n{reference.source.iloc[0]}, RMSE {stats[-1][0]:.2f}', fontsize=6.5)
        ax.tick_params(labelsize=5, length=2, pad=1)
    for ax in axes.flat[len(summary):]:
        ax.axis('off')
    rmse, bias, inside = np.array(stats).T
    lines = [f'Dots: mean annual mass balance per elevation bin 2000-2019 (bins with >= 10 years). '
             f'Line and shade: posterior SMB, mean +- 2 std of the members (m ice x 0.91). '
             f'x: m w.e./yr, y: elevation (m). {len(stats)} glaciers: RMSE {rmse.mean():.2f}, '
             f'bias {bias.mean():+.2f} m w.e./yr, {100 * inside.mean():.0f} % of the bins inside '
             f'+-2 std. Compares the SMB directly, without the ELA and gradient definitions of the '
             f'references (their ELA is capped at the glacier top in years without accumulation '
             f'area).']
    for i, line in enumerate(textwrap.wrap(lines[0], 110)):
        fig.text(0.06, 0.95 - 0.012 * i, line, fontsize=6.5, va='top', color='0.6' if SUFFIX else '0.3')
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.88])
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
    fig, axes = plt.subplots(2, 2, figsize=A4)
    fig.text(0.06, 0.975, 'All glaciers: calibration against GLAMOS / WGMS', fontsize=13,
             weight='bold', va='top')
    note = ('Posterior mean +- 1 std; gradients in m w.e. (model m ice x 0.91). Open markers: '
            'reference ELA at the glacier top in > 25 % of the years (a lower bound), not in r, '
            'bias, MAE. * without ice flow.')
    for i, line in enumerate(textwrap.wrap(note, 112)):
        fig.text(0.06, 0.95 - 0.012 * i, line, fontsize=7, va='top', color='0.6' if SUFFIX else '0.3')
    panels = [('ela', 'ELA (m)', 1.0), ('abl', 'Ablation gradient (m w.e./yr per km)', 0.91),
              ('acc', 'Accumulation gradient (m w.e./yr per km)', 0.91)]
    for ax, (p, label, factor) in zip(axes.flat, panels):
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
            ax.scatter(x, y, marker=markers[i], s=30, lw=1, zorder=3, edgecolors=colors[i],
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
        ax.text(0.97, 0.03, f'r {np.corrcoef(xs, ys)[0, 1]:.2f}\n'
                f'bias {np.mean(ys - xs):+{".0f" if p == "ela" else ".2f"}}\n'
                f'MAE {np.mean(np.abs(ys - xs)):{".0f" if p == "ela" else ".2f"}}\nn {len(xs)}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        ax.set_xlabel(f'GLAMOS / WGMS {label}')
        ax.set_ylabel(f'FROST {label}')
        ax.set_title(label.split(' (')[0], fontsize=9)
        small(ax)
    ax = axes[1, 1]
    xs, ys = [], []
    for i, row in enumerate(summary.itertuples()):
        if row.rgi_id not in aggregated.index:
            continue
        x, y = aggregated.loc[row.rgi_id, 'dhdt_mean'], aggregated.loc[row.rgi_id, 'annual_mass_balance']
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, marker=markers[i], color=colors[i], s=30, zorder=3)
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
    ax.text(0.97, 0.03, f'r {np.corrcoef(xs, ys)[0, 1]:.2f}\nMAE {np.mean(np.abs(ys - xs)):.2f}\n'
            f'n {len(xs)}', transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
    ax.set_xlabel('Hugonnet dh/dt 2000-2020 (m/yr)')
    ax.set_ylabel('GLAMOS / WGMS annual balance (m w.e./yr)')
    ax.set_title('The references themselves: geodetic\nagainst glaciological (no FROST result)',
                 fontsize=9)
    small(ax)
    glacier_legend(fig, summary, colors, markers, y=0.025)
    fig.tight_layout(rect=[0.03, 0.12, 0.97, 0.94])
    return fig


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
    flat = no_flow(summary)
    flowing = summary[~flat]
    runs = (f'{len(flowing)} with ice flow (IGM, {int(flowing.members.max())} members, '
            f'inversion year {flowing.inversion_year.iloc[0]:g})' if len(flowing) else '')
    if flat.any():
        runs += (', ' if runs else '') + (f'{flat.sum()} without ice flow ('
                                          f'{int(summary[flat].members.max())} members)')
    header = [f'Results: {results}. {len(summary)} glaciers: {runs}; band height '
              f'{settings["elevation_step"]} m, model error {settings["model_error"]} m/yr. '
              f'Created {date.today().isoformat()} with experiments/central_europe_esmda/'
              'sanity_report.py.']
    # (topic, figure or a function making it); glacier pages are the PNGs
    pages = [('Findings', fig) for fig in text_pages('FROST sanity report',
                                                     header + findings_paragraphs(out))]
    pages.append(('Thickness and elevation change', overview_thickness_dhdt(summary, data)))
    if not flat.all():
        pages.append(('Velocity', overview_velocity(summary, data)))
    reference = reference_page(summary, folder)
    if reference is not None:
        pages.append(('GLAMOS / WGMS', reference))
    pages.append(('SMB profiles', profile_page(summary, results)))
    pages.append(('Summary table', table_page(summary)))
    glaciers = [(row.glacier.replace('_', ' '), row.glacier) for row in summary.itertuples()]
    total = len(pages) + len(glaciers)
    experiment = os.path.basename(os.path.normpath(results))
    muted = '0.6' if SUFFIX else '0.35'

    def finish(pdf, fig, topic, number, **kwargs):
        fig.text(0.035, 0.993, f'FROST sanity report  ·  {experiment}', fontsize=6.5,
                 va='top', color=muted)
        fig.text(0.965, 0.993, topic, fontsize=6.5, va='top', ha='right', color=muted)
        fig.text(0.5, 0.008, f'{number} / {total}', fontsize=7, ha='center', va='bottom',
                 color=muted)
        pdf.savefig(fig, **kwargs)
        plt.close(fig)

    with PdfPages(path) as pdf:
        for number, (topic, fig) in enumerate(pages, start=1):
            finish(pdf, fig, topic, number)
        for number, (name, folder_name) in enumerate(glaciers, start=len(pages) + 1):
            fig = plt.figure(figsize=A4)
            # the glacier page, slightly reduced: side margins, room for header and footer
            ax = fig.add_axes([0.035, 0.025, 0.93, 0.95])
            ax.imshow(plt.imread(os.path.join(out, folder_name, f'glacier{SUFFIX}.png')))
            ax.axis('off')
            finish(pdf, fig, name, number, dpi=DPI)
    return path


def write_markdown(summary, out, results):
    lines = [f'# Sanity report: {results}', '',
             'The full report is report.pdf (A4). Flags:', '']
    lines += [f'- **{column}**: {text}' for column, (_, text) in FLAGS.items()]
    lines += ['', '| Glacier | Thk bias | Speed obs / fwd 2000 / fwd obs. year / fwd 2020 | '
              'dh/dt obs / model | chi2 data | Melted bands | Flags | Figures |',
              '|---|---|---|---|---|---|---|---|']
    for _, row in summary.iterrows():
        flags = [column for column, (test, _) in FLAGS.items()
                 if pd.notna(row[column]) and test(row[column])]
        figures = f'[glacier]({row.glacier}/glacier.png)'
        lines.append(
            f'| {row.glacier} | {row.thk_bias_glathida_m:+.0f} | {row.speed_obs:.1f} / '
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
                 'inversion_year': g['inversion_year']}
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
