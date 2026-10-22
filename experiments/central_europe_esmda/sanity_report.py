#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Sanity report of a FROST run (inversion + ES-MDA calibration): per glacier
four checks with maps and time series, and a summary over all glaciers.

1. Bed (bed.png): the bed is the surface of 2000 minus the thickness of
   2000 (the thickness product moved to 2000). Is it smooth, or does it have
   artefacts? High-pass residual of the bed (bed minus a 200 m Gaussian
   smoothing) on the ice against the same on the ice-free terrain (where the
   bed is the DEM), single-cell pits, the thickness step at the ice margin,
   and the thickness against GlaThiDa moved to 2000.
2. Velocity in time (velocity.png): the inversion fits the observed speeds
   on the geometry of the inversion year. Do the forward runs of the
   posterior ensemble keep these speeds? Yearly mean surface speed of every
   member (whole glacier and the lower, middle and upper third by the
   elevation of 2000) against the inverted and the observed speed, maps of
   the forward speed in the inversion year against the inverted one, and a
   profile over elevation.
3. tau_ref (tau_ref.png): realistic or overfitted? Share of the ice at the
   bounds of the inversion, roughness of log10 tau_ref, the correlation of
   its small-scale pattern with that of the observed speed (tau_ref copying
   the noise of the observations), the sliding share velbase / velsurf and
   the velocity misfit of the inversion.
4. dh/dt (dhdt.png): maps of the observed and modelled dh/dt 2000-2020, the
   band profile with the observation error and the ensemble spread, the
   SMB and flux divergence per band (dh/dt = smb - divflux) and the volume
   of every member in time.

Run from the repository root:
    python experiments/central_europe_esmda/sanity_report.py \\
        --results data/results/central_europe_esmda_velyear
Writes <out>/<glacier>/{bed,velocity,tau_ref,dhdt}.png, <out>/summary.tsv,
<out>/overview.png, <out>/REPORT.md and <out>/report.pdf (everything in one
file; the text of <out>/findings.md, if present, on its first page, for the
interpretation of the run; --pdf_only rebuilds only the PDF); <out> defaults to
experiments/<results name>/sanity_report. The experiment folder (same name
as the results) gives the band height, model error and tau_ref bounds.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from netCDF4 import Dataset
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LogNorm, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from frost.calibration.observation_provider import ObservationProvider
from frost.preprocess.thickness_validation import read_field, thkobs_at_year

SMOOTHING = 200.0   # m, Gaussian sigma of the high-pass checks
DPI = 110

# Flags of REPORT.md
FLAGS = {
    'bed_roughness_ratio': (lambda v: v > 1.5, 'bed rougher than the ice-free terrain'),
    'bed_pits': (lambda v: v > 0, 'single-cell pits in the bed'),
    'tau_at_bounds_pct': (lambda v: v > 10, 'tau_ref at its bounds on > 10 % of the ice'),
    'tau_noise_corr': (lambda v: v < -0.5, 'tau_ref copies the small-scale pattern of the observed speed'),
    'speed_ratio_inv_year': (lambda v: abs(v - 1) > 0.25, 'forward speed in the inversion year off the inverted speed by > 25 %'),
    'dhdt_chi2_data': (lambda v: v > 4, 'modelled band dh/dt outside the data error (chi2/n > 4)'),
    'dead_bands': (lambda v: v > 0, 'bands that melt out in every member'),
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
    g = {}
    with Dataset(os.path.join(rgi_dir, 'Preprocess', 'outputs', 'output.nc')) as ds:
        for name in ['usurf', 'thk', 'topg', 'tau_ref', 'velbase_mag',
                     'velsurf_mag', 'velsurfobs_mag']:
            g[name] = read_field(ds, name)
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
    g['regions'] = {'glacier': region}
    for name, lo, hi in zip(['lower', 'middle', 'upper'], edges[:-1], edges[1:]):
        g['regions'][name] = region & (g['usurf'] >= lo) & (g['usurf'] <= hi)

    members = sorted(glob.glob(os.path.join(rgi_dir, 'Ensemble', 'Member_*',
                                            'outputs', 'output.nc')))
    speeds, volumes, band_dhdt, band_smb, band_div, band_gone = [], [], [], [], [], []
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
    g['volumes'] = np.array(volumes)          # member, year
    g['speed_maps'] = {key: np.where(speed_count[key] > 0, value / np.maximum(speed_count[key], 1),
                                     np.nan) for key, value in speed_sum.items()}
    g['dhdt_model'] = dhdt_sum / n
    g['band_dhdt'] = np.array(band_dhdt)      # member, band
    g['band_smb'] = np.array(band_smb)
    g['band_div'] = np.array(band_div)
    g['band_gone'] = np.array(band_gone)
    g['band_centre'] = provider.band_mean(g['usurf'])
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
    bed, ice, dx = g['topg'], g['ice'], g['dx']
    residual = high_pass(bed, dx)
    inner_ice = binary_erosion(ice, iterations=2)
    free = ~binary_dilation(ice, iterations=3)
    # single-cell pits: lower than all 8 neighbours by more than 5 m
    padded = np.pad(bed, 1, mode='edge')
    neighbours = np.min([padded[1 + i:padded.shape[0] - 1 + i, 1 + j:padded.shape[1] - 1 + j]
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)], axis=0)
    pits = inner_ice & (bed < neighbours - 5)
    margin = ice & ~binary_erosion(ice)
    observed = ice & np.isfinite(g['thkobs']) & (g['thkobs'] > 0)
    rough_ice = np.sqrt(np.nanmean(residual[inner_ice] ** 2))
    rough_free = np.sqrt(np.nanmean(residual[free] ** 2)) if free.any() else np.nan
    g['bed_residual'], g['bed_pits'] = residual, pits
    return {
        'bed_roughness_ice_m': rough_ice,
        'bed_roughness_free_m': rough_free,
        'bed_roughness_ratio': rough_ice / rough_free,
        'bed_pits': int(pits.sum()),
        'margin_thk_m': float(np.mean(g['thk'][margin])),
        'thk_bias_glathida_m': float(np.mean(g['thk'][observed] - g['thkobs'][observed]))
        if observed.any() else np.nan,
        'thk_cells_glathida': int(observed.sum()),
    }


def tau_checks(g, bounds):
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
    g['tau_at_bounds'], g['tau_ice'] = at_bounds, ice
    return {
        'tau_median': float(np.median(tau[ice])),
        'tau_at_bounds_pct': 100 * at_bounds.sum() / ice.sum(),
        'tau_roughness_dec': float(np.sqrt(np.nanmean(log_tau_hp[ice] ** 2))),
        'tau_noise_corr': corr,
        'sliding_share_median': float(np.nanmedian(sliding[moving])),
        'vel_misfit_rms': float(np.sqrt(np.nanmean(misfit[moving] ** 2))),
        'vel_obs_mean': float(np.nanmean(obs[moving])),
        'cost_reg_over_data': g['cost_reg'] / g['cost_data'] if g['cost_data'] else np.nan,
    }


def velocity_checks(g):
    region = g['regions']['glacier']
    inverted = np.nanmean(g['velsurf_mag'][region])
    series = np.nanmedian(g['speeds'][:, :, 0], axis=0)
    return {
        'speed_obs': float(np.nanmean(g['velsurfobs_mag'][region])),
        'speed_inverted': float(inverted),
        'speed_forward_2000': float(series[0]),
        'speed_forward_inv_year': float(series[g['inv_index']]),
        'speed_forward_2020': float(series[-1]),
        'speed_ratio_inv_year': float(series[g['inv_index']] / inverted),
        'speed_change_2000_2020_pct': float(100 * (series[-1] / series[0] - 1)),
    }


def dhdt_checks(g):
    provider = g['provider']
    n = provider.num_bins
    obs = provider.observation[:n]
    model = g['band_dhdt'].mean(0)
    residual = model - obs
    data_cov = provider.data_covariance
    total_cov = provider.covariance[:n, :n]
    dead = (g['band_gone'].min(0) > 0.99) & (g['band_dhdt'].std(0) < 1e-3)
    weights = np.bincount(provider.pixel_band, minlength=n)
    weights = weights / weights.sum()
    return {
        'dhdt_obs': float(weights @ obs),
        'dhdt_model': float(weights @ model),
        'dhdt_chi2_data': float(residual @ np.linalg.solve(data_cov, residual) / n),
        'dhdt_chi2_total': float(residual @ np.linalg.solve(total_cov, residual) / n),
        'bands_outside_2sigma': int(np.sum(np.abs(residual) > 2 * np.sqrt(np.diag(total_cov)))),
        'bands': n,
        'dead_bands': int(dead.sum()),
        'volume_change_pct': float(100 * (np.median(g['volumes'][:, -1])
                                          / np.median(g['volumes'][:, 0]) - 1)),
    }


# --------------------------------------------------------------- plots

def show(ax, g, field, title, cmap='viridis', norm=None, label='', **kwargs):
    image = ax.imshow(field, origin='lower', extent=g['extent'], cmap=cmap,
                      norm=norm, interpolation='nearest', **kwargs)
    ax.contour(g['ice'].astype(float), levels=[0.5], colors='k', linewidths=0.6,
               origin='lower', extent=g['extent'])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('km')
    ax.set_ylabel('km')
    plt.colorbar(image, ax=ax, shrink=0.8, label=label)
    return image


def masked(field, mask):
    return np.where(mask, field, np.nan)


def plot_bed(g, stats, path, name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    ax = axes[0, 0]
    shade = LightSource(azdeg=315, altdeg=40).hillshade(g['topg'], vert_exag=1, dx=g['dx'], dy=g['dx'])
    ax.imshow(shade, origin='lower', extent=g['extent'], cmap='gray', interpolation='nearest')
    show(ax, g, masked(g['topg'], g['ice']), 'Bed (surface 2000 - thickness 2000)',
         cmap='terrain', label='m', alpha=0.6)
    ax = axes[0, 1]
    show(ax, g, masked(g['thk'], g['ice']), 'Thickness 2000, GlaThiDa cells (dots)',
         cmap='Blues', label='m')
    iy, ix = np.nonzero(np.isfinite(g['thkobs']) & (g['thkobs'] > 0))
    if len(iy):
        vmax = np.nanmax(g['thk'][g['ice']])
        ax.scatter(g['extent'][1] * ix / (len(g['x']) - 1), g['extent'][3] * iy / (len(g['y']) - 1),
                   c=g['thkobs'][iy, ix], cmap='Blues', vmin=0, vmax=vmax, s=6,
                   edgecolors='k', linewidths=0.2)
    ax = axes[1, 0]
    limit = np.nanpercentile(np.abs(g['bed_residual']), 99)
    show(ax, g, g['bed_residual'],
         f'Bed high-pass ({SMOOTHING:.0f} m): RMS ice {stats["bed_roughness_ice_m"]:.1f} m, '
         f'ice-free {stats["bed_roughness_free_m"]:.1f} m', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m')
    iy, ix = np.nonzero(g['bed_pits'])
    ax.plot(g['extent'][1] * ix / (len(g['x']) - 1), g['extent'][3] * iy / (len(g['y']) - 1),
            'kx', ms=5, label=f'pits ({len(iy)})')
    if len(iy):
        ax.legend(loc='lower right', fontsize=8)
    ax = axes[1, 1]
    observed = g['ice'] & np.isfinite(g['thkobs']) & (g['thkobs'] > 0)
    if observed.any():
        ax.scatter(g['thkobs'][observed], g['thk'][observed], s=6, alpha=0.6)
        top = max(g['thkobs'][observed].max(), g['thk'][observed].max())
        ax.plot([0, top], [0, top], 'k--', lw=0.8)
        ax.set_title(f'Thickness 2000 vs GlaThiDa moved to 2000 '
                     f'(bias {stats["thk_bias_glathida_m"]:+.0f} m, n {observed.sum()})', fontsize=10)
    else:
        ax.set_title('No GlaThiDa cells on the ice', fontsize=10)
    ax.set_xlabel('GlaThiDa (m)')
    ax.set_ylabel('Model (m)')
    fig.suptitle(f'{name}: bed', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_velocity(g, stats, path, name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    region = g['regions']['glacier']
    vmax = np.nanpercentile(g['velsurfobs_mag'][region], 99)
    norm = LogNorm(max(vmax / 100, 0.1), vmax)
    year = g['years'][g['inv_index']]
    show(axes[0, 0], g, masked(g['velsurfobs_mag'], g['ice']), 'Observed surface speed',
         cmap='magma', norm=norm, label='m/yr')
    show(axes[0, 1], g, masked(g['velsurf_mag'], g['ice']),
         f'Inverted ({g["inversion_year"]:g} geometry)', cmap='magma', norm=norm, label='m/yr')
    show(axes[0, 2], g, masked(g['speed_maps']['inversion'], g['ice']),
         f'Forward {year:.0f}, mean of the members with ice', cmap='magma', norm=norm, label='m/yr')
    ratio = g['speed_maps']['inversion'] / np.maximum(g['velsurf_mag'], 1e-3)
    show(axes[1, 0], g, masked(ratio, g['ice'] & (g['velsurf_mag'] > 0.5)),
         f'Forward {year:.0f} / inverted', cmap='RdBu_r', norm=LogNorm(0.33, 3), label='ratio')

    ax = axes[1, 1]
    colors = {'glacier': 'k', 'lower': 'tab:red', 'middle': 'tab:orange', 'upper': 'tab:blue'}
    for r, (key, mask) in enumerate(g['regions'].items()):
        series = g['speeds'][:, :, r]
        ax.fill_between(g['years'], np.nanpercentile(series, 5, 0), np.nanpercentile(series, 95, 0),
                        color=colors[key], alpha=0.2, lw=0)
        ax.plot(g['years'], np.nanmedian(series, 0), color=colors[key], label=key)
        ax.plot(g['inversion_year'], np.nanmean(g['velsurf_mag'][mask]), '*', ms=12,
                color=colors[key], mec='k')
        ax.hlines(np.nanmean(g['velsurfobs_mag'][mask]), 2014, 2022, color=colors[key],
                  lw=4, alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean surface speed (m/yr)')
    ax.set_title('Forward members (line: median, band: 5-95 %),\n'
                 'star: inverted, bar: observed 2014-2022', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    provider = g['provider']
    centre = g['band_centre']
    for field, label, style in [(g['velsurfobs_mag'], 'observed', 'k-'),
                                (g['velsurf_mag'], f'inverted ({g["inversion_year"]:g})', 'k--'),
                                (g['speed_maps']['start'], 'forward 2000', 'C0-'),
                                (g['speed_maps']['inversion'], f'forward {year:.0f}', 'C1-'),
                                (g['speed_maps']['end'], 'forward 2020', 'C3-')]:
        values = np.where(np.isfinite(field), field, 0.0)
        count = provider.band_mean(np.isfinite(field).astype(float))
        ax.plot(provider.band_mean(values) / np.maximum(count, 1e-6), centre, style, label=label)
    ax.set_xlabel('Band mean surface speed (m/yr)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title(f'Speed by elevation band (forward/inverted {stats["speed_ratio_inv_year"]:.2f})',
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f'{name}: velocity in time', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_tau(g, stats, bounds, path, name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    ice = g['tau_ice']
    lo, hi = bounds
    show(axes[0, 0], g, masked(g['tau_ref'], ice),
         f'tau_ref ({stats["tau_at_bounds_pct"]:.0f} % at the bounds, hatched)',
         cmap='viridis', norm=LogNorm(lo, hi), label='tau_ref')
    axes[0, 0].contourf(g['tau_at_bounds'].astype(float), levels=[0.5, 1.5], colors='none',
                        hatches=['////'], origin='lower', extent=g['extent'])
    sliding = g['velbase_mag'] / np.maximum(g['velsurf_mag'], 1e-3)
    show(axes[0, 1], g, masked(sliding, ice), f'Sliding share velbase / velsurf '
         f'(median {stats["sliding_share_median"]:.2f})', cmap='cividis', vmin=0, vmax=1,
         label='-')
    misfit = g['velsurf_mag'] - g['velsurfobs_mag']
    limit = max(np.nanpercentile(np.abs(misfit[ice]), 95), 0.5)
    show(axes[0, 2], g, masked(misfit, ice), f'Inverted - observed speed '
         f'(RMS {stats["vel_misfit_rms"]:.1f} m/yr)', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -limit, limit), label='m/yr')

    ax = axes[1, 0]
    ax.hist(np.log10(g['tau_ref'][ice]), bins=40, color='C0')
    for bound in bounds:
        ax.axvline(np.log10(bound), color='k', ls='--')
    ax.set_xlabel('log10 tau_ref')
    ax.set_ylabel('Cells')
    ax.set_title(f'Distribution (median {stats["tau_median"]:.2f}, roughness '
                 f'{stats["tau_roughness_dec"]:.2f} decades)', fontsize=10)

    ax = axes[1, 1]
    slope = np.hypot(*np.gradient(g['usurf_inversion'], g['dx']))
    scatter = ax.scatter(g['thk_inversion'][ice], g['tau_ref'][ice], c=np.degrees(np.arctan(slope[ice])),
                         s=4, cmap='plasma', vmin=0, vmax=40)
    ax.set_yscale('log')
    ax.set_xlabel(f'Thickness {g["inversion_year"]:g} (m)')
    ax.set_ylabel('tau_ref')
    ax.set_title('tau_ref vs thickness (colour: surface slope)', fontsize=10)
    plt.colorbar(scatter, ax=ax, label='slope (deg)')

    ax = axes[1, 2]
    moving = ice & np.isfinite(g['velsurfobs_mag'])
    ax.scatter(g['velsurfobs_mag'][moving], g['velsurf_mag'][moving], s=4, alpha=0.5)
    top = np.nanmax(g['velsurfobs_mag'][moving])
    ax.plot([0, top], [0, top], 'k--', lw=0.8)
    ax.set_xlabel('Observed (m/yr)')
    ax.set_ylabel('Inverted (m/yr)')
    ax.set_title(f'Inversion fit; small-scale corr. tau_ref vs observed speed '
                 f'{stats["tau_noise_corr"]:.2f}', fontsize=10)
    fig.suptitle(f'{name}: tau_ref', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_dhdt(g, stats, path, name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    provider = g['provider']
    valid = provider.valid
    limit = np.nanpercentile(np.abs(provider.dhdt[valid]), 98)
    norm = TwoSlopeNorm(0, -limit, max(limit / 4, 0.5))
    show(axes[0, 0], g, masked(provider.dhdt, valid), 'Observed dh/dt 2000-2020 (Hugonnet)',
         cmap='RdBu', norm=norm, label='m/yr')
    show(axes[0, 1], g, masked(g['dhdt_model'], g['ice']), 'Modelled dh/dt, ensemble mean',
         cmap='RdBu', norm=norm, label='m/yr')
    difference = g['dhdt_model'] - provider.dhdt
    dlimit = max(np.nanpercentile(np.abs(difference[valid]), 95), 0.5)
    show(axes[0, 2], g, masked(difference, valid), 'Modelled - observed', cmap='RdBu_r',
         norm=TwoSlopeNorm(0, -dlimit, dlimit), label='m/yr')

    ax = axes[1, 0]
    n = provider.num_bins
    centre = g['band_centre']
    obs = provider.observation[:n]
    data_sigma = np.sqrt(np.diag(provider.data_covariance))
    total_sigma = np.sqrt(np.diag(provider.covariance[:n, :n]))
    ax.fill_betweenx(centre, obs - 2 * total_sigma, obs + 2 * total_sigma, color='gray', alpha=0.25,
                     lw=0, label='obs +-2 sigma (with model error)')
    ax.errorbar(obs, centre, xerr=2 * data_sigma, fmt='ko', ms=3, lw=0.8,
                label='obs +-2 sigma (data)')
    ax.fill_betweenx(centre, np.percentile(g['band_dhdt'], 5, 0), np.percentile(g['band_dhdt'], 95, 0),
                     color='C1', alpha=0.4, lw=0, label='members 5-95 %')
    ax.plot(np.median(g['band_dhdt'], 0), centre, 'C1-', label='members median')
    dead = (g['band_gone'].min(0) > 0.99) & (g['band_dhdt'].std(0) < 1e-3)
    if dead.any():
        ax.plot(np.median(g['band_dhdt'], 0)[dead], centre[dead], 'rx', ms=8, label='melted out')
    ax.set_xlabel('Band dh/dt (m/yr)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title(f'Band profile (chi2/n data {stats["dhdt_chi2_data"]:.1f}, '
                 f'with model error {stats["dhdt_chi2_total"]:.1f})', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    smb, div = g['band_smb'].mean(0), g['band_div'].mean(0)
    ax.plot(smb, centre, 'C2-', label='SMB')
    ax.plot(-div, centre, 'C4-', label='- flux divergence')
    ax.plot(smb - div, centre, 'C1--', label='SMB - divflux')
    ax.plot(obs, centre, 'ko', ms=3, label='observed dh/dt')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('m ice/yr (ensemble mean, 2001-2020)')
    ax.set_ylabel('Elevation 2000 (m)')
    ax.set_title('Mass budget by band', fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    volumes = g['volumes']
    for member in volumes:
        ax.plot(g['years'], member, color='C0', alpha=0.3, lw=0.7)
    area = valid.sum() * g['dx'] ** 2
    observed = volumes[:, 0].mean() + stats['dhdt_obs'] * area * 1e-9 * (g['years'] - g['years'][0])
    ax.plot(g['years'], observed, 'k--', label='observed mean dh/dt x observed area')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Year')
    ax.set_ylabel('Ice volume (km3)')
    ax.set_title(f'Volume of the members ({stats["volume_change_pct"]:+.0f} % 2000-2020)',
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f'{name}: dh/dt', fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_overview(summary, path):
    columns = [('bed_roughness_ratio', 'Bed roughness ice / ice-free', 1.5),
               ('tau_at_bounds_pct', 'tau_ref at bounds (%)', 10),
               ('tau_noise_corr', 'Small-scale corr. tau_ref vs obs speed', -0.5),
               ('speed_ratio_inv_year', 'Forward / inverted speed, inversion year', None),
               ('speed_change_2000_2020_pct', 'Forward speed change 2000-2020 (%)', None),
               ('dhdt_chi2_data', 'dh/dt chi2/n (data error)', 4)]
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.4 * len(columns)), sharex=True)
    names = summary.glacier.str.replace('_', ' ').str[:18]
    for ax, (column, title, threshold) in zip(axes, columns):
        ax.bar(range(len(summary)), summary[column], color='C0')
        if threshold is not None:
            ax.axhline(threshold, color='r', ls='--', lw=0.8)
        if column == 'speed_ratio_inv_year':
            ax.axhline(1, color='k', lw=0.8)
            ax.axhspan(0.75, 1.25, color='gray', alpha=0.2)
        ax.set_title(title, fontsize=10)
    axes[-1].set_xticks(range(len(summary)))
    axes[-1].set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def write_report(summary, out, results, settings):
    lines = [f'# Sanity report: {results}', '',
             f'Band height {settings["elevation_step"]} m, model error '
             f'{settings["model_error"]} m/yr, tau_ref bounds {settings["tau_bounds"]}. '
             'Glaciers sorted as in glaciers.txt. Flags:', '']
    for column, (_, text) in FLAGS.items():
        lines.append(f'- **{column}**: {text}')
    lines += ['', '![overview](overview.png)', '',
              '| Glacier | Bed rough. ratio | Pits | tau at bounds % | tau noise corr | '
              'Speed obs / inv / fwd(inv yr) / fwd 2020 | dh/dt obs / model | chi2 data | '
              'Dead bands | Flags | Figures |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for _, row in summary.iterrows():
        flags = [column for column, (test, _) in FLAGS.items()
                 if np.isfinite(row[column]) and test(row[column])]
        folder = row.glacier
        figures = ' · '.join(f'[{name}]({folder}/{name}.png)'
                             for name in ['bed', 'velocity', 'tau_ref', 'dhdt'])
        lines.append(
            f'| {row.glacier} | {row.bed_roughness_ratio:.2f} | {row.bed_pits} | '
            f'{row.tau_at_bounds_pct:.0f} | {row.tau_noise_corr:.2f} | '
            f'{row.speed_obs:.1f} / {row.speed_inverted:.1f} / {row.speed_forward_inv_year:.1f} / '
            f'{row.speed_forward_2020:.1f} | {row.dhdt_obs:.2f} / {row.dhdt_model:.2f} | '
            f'{row.dhdt_chi2_data:.1f} | {row.dead_bands} | '
            f'{", ".join(flags) if flags else "-"} | {figures} |')
    with open(os.path.join(out, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


A4 = (11.69, 8.27)   # landscape, inches


def text_page(pdf, title, paragraphs, size=10):
    """A page of wrapped text; paragraphs starting with '- ' are bullets."""
    import textwrap
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.94, title, fontsize=16, weight='bold', va='top')
    y = 0.88
    for paragraph in paragraphs:
        bullet = paragraph.startswith('- ')
        body = paragraph[2:] if bullet else paragraph
        lines = textwrap.wrap(body, 130 if size <= 10 else 110) or ['']
        for i, line in enumerate(lines):
            prefix = ('• ' if i == 0 else '  ') if bullet else ''
            fig.text(0.05 + (0.015 if bullet else 0), y, prefix + line, fontsize=size,
                     va='top', family='monospace' if paragraph.startswith('    ') else None)
            y -= 0.022 * size / 10
        y -= 0.012
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=A4)
            fig.text(0.05, 0.94, title + ' (continued)', fontsize=16, weight='bold', va='top')
            y = 0.88
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, path, title):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.97, title, fontsize=12, ha='center', va='top')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.91])
    ax.imshow(plt.imread(path))
    ax.axis('off')
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def table_page(pdf, summary):
    columns = [('glacier', 'Glacier', lambda v: v.replace('_', ' ')[:22]),
               ('bed_roughness_ratio', 'Bed rough.\nratio', '{:.2f}'),
               ('bed_pits', 'Pits', '{:.0f}'),
               ('margin_thk_m', 'Margin\nthk (m)', '{:.0f}'),
               ('thk_bias_glathida_m', 'Thk bias\nGlaThiDa', '{:+.0f}'),
               ('tau_at_bounds_pct', 'tau at\nbounds %', '{:.1f}'),
               ('tau_noise_corr', 'tau noise\ncorr.', '{:+.2f}'),
               ('speed_obs', 'Speed\nobs', '{:.1f}'),
               ('speed_inverted', 'Speed\ninv.', '{:.1f}'),
               ('speed_forward_2000', 'Fwd\n2000', '{:.1f}'),
               ('speed_forward_inv_year', 'Fwd\ninv. yr', '{:.1f}'),
               ('speed_ratio_inv_year', 'Fwd/inv', '{:.2f}'),
               ('dhdt_obs', 'dh/dt\nobs', '{:.2f}'),
               ('dhdt_model', 'dh/dt\nmodel', '{:.2f}'),
               ('dhdt_chi2_data', 'chi2/n\ndata', '{:.1f}'),
               ('dead_bands', 'Dead\nbands', '{:.0f}')]
    cells, colours = [], []
    for _, row in summary.iterrows():
        line, colour = [], []
        for column, _, fmt in columns:
            value = row[column]
            text = fmt(value) if callable(fmt) else (fmt.format(value) if pd.notna(value) else '-')
            flagged = column in FLAGS and pd.notna(value) and FLAGS[column][0](value)
            line.append(text)
            colour.append('#f4b6b6' if flagged else 'white')
        cells.append(line)
        colours.append(colour)
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.96, 'Summary (red: flagged)', fontsize=14, weight='bold', va='top')
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.88])
    ax.axis('off')
    widths = [0.15] + [0.85 / (len(columns) - 1)] * (len(columns) - 1)
    table = ax.table(cellText=cells, colLabels=[c[1] for c in columns], cellColours=colours,
                     colWidths=widths, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_height(cell.get_height() * 1.8)
            cell.set_facecolor('#dddddd')
        if c == 0:
            cell.get_text().set_ha('left')
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(summary, out, results, settings):
    """report.pdf: findings (findings.md in the report folder, if any), the
    checks, the summary table, the overview and the four figures of every
    glacier."""
    from datetime import date
    from matplotlib.backends.backend_pdf import PdfPages
    path = os.path.join(out, 'report.pdf')
    with PdfPages(path) as pdf:
        header = [f'Results: {results}',
                  f'Glaciers: {len(summary)}, members {int(summary.members.max())}, inversion year '
                  f'{summary.inversion_year.iloc[0]:g}, band height {settings["elevation_step"]} m, '
                  f'model error {settings["model_error"]} m/yr, tau_ref bounds {settings["tau_bounds"]}',
                  f'Created {date.today().isoformat()} by experiments/central_europe_esmda/sanity_report.py', '']
        findings = os.path.join(out, 'findings.md')
        if os.path.exists(findings):
            with open(findings) as f:
                paragraphs = [p.strip().replace('\n', ' ') if not p.startswith('- ') else p
                              for p in f.read().split('\n\n') if p.strip()]
            # bullets: one paragraph per '- ' line
            body = []
            for p in paragraphs:
                if p.lstrip().startswith('- '):
                    body += ['- ' + item.strip().replace('\n', ' ')
                             for item in p.strip()[2:].split('\n- ')]
                else:
                    body.append(p.lstrip('# '))
            header += body
        text_page(pdf, 'FROST sanity report', header)
        checks = [p for p in __doc__.split('\n\n')[1].split('\n') if p.strip()]
        items, current = [], ''
        for line in checks:
            if line[:2].strip() and line[0].isdigit():
                if current:
                    items.append(current)
                current = '- ' + line.strip()
            else:
                current += ' ' + line.strip()
        items.append(current)
        flags = [f'- {column}: {text}' for column, (_, text) in FLAGS.items()]
        text_page(pdf, 'Checks', items + ['Flags (red in the summary table):'] + flags)
        table_page(pdf, summary)
        image_page(pdf, os.path.join(out, 'overview.png'), 'Overview over all glaciers')
        for _, row in summary.iterrows():
            for name in ['bed', 'velocity', 'tau_ref', 'dhdt']:
                image_page(pdf, os.path.join(out, row.glacier, f'{name}.png'),
                           f'{row.glacier.replace("_", " ")} ({row.rgi_id}): {name}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--results', required=True, help='results folder of the run')
    parser.add_argument('--out', help='report folder (default experiments/<name>/sanity_report)')
    parser.add_argument('--glaciers', nargs='*',
                        help='RGI ids or glacier names (default: all of glaciers.txt)')
    parser.add_argument('--pdf_only', action='store_true',
                        help='only rebuild report.pdf from the figures and summary.tsv '
                             'of a previous run (e.g. after editing findings.md)')
    args = parser.parse_args()

    folder, settings = experiment_settings(args.results)
    out = args.out or os.path.join(folder, 'sanity_report')
    os.makedirs(out, exist_ok=True)
    if args.pdf_only:
        summary = pd.read_csv(os.path.join(out, 'summary.tsv'), sep='\t')
        print(f'PDF: {write_pdf(summary, out, args.results, settings)}')
        return
    glaciers_file = os.path.join(folder, 'glaciers.txt')
    if not os.path.exists(glaciers_file):
        glaciers_file = os.path.join(HERE, 'glaciers.txt')
    with open(glaciers_file) as f:
        glaciers = [tuple(line.split()[:2]) for line in f if line.strip()]
    if args.glaciers:
        glaciers = [g for g in glaciers if g[0] in args.glaciers or g[1] in args.glaciers]

    rows = []
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
        plot_bed(g, stats, os.path.join(target, 'bed.png'), title)
        plot_velocity(g, stats, os.path.join(target, 'velocity.png'), title)
        plot_tau(g, stats, settings['tau_bounds'], os.path.join(target, 'tau_ref.png'), title)
        plot_dhdt(g, stats, os.path.join(target, 'dhdt.png'), title)
        rows.append(stats)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out, 'summary.tsv'), sep='\t', index=False, float_format='%.4g')
    plot_overview(summary, os.path.join(out, 'overview.png'))
    write_report(summary, out, args.results, settings)
    print(f'Report: {os.path.join(out, "REPORT.md")}')
    print(f'PDF: {write_pdf(summary, out, args.results, settings)}')


if __name__ == '__main__':
    main()
