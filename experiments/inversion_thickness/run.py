#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Thickness free in the inversion (2026-11-11): does fitting the thickness
together with tau_ref, with IceBoost as a soft prior, give deeper ice where
IceBoost is too thin (Aletsch, Rhone) and close the flux gap of
central_europe_combined_retrain/flux_gate_report.pdf? Generic settings only,
the same for every glacier; GlaThiDa and GLAMOS are used for checking only.

Inputs: Preprocess/data of central_europe_retrain (IceBoost, ITS_LIVE,
Hugonnet), copied per variant to data/results/inversion_thickness/<variant>/.

Run from the repository root:
    python experiments/inversion_thickness/run.py --rgi_id <id> [--variants a b]
        inversions (GPU job, run_sbatch.sh)
    python experiments/inversion_thickness/run.py --collect
        scores of all variants: tables/scores.tsv, tables/flux_gates.tsv
        (inputs and results only, fine on the login node)
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

SOURCE = os.path.join('data', 'results', 'central_europe_retrain')
RESULTS = os.path.join('data', 'results', 'inversion_thickness')
VARIANTS = ['ref', 'joint_s50', 'joint_s100', 'joint_s50_retrain', 'joint_mc_ti', 'diag_mc_glamos']
NAMES = {'RGI2000-v7.0-G-11-02596': 'Grosser Aletschgletscher',
         'RGI2000-v7.0-G-11-01706': 'Rhonegletscher'}
GATES = {'RGI2000-v7.0-G-11-02596': (2500, 2800), 'RGI2000-v7.0-G-11-01706': (2800,)}
DEEP = 300.0          # m: "deep ice" for the GlaThiDa check
DEPTH_AVG = 0.9       # depth-averaged / surface speed for the observed flux


def ti_prior_smb(rgi_id, s, ice, dhdt, years):
    """Mean TI SMB (m ice/yr) of `years` on the surface s with OGGM's prior
    parameters (mb_calib.json), temp_bias shifted so that the glacier-wide
    SMB equals the glacier-wide dh/dt (the budget closes: no net flux)."""
    from frost.glacier_model.no_flow import load_climate, smb_ti
    d = os.path.join(SOURCE, rgi_id, 'Preprocess', 'data', rgi_id)
    with open(os.path.join(d, 'mb_calib.json')) as f:
        cal = json.load(f)
    climate = load_climate(os.path.join(d, 'climate_historical.nc'))
    target = np.nanmean(dhdt[ice])

    def smb(shift):
        p = {k: np.array([cal[k] + (shift if k == 'temp_bias' else 0.0)],
                         np.float32).reshape(-1, 1, 1)
             for k in ('melt_f', 'prcp_fac', 'temp_bias')}
        return np.mean([smb_ti(s, climate, y, p)[0] for y in range(years[0], years[1] + 1)], 0)

    lo, hi = -3.0, 3.0                    # warmer -> more negative
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if np.mean(smb(mid)[ice]) > target:
            lo = mid
        else:
            hi = mid
    print(f'TI prior: temp_bias {cal["temp_bias"]:.2f} {mid:+.2f} K closes the budget '
          f'({target:.2f} m/yr)', flush=True)
    return smb(mid)


def write_divflux_obs(rgi_id, rgi_dir, source, years=(2015, 2019)):
    """divfluxobs = SMB - dh/dt (m ice/yr) of the ITS_LIVE years on the
    surface of the velocity year, into input.nc, for a divflux misfit (mass
    conservation). source 'ti': TI with OGGM's prior parameters (generic);
    'glamos': GLAMOS band balances (diagnostic only)."""
    from frost.preprocess.igm_inversion import velocity_year
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'central_europe_combined_retrain'))
    from flux_gate_report import glamos_bands, hugonnet
    inp = os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc')
    with Dataset(inp) as nc:
        x, y = read(nc, 'x'), read(nc, 'y')
        ice = read(nc, 'icemask') > 0.5
        s2000 = read(nc, 'usurf_2000') if 'usurf_2000' in nc.variables else read(nc, 'usurf')
        dhdt_long = np.nan_to_num(read(nc, 'dhdt'))
        epsg = nc.epsg
    year = velocity_year(inp)
    s = s2000 + dhdt_long * (year - 2000)
    dhdt = hugonnet(f'{years[0]}-01-01_{years[1] + 1}-01-01', x, y, epsg)
    if source == 'ti':
        smb = ti_prior_smb(rgi_id, s, ice, dhdt, years)
    elif source == 'glamos':
        ba = glamos_bands(NAMES[rgi_id], years)
        band = np.floor(s / 100) * 100
        smb = np.full(s.shape, np.nan)
        smb[ice] = pd.Series(band[ice]).map(ba).values / 0.91
        # constant offset so that the budget closes (outline / period of
        # GLAMOS vs Hugonnet); the shape of the profile stays GLAMOS
        valid = ice & np.isfinite(smb) & np.isfinite(dhdt)
        offset = np.mean(dhdt[valid]) - np.mean(smb[valid])
        smb = smb + offset
        print(f'GLAMOS shifted by {offset:+.2f} m ice/yr to close the budget', flush=True)
    else:
        raise ValueError(source)
    obs = np.where(ice & np.isfinite(smb) & np.isfinite(dhdt), smb - dhdt, np.nan)
    print(f'divfluxobs ({source}, {years[0]}-{years[1]}): glacier mean '
          f'{np.nanmean(obs):+.2f} m/yr, tongue (lowest 10 %) '
          f'{np.nanmean(obs[ice & (s < np.nanpercentile(s[ice], 10))]):+.2f}', flush=True)
    with Dataset(inp, 'r+') as nc:
        if 'divfluxobs' not in nc.variables:
            nc.createVariable('divfluxobs', 'f4', ('y', 'x'))
        nc['divfluxobs'][:] = obs.astype(np.float32)
        nc['divfluxobs'].setncattr('description', f'{source} SMB - Hugonnet dh/dt, '
                                   f'{years[0]}-{years[1]}, m ice/yr')


def run(rgi_id, variants):
    import yaml
    from frost.preprocess import igm_inversion
    for variant in variants:
        params = os.path.join(HERE, 'variants', f'{variant}.yaml')
        with open(params) as f:
            frost = yaml.safe_load(f).get('frost') or {}
        rgi_dir = os.path.join(RESULTS, variant, rgi_id)
        data = os.path.join(rgi_dir, 'Preprocess', 'data')
        if os.path.exists(data):
            shutil.rmtree(data)
        shutil.copytree(os.path.join(SOURCE, rgi_id, 'Preprocess', 'data'), data)
        print(f'=== {variant} {rgi_id}', flush=True)
        if frost.get('divflux_obs'):
            write_divflux_obs(rgi_id, rgi_dir, frost['divflux_obs'])
        igm_inversion.main(rgi_id_dir=rgi_dir, params_inversion_path=params)


def read(nc, name):
    v = nc[name][:]
    return np.array(v.filled(np.nan) if np.ma.isMaskedArray(v) else v, dtype=float)


def scores(rgi_id, variant):
    from frost.preprocess.thickness_validation import thkobs_at_year
    rgi_dir = os.path.join(RESULTS, variant, rgi_id)
    out = os.path.join(rgi_dir, 'Preprocess', 'outputs', 'output.nc')
    if not os.path.exists(out):
        return None
    with Dataset(out) as nc:
        thk = read(nc, 'thk')                 # 2000, the forward-run start
        thk_inv = read(nc, 'thk_inversion')   # velocity year
        vel, vel_obs = read(nc, 'velsurf_mag'), read(nc, 'velsurfobs_mag')
        tau = read(nc, 'tau_ref')
        x, y = read(nc, 'x'), read(nc, 'y')
    inp = os.path.join(rgi_dir, 'Preprocess', 'data', 'input.nc')
    with Dataset(inp) as nc:
        ice = read(nc, 'icemask') > 0.5
        prior = read(nc, 'thk_2000')
        dhdt = np.nan_to_num(read(nc, 'dhdt'))
    dx = abs(x[1] - x[0])
    glathida = os.path.join(rgi_dir, 'Preprocess', 'data', rgi_id, 'glathida_data.csv')
    thkobs, _ = thkobs_at_year(glathida, x, y, dhdt, 2000)
    obs = ice & np.isfinite(thkobs) & (thkobs > 0)
    deep = obs & (thkobs > DEEP)
    moving = ice & np.isfinite(vel_obs) & (vel_obs > 0)
    return dict(
        glacier=NAMES[rgi_id], variant=variant,
        volume_km3=thk[ice].sum() * dx * dx / 1e9,
        volume_prior_km3=prior[ice].sum() * dx * dx / 1e9,
        thk_max=np.nanmax(thk[ice]), thk_change_mean=np.mean((thk - prior)[ice]),
        glathida_n=int(obs.sum()), glathida_bias=np.mean((thk - thkobs)[obs]),
        glathida_rms=np.sqrt(np.mean((thk - thkobs)[obs] ** 2)),
        deep_n=int(deep.sum()), deep_bias=np.mean((thk - thkobs)[deep]),
        deep_ratio=np.median(thkobs[deep] / thk[deep]),
        speed_ratio_median=np.median(vel[moving] / vel_obs[moving]),
        speed_rms=np.sqrt(np.mean((vel - vel_obs)[moving] ** 2)),
        tau_ref_median=np.median(tau[ice]))


def flux_gates(rgi_id, variant):
    """ITS_LIVE x thickness of the inversion (velocity year) through the
    gates, against the flux the data need in the ITS_LIVE years (Hugonnet
    2015-2020 - GLAMOS 2015-2019): validation only."""
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'central_europe_combined_retrain'))
    from flux_gate import gate_inflow
    from flux_gate_report import load, required
    out = os.path.join(RESULTS, variant, rgi_id, 'Preprocess', 'outputs', 'output.nc')
    if not os.path.exists(out):
        return []
    g = load(rgi_id, NAMES[rgi_id])
    req = required(g, 'em_1519')
    inp = os.path.join(SOURCE, rgi_id, 'Preprocess', 'data', 'input.nc')
    with Dataset(inp) as nc:
        ux, uy = read(nc, 'uvelsurfobs'), read(nc, 'vvelsurfobs')
    with Dataset(out) as nc:
        thk_inv = read(nc, 'thk_inversion')
    rows = []
    for h in GATES[rgi_id]:
        inside = g['ice'] & (g['s'] < h)
        q = gate_inflow(inside, g['ice'], ux, uy, thk_inv, g['dx']) * DEPTH_AVG / 1e9
        rows.append(dict(glacier=NAMES[rgi_id], variant=variant, gate_m=h,
                         required_1519=req.loc[h], its_live_x_thk=q, ratio=q / req.loc[h]))
    return rows


def collect():
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    rows, gates = [], []
    for variant in VARIANTS:
        for rgi_id in NAMES:
            s = scores(rgi_id, variant)
            if s:
                rows.append(s)
                gates += flux_gates(rgi_id, variant)
    for name, table in (('scores', rows), ('flux_gates', gates)):
        df = pd.DataFrame(table)
        df.to_csv(os.path.join(HERE, 'tables', f'{name}.tsv'), sep='\t', index=False,
                  float_format='%.4g')
        print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgi_id')
    parser.add_argument('--variants', nargs='+', default=VARIANTS)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    else:
        run(args.rgi_id, args.variants)
