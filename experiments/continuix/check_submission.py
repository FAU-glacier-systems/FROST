#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Sanity check of the ContinuIX upload folder GROUP_<group>/: names, the
grid and CRS of the ContinuIX input, the ice mask, value ranges, units and
the glacier-mean mass balance (SMB - FDIV against the observed dh/dt).
Prints one line per file and every problem found.

Run from the repository root:
    python experiments/continuix/check_submission.py
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import xarray as xr
import yaml

from frost.preprocess import continuix

warnings.filterwarnings('ignore')

EXPERIMENT_DIR = 'experiments/continuix'
REQUIRED = {'SMB': 'm i.e./yr', 'UNCT_SMB': 'm i.e./yr', 'FDIV': 'm i.e./yr',
            'UNCT_FDIV': 'm i.e./yr', 'DENSITY': 'kg m-3'}
OPTIONAL = {'THK': 'm i.e.', 'BED': 'm a.s.l.'}
# plausible ranges (m i.e./yr; m) and the share of the ice that may lie
# outside: FDIV has local extremes in thick ice (EXP04 G01 up to 560 m/yr
# on 0.06 % of the cells), where the ensemble members also drift apart
RANGES = {'SMB': (-20, 10), 'UNCT_SMB': (0, 5), 'FDIV': (-20, 20),
          'UNCT_FDIV': (0, 10), 'THK': (0, 1500)}
MAX_OUTSIDE = {'FDIV': 0.01, 'UNCT_FDIV': 0.01}
# |SMB - FDIV - DHDT| of the glacier means, m/yr
BUDGET_TOL = 0.5


def check_file(path, cfg, problems):
    name = os.path.basename(path)
    match = re.fullmatch(r'(EXP\d\d)_([GS]\d\d)_method01\.nc', name)
    if not match:
        problems.append(f'{name}: name does not follow EXP##_G##_method##.nc')
        return None
    exp, glacier = match.groups()
    if os.path.basename(os.path.dirname(path)) != exp:
        problems.append(f'{name}: not in folder {exp}/')

    data_dir = cfg['exp02_dir'] if exp == 'EXP02' else cfg['data_dir']
    ref = continuix.open_experiment(data_dir, exp, glacier)
    ice = continuix.experiment_icemask(data_dir, exp, glacier, ref)
    raw = xr.open_dataset(path)
    out = raw.sortby(['x', 'y'])

    def problem(text):
        problems.append(f'{exp} {glacier}: {text}')

    # grid, order and CRS of the ContinuIX input
    if not (np.array_equal(out['x'].values, ref['x'].values)
            and np.array_equal(out['y'].values, ref['y'].values)):
        problem('grid differs from the ContinuIX input')
    y = raw['y'].values
    if int(y[0] > y[-1]) != ref.attrs['y_descending']:
        problem('y order differs from the ContinuIX input')
    if out['spatial_ref'].attrs.get('crs_wkt') \
            != ref['spatial_ref'].attrs.get('crs_wkt'):
        problem('CRS differs from the ContinuIX input')

    # variables, units, mask, ranges
    for var, units in {**REQUIRED, **OPTIONAL}.items():
        if var not in out:
            if var in REQUIRED:
                problem(f'{var} missing')
            continue
        v = out[var].values
        if out[var].attrs.get('units') != units:
            problem(f'{var} units {out[var].attrs.get("units")!r}')
        if out[var].attrs.get('grid_mapping') != 'spatial_ref':
            problem(f'{var} has no grid_mapping')
        if np.isinf(v).any():
            problem(f'{var} has inf')
        if np.isfinite(v[~ice]).any():
            problem(f'{var} has values outside the ice mask')
        missing = ~np.isfinite(v[ice])
        if missing.any():
            problem(f'{var} NaN on {missing.mean():.1%} of the ice')
        if var in RANGES and np.isfinite(v[ice]).any():
            lo, hi = RANGES[var]
            values = v[ice][np.isfinite(v[ice])]
            outside = np.mean((values < lo) | (values > hi))
            if outside > MAX_OUTSIDE.get(var, 0.0):
                problem(f'{var} outside {lo}..{hi} on {outside:.2%} of the '
                        f'ice (range {values.min():.1f}..{values.max():.1f})')
    if 'DENSITY' in out and not np.allclose(out['DENSITY'].values[ice], 910):
        problem('DENSITY is not 910')
    if 'UNCT_SMB' in out and np.nanmin(out['UNCT_SMB'].values[ice]) <= 0:
        problem('UNCT_SMB is zero somewhere (collapsed ensemble)')
    if ('THK' in out) != ('BED' in out):
        problem('THK without BED or BED without THK')
    if 'THK' in out:
        dem = ref['DEM'].values.astype(np.float64)
        diff = (out['BED'].values + out['THK'].values - dem)[ice]
        ok = np.isfinite(diff) & np.isfinite(dem[ice])
        if ok.any() and np.max(np.abs(diff[ok])) > 0.1:
            problem(f'BED + THK differs from DEM by up to '
                    f'{np.max(np.abs(diff[ok])):.1f} m')

    # attributes
    for attr in ['description', 'period', 'smb_parameters',
                 'smb_parameters_mean', 'smb_parameters_std']:
        if not out.attrs.get(attr):
            problem(f'global attribute {attr} missing or empty')

    # glacier-mean balance: SMB - FDIV against the observed dh/dt
    dhdt = continuix._gaps_to_nan(ref['DHDT'].values, ice)[ice]
    smb = np.nanmean(out['SMB'].values[ice])
    fdiv = np.nanmean(out['FDIV'].values[ice])
    obs = np.nanmean(dhdt)
    if abs(smb - fdiv - obs) > BUDGET_TOL:
        problem(f'SMB - FDIV = {smb - fdiv:.2f} m/yr, observed dh/dt '
                f'{obs:.2f} m/yr')
    return (exp, glacier, ice.sum(), smb, np.nanmean(out['UNCT_SMB'].values[ice]),
            fdiv, obs, sorted(set(OPTIONAL) & set(out.data_vars)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--group', default='FAU2')
    args = parser.parse_args()
    with open(os.path.join(EXPERIMENT_DIR, 'config.yml')) as f:
        cfg = yaml.safe_load(f)
    group = f'GROUP_{args.group}'
    folder = os.path.join(cfg['results_dir'], group)

    problems = []
    for name in [f'README_{group}_method01.txt', f'log_{group}.txt',
                 'SUBMISSION_CHECKLIST.txt']:
        if not os.path.exists(os.path.join(folder, name)):
            problems.append(f'{name} missing')
    files = sorted(glob.glob(os.path.join(folder, 'EXP*', '*.nc')))
    print(f"{'exp':6}{'glac':5}{'cells':>8}{'SMB':>7}{'UNCT':>6}{'FDIV':>7}"
          f"{'dh/dt':>7}{'SMB-FDIV-dhdt':>15}  extra")
    for path in files:
        row = check_file(path, cfg, problems)
        if row:
            exp, g, n, smb, unct, fdiv, obs, extra = row
            print(f'{exp:6}{g:5}{n:8d}{smb:7.2f}{unct:6.2f}{fdiv:7.2f}'
                  f'{obs:7.2f}{smb - fdiv - obs:15.2f}  {",".join(extra)}')
    print(f'\n{len(files)} files, {len(problems)} problems')
    for p in problems:
        print('  -', p)


if __name__ == '__main__':
    main()
