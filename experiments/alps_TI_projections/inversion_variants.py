#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Compare inversion strategies (inversion_variants/<variant>.yaml) on the
glaciers of alps_TI_projections: thickness against GlaThiDa (not used in the
fit) and the Millan product, surface velocity against the observations.

Run from the repository root:
    python experiments/alps_TI_projections/inversion_variants.py --rgi_id <id>
        inversions of all variants for one glacier (GPU job, see
        inversion_variants_sbatch.sh)
    python experiments/alps_TI_projections/inversion_variants.py --collect
        table of the scores: tables/inversion_variants.tsv
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(EXPERIMENT_DIR)))
from frost.preprocess import download_data, igm_inversion

VARIANTS = ['igm_guess', 'slide_only', 'joint']
SOURCE_DIR = os.path.join('data', 'results', 'alps_TI_projections')
RESULTS_DIR = os.path.join('data', 'results', 'alps_TI_inversion_variants')


def source_data(rgi_id):
    """Downloaded inputs of the glacier (OGGM shop, TI climate); downloads
    them if the glacier has none yet."""
    data_dir = os.path.join(SOURCE_DIR, rgi_id, 'Preprocess', 'data')
    if not os.path.exists(os.path.join(data_dir, 'input.nc')):
        download_data.main(rgi_id=rgi_id,
                           rgi_id_dir=os.path.join(SOURCE_DIR, rgi_id),
                           smb_model='TI', target_resolution='None',
                           oggm_shop=True)
    return data_dir


def scores(rgi_id_dir):
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'outputs', 'output.nc')) as ds:
        icemask = np.array(ds['icemask'][:]) > 0.5
        thk = np.array(ds['thk'][:])
        vel = np.array(ds['velsurf_mag'][:])
        vel_obs = np.array(ds['velsurfobs_mag'][:])
        tau_ref = np.array(ds['tau_ref'][:])
        dx = float(ds['x'][1] - ds['x'][0])
    with Dataset(os.path.join(rgi_id_dir, 'Preprocess', 'data', 'input.nc')) as ds:
        product = np.array(ds['thk_product'][:]) if 'thk_product' in ds.variables \
            else np.array(ds['thk'][:])
    observed = icemask & np.isfinite(vel_obs) & (vel_obs > 0)
    result = dict(
        volume_km3=float(thk[icemask].sum() * dx * dx / 1e9),
        thk_mean=float(thk[icemask].mean()),
        product_thk_mean=float(np.nanmean(np.where(product > 0, product, np.nan)[icemask])),
        vel_obs_median=float(np.median(vel_obs[observed])),
        vel_median=float(np.median(vel[observed])),
        vel_rms=float(np.sqrt(np.mean((vel - vel_obs)[observed] ** 2))),
        tau_ref_median=float(np.median(tau_ref[icemask])),
        tau_ref_p10=float(np.percentile(tau_ref[icemask], 10)),
        tau_ref_p90=float(np.percentile(tau_ref[icemask], 90)),
        # patchiness: mean jump of tau_ref between neighbouring ice cells
        tau_ref_roughness=float(np.nanmean(np.abs(np.diff(
            np.where(icemask, tau_ref, np.nan), axis=1)))))
    validation = os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                              'thickness_validation.json')
    if os.path.exists(validation):
        with open(validation) as f:
            glathida = json.load(f)
        result.update(glathida_n=glathida.get('n_observed'),
                      glathida_mean=glathida.get('mean_observed'),
                      glathida_bias=glathida.get('bias'),
                      glathida_rms=glathida.get('rms'))
    return result


def run(rgi_id, variants):
    data_dir = source_data(rgi_id)
    if len(variants) > 1:
        # one process per variant: IGM runs through hydra, which does not
        # start twice in one process and exits it on errors
        for variant in variants:
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            '--rgi_id', rgi_id, '--variants', variant])
        return
    for variant in variants:
        rgi_id_dir = os.path.join(RESULTS_DIR, variant, rgi_id)
        shutil.rmtree(rgi_id_dir, ignore_errors=True)
        shutil.copytree(data_dir, os.path.join(rgi_id_dir, 'Preprocess', 'data'))
        try:
            igm_inversion.main(
                rgi_id_dir=rgi_id_dir,
                params_inversion_path=os.path.join(
                    EXPERIMENT_DIR, 'inversion_variants', f'{variant}.yaml'))
            result = scores(rgi_id_dir)
        except (Exception, SystemExit) as error:
            result = dict(error=repr(error))
        print(variant, rgi_id, result)
        with open(os.path.join(rgi_id_dir, 'scores.json'), 'w') as f:
            json.dump(result, f, indent=1)


def collect():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*', '*', 'scores.json'))):
        variant, rgi_id = path.split(os.sep)[-3:-1]
        with open(path) as f:
            rows.append(dict(variant=variant, rgi_id=rgi_id, **json.load(f)))
    table = pd.DataFrame(rows)
    os.makedirs(os.path.join(EXPERIMENT_DIR, 'tables'), exist_ok=True)
    out_file = os.path.join(EXPERIMENT_DIR, 'tables', 'inversion_variants.tsv')
    table.to_csv(out_file, sep='\t', index=False, float_format='%.3g')
    print(table.to_string(index=False, float_format='%.3g'))
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rgi_id')
    parser.add_argument('--variants', nargs='+', default=VARIANTS)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    else:
        run(args.rgi_id, args.variants)
