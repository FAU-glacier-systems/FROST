#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
FROST for one ContinuIX experiment and glacier: (EXP02: raw GeoTIFFs and
shapefiles -> netCDF on the EXP01 grid ->) ContinuIX input -> IGM
inversion -> EnKF calibration of the ELA SMB model against the ContinuIX
dh/dt (ending with a forward run of the final ensemble) ->
EXP##_G##_method##.nc.

Run from the repository root:
    python experiments/continuix/run_continuix.py --exp EXP01 --glacier S01
"""

import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.getcwd())
import frost_calibration
from frost.preprocess import continuix, continuix_raw, igm_inversion

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

STEPS = ['raw', 'prepare', 'inversion', 'calibrate', 'submit']


def merge(base, override):
    """base with the values of override, nested dicts merged."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge(base[key], value)
        else:
            merged[key] = value
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--exp', required=True, help='e.g. EXP01')
    parser.add_argument('--glacier', required=True, help='e.g. G05')
    parser.add_argument('--config', default='experiments/continuix/config.yml')
    parser.add_argument('--steps', default=','.join(STEPS),
                        help=f'comma-separated subset of {STEPS}')
    args = parser.parse_args()
    steps = args.steps.split(',')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = merge(cfg, cfg.pop('glaciers', {}).get(args.glacier, {}))
    experiment_dir = os.path.dirname(args.config)

    rgi_id_dir = os.path.join(cfg['results_dir'], args.exp,
                              args.glacier)
    submission = os.path.join(
        cfg['results_dir'], 'submission', args.exp,
        f"{args.exp}_{args.glacier}_{cfg['method']}.nc")
    timings = {}

    def timed(step, function, *a, **kw):
        if step not in steps:
            return
        t0 = time.time()
        function(*a, **kw)
        timings[step] = time.time() - t0
        print(f'--- {step}: {timings[step]:.0f} s')

    # EXP02 comes as GeoTIFFs and shapefiles: built into a netCDF first
    data_dir = cfg['data_dir']
    if args.exp == continuix_raw.EXP:
        data_dir = cfg['exp02_dir']
        timed('raw', continuix_raw.build, cfg['data_dir'], cfg['data_dir'],
              args.glacier, data_dir)
    timed('prepare', continuix.prepare_input, data_dir, args.exp,
          args.glacier, rgi_id_dir, resolution=cfg['resolution'])

    def inversion():
        # thickness only along GPR profiles (EXP02): inverted as well
        with open(continuix.meta_path(rgi_id_dir)) as f:
            points = json.load(f).get('thk_source') == 'points'
        params = cfg['params_inversion_thkobs' if points
                     else 'params_inversion']
        igm_inversion.main(rgi_id_dir=rgi_id_dir,
                           params_inversion_path=os.path.join(experiment_dir,
                                                              params),
                           min_velocity_p99=cfg['min_velocity_p99'])

    timed('inversion', inversion)

    def calibrate():
        continuix.write_observations(rgi_id_dir)
        with open(continuix.meta_path(rgi_id_dir)) as f:
            meta = json.load(f)
        enkf = dict(cfg['EnKF'])
        enkf['smb_prior_mean'], enkf['smb_prior_std'] = continuix.smb_prior(
            meta, enkf['smb_prior_mean'], enkf['smb_prior_std'])
        frost_calibration.main(rgi_id=args.glacier, rgi_id_dir=rgi_id_dir,
                               smb_model=cfg['smb_model'], **enkf)

    timed('calibrate', calibrate)
    timed('submit', continuix.write_submission, rgi_id_dir, submission)

    # Computation log for log_GROUP.txt
    log = os.path.join(rgi_id_dir, 'timings.json')
    previous = json.load(open(log)) if os.path.exists(log) else {}
    previous.update(timings)
    previous['job'] = os.environ.get('SLURM_JOB_ID', 'interactive')
    previous['node'] = os.uname().nodename
    with open(log, 'w') as f:
        json.dump(previous, f, indent=4)


if __name__ == '__main__':
    main()
