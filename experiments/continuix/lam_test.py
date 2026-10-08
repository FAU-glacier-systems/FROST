#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Which lam (tau_ref regularisation) and grid give the best dh/dt fit? The
inversion test (inversion_test.py) showed that lower lam and more
iterations fit the velocities better at 25 m, but a lower lam always fits
velocities better; the calibrated dh/dt decides. Full runs (prepare,
inversion with nbitmax 2000, calibrate, submit) of EXP01 for G04, G05,
G06 at 25 and 50 m with lam 3e10, 1e10, 3e9; one lam for all glaciers.
The current runs (lam 1e11, nbitmax 500) are the reference: EXP01 (50 m)
and res25/EXP01 (25 m).

Every task scores its posterior: band-mean dh/dt of the ensemble mean
against the observed band means (band RMS), chi2/n, glacier means, and
the velocity fit of the inversion (band_score.json in the glacier dir).

    python experiments/continuix/lam_test.py --setup    # configs, task list
    sbatch --array=1-18 experiments/continuix/lam_test.sh
    python experiments/continuix/lam_test.py --summary  # login node
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.getcwd())

EXPERIMENT_DIR = os.path.join('experiments', 'continuix')
OUT_DIR = os.path.join('data', 'results', 'continuix', 'lam_test')
GLACIERS = ['G04', 'G05', 'G06']
LAMS = [3e10, 1e10, 3e9]
RESOLUTIONS = [25, 50]
NBITMAX = 2000
# reference runs (lam 1e11, nbitmax 500) per resolution
REFERENCE = {50: os.path.join('data', 'results', 'continuix'),
             25: os.path.join('data', 'results', 'continuix', 'res25')}


def variant(resolution, lam):
    return f'r{resolution}_lam{lam:.0e}'.replace('+', '')


def setup():
    """One folder per variant with config.yml and params_inversion_tau.yaml
    (run_continuix reads the params next to the config), and the task
    list lam_test_tasks.txt ("<glacier> <variant>")."""
    with open(os.path.join(EXPERIMENT_DIR, 'config.yml')) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(EXPERIMENT_DIR, 'params_inversion_tau.yaml')) as f:
        params = yaml.safe_load(f)
    tasks = []
    for resolution in RESOLUTIONS:
        for lam in LAMS:
            v = variant(resolution, lam)
            d = os.path.join(OUT_DIR, v)
            os.makedirs(d, exist_ok=True)
            c = dict(cfg, results_dir=d, resolution=resolution)
            with open(os.path.join(d, 'config.yml'), 'w') as f:
                f.write(f'# lam_test.py: config.yml with resolution {resolution} m '
                        f'and lam {lam:.0e}\n')
                yaml.dump(c, f, sort_keys=False)
            p = json.loads(json.dumps(params))
            fi = p['assimilations']['field_inversion']
            fi['objective']['regularization'][0]['lam'] = float(lam)
            fi.setdefault('optimization', {})['nbitmax'] = NBITMAX
            with open(os.path.join(d, 'params_inversion_tau.yaml'), 'w') as f:
                f.write('# @package _global_\n')
                yaml.dump(p, f, sort_keys=False)
            tasks += [f'{g} {v}' for g in GLACIERS]
    with open(os.path.join(EXPERIMENT_DIR, 'lam_test_tasks.txt'), 'w') as f:
        f.write('\n'.join(tasks) + '\n')
    print(f'{len(tasks)} tasks in {EXPERIMENT_DIR}/lam_test_tasks.txt')


def score(rgi_id_dir):
    """Posterior band fit of one calibrated glacier (reads all members:
    run on a compute node)."""
    from netCDF4 import Dataset
    from frost.calibration.observation_provider import ObservationProvider
    with open(os.path.join(EXPERIMENT_DIR, 'config.yml')) as f:
        cfg = yaml.safe_load(f)
    glacier = os.path.basename(rgi_id_dir.rstrip('/'))
    enkf = dict(cfg['EnKF'])
    enkf.update(cfg.get('glaciers', {}).get(glacier, {}).get('EnKF', {}))
    obs = ObservationProvider(rgi_id_dir, glacier, enkf['elev_band_height'],
                              None, False, enkf['model_error'])
    members = sorted(glob.glob(os.path.join(rgi_id_dir, 'Ensemble',
                                            'Member_*', 'outputs', 'output.nc')))
    bands = []
    for m in members:
        with Dataset(m) as nc:
            usurf = np.array(nc['usurf'][-1], dtype=float)
        bands.append(obs.band_mean((usurf - obs.usurf) / obs.period))
    bands = np.array(bands)
    model = bands.mean(0)
    residual = model - obs.observation
    counts = np.bincount(obs.pixel_band, minlength=obs.num_bins)
    cal = json.load(open(os.path.join(rgi_id_dir, 'calibration_results.json')))
    post = cal['diagnostics'][-1]
    sys.path.insert(0, EXPERIMENT_DIR)
    from inversion_test import statistics
    inv = statistics(os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                                  'output.nc'))
    result = dict(
        members=len(members), bands=int(obs.num_bins),
        band_rms=float(np.sqrt(np.mean(residual ** 2))),
        # area-weighted: the glacier-wide error
        band_rms_area=float(np.sqrt(np.average(residual ** 2, weights=counts))),
        obs_mean=float(np.average(obs.observation, weights=counts)),
        model_mean=float(np.average(model, weights=counts)),
        chi2=float(post['misfit']),
        parameters=post['parameter_mean'], parameter_std=post['parameter_std'],
        velocity=inv['at_50m'], inversion_iterations=inv['iterations'])
    with open(os.path.join(rgi_id_dir, 'band_score.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def summary():
    rows = []
    for g in GLACIERS:
        for resolution in RESOLUTIONS:
            runs = [('lam1e11', os.path.join(REFERENCE[resolution], 'EXP01', g))]
            runs += [(f'lam{lam:.0e}'.replace('+', ''),
                      os.path.join(OUT_DIR, variant(resolution, lam), 'EXP01', g))
                     for lam in LAMS]
            for name, d in runs:
                p = os.path.join(d, 'band_score.json')
                if os.path.exists(p):
                    rows.append((g, resolution, name, json.load(open(p))))
    print(f"{'':4} {'grid':>4} {'lam':8} {'band RMS':>8} {'area-w.':>7} {'chi2/n':>6}"
          f" {'obs':>6} {'model':>6} {'ELA':>6} {'abl':>5} {'acc':>5} {'v r':>5} {'v RMS':>5}")
    for g, resolution, name, s in rows:
        p = s['parameters']
        print(f"{g:4} {resolution:>4} {name:8} {s['band_rms']:>8.2f} {s['band_rms_area']:>7.2f}"
              f" {s['chi2']:>6.2f} {s['obs_mean']:>6.2f} {s['model_mean']:>6.2f}"
              f" {p[0]:>6.0f} {p[1]:>5.1f} {p[2]:>5.1f}"
              f" {s['velocity']['r']:>5.2f} {s['velocity']['rms']:>5.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--score', metavar='RGI_ID_DIR')
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.setup:
        setup()
    if args.score:
        score(args.score)
    if args.summary:
        summary()


if __name__ == '__main__':
    main()
