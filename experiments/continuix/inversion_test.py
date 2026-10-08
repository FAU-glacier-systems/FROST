#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Does the tau_ref inversion need other settings on a 25 m grid? At 25 m the
velocity fit of G03-G06 got worse than at 50 m (res25 test, 8 Oct): three
glaciers stopped at nbitmax 500 with the cost still falling, lam 1e11 keeps
tau_ref as smooth as at 50 m, and the pretrained emulator is never adapted
to the glacier (retrain_iter 0). One inversion per glacier and variant
(params_inversion_tau.yaml with the overrides in VARIANTS), EXP01 only.

Velocity statistics on ice with observations: on the model grid, and for
25 m runs also on 2x2 block means (50 m), comparable with the 50 m runs
(a finer grid has more noise per cell).

Run from the repository root, on a GPU node (one task per line of
inversion_test_tasks.txt, "<glacier> <variant>"):
    sbatch --array=1-15 experiments/continuix/inversion_test.sh
and the table on a login node:
    python experiments/continuix/inversion_test.py --summary
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import yaml
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
from frost.preprocess import continuix, igm_inversion

OUT_DIR = os.path.join('data', 'results', 'continuix', 'inversion_test')
PARAMS = os.path.join('experiments', 'continuix', 'params_inversion_tau.yaml')
CONFIG = os.path.join('experiments', 'continuix', 'config.yml')

# resolution (m) and overrides of assimilations.field_inversion
VARIANTS = {
    # converged: does the cap alone explain the worse fit?
    'r25_nbit2000': dict(resolution=25, nbitmax=2000),
    # finer sliding, which only the finer grid can carry
    'r25_lam3e10': dict(resolution=25, nbitmax=2000, lam=3e10),
    'r25_lam1e10': dict(resolution=25, nbitmax=2000, lam=1e10),
    # emulator fine-tuned on the glacier between minimisation phases
    'r25_retrain': dict(resolution=25, nbitmax=1000, retrain_iter=2),
    # the same at 50 m: would retraining help the current pipeline?
    'r50_retrain': dict(resolution=50, nbitmax=1000, retrain_iter=2),
}


def variant_dir(glacier, variant):
    return os.path.join(OUT_DIR, glacier, variant)


def params_for(variant):
    with open(PARAMS) as f:
        params = yaml.safe_load(f)
    params = copy.deepcopy(params)
    v = VARIANTS[variant]
    fi = params['assimilations']['field_inversion']
    opt = fi.setdefault('optimization', {})
    if 'nbitmax' in v:
        opt['nbitmax'] = int(v['nbitmax'])
    if 'retrain_iter' in v:
        opt['retrain_iter'] = int(v['retrain_iter'])
    if 'lam' in v:
        fi['objective']['regularization'][0]['lam'] = float(v['lam'])
    return params


def run(glacier, variant):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    rgi_id_dir = variant_dir(glacier, variant)
    continuix.prepare_input(cfg['data_dir'], 'EXP01', glacier, rgi_id_dir,
                            resolution=VARIANTS[variant]['resolution'])
    params_path = os.path.abspath(os.path.join(rgi_id_dir,
                                               'params_inversion.yaml'))
    with open(params_path, 'w') as f:
        f.write('# @package _global_\n')
        yaml.dump(params_for(variant), f, sort_keys=False)
    t0 = time.time()
    igm_inversion.main(rgi_id_dir=os.path.abspath(rgi_id_dir),
                       params_inversion_path=params_path,
                       min_velocity_p99=cfg['min_velocity_p99'])
    seconds = time.time() - t0
    stats = statistics(os.path.join(rgi_id_dir, 'Preprocess', 'outputs',
                                    'output.nc'))
    stats.update(glacier=glacier, variant=variant, seconds=seconds,
                 **VARIANTS[variant])
    with open(os.path.join(rgi_id_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def blocks(field, n=2):
    """Mean over n x n blocks (NaN if any cell is NaN)."""
    ny, nx = (s // n * n for s in field.shape)
    return field[:ny, :nx].reshape(ny // n, n, nx // n, n).mean(axis=(1, 3))


def velocity_fit(model, obs):
    ok = np.isfinite(model) & np.isfinite(obs)
    e = model[ok] - obs[ok]
    return dict(n=int(ok.sum()), rms=float(np.sqrt(np.mean(e ** 2))),
                ratio=float(model[ok].mean() / obs[ok].mean()),
                r=float(np.corrcoef(model[ok], obs[ok])[0, 1]))


def statistics(output_file):
    """Velocity fit, convergence and tau_ref spread of one inversion
    (also used for the existing runs: EXP01 50 m, res25 25 m)."""
    with Dataset(output_file) as nc:
        f = {k: np.array(nc[k][:], dtype=float) for k in
             ['icemask', 'velsurf_mag', 'velsurfobs_mag', 'tau_ref',
              'da_cost_data_hist', 'da_cost_reg_hist', 'x']}
    ice = f['icemask'] > 0.5
    obs = np.where(ice & (f['velsurfobs_mag'] > 0), f['velsurfobs_mag'],
                   np.nan)
    model = np.where(np.isfinite(obs), f['velsurf_mag'], np.nan)
    dx = float(abs(f['x'][1] - f['x'][0]))
    total = f['da_cost_data_hist'] + f['da_cost_reg_hist']
    log_tau = np.log10(f['tau_ref'][ice])
    stats = dict(dx=dx, iterations=len(total),
                 cost_data=float(f['da_cost_data_hist'][-1]),
                 cost_reg=float(f['da_cost_reg_hist'][-1]),
                 # relative decrease of the total cost in the last 50 steps
                 still_falling=float((total[max(-51, -len(total))] - total[-1])
                                     / abs(total[-1])),
                 log_tau_p5=float(np.percentile(log_tau, 5)),
                 log_tau_p95=float(np.percentile(log_tau, 95)),
                 grid=velocity_fit(model, obs))
    factor = int(round(50 / dx))
    stats['at_50m'] = (velocity_fit(blocks(model, factor), blocks(obs, factor))
                       if factor > 1 else stats['grid'])
    return stats


def summary():
    rows = []
    for g in sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                     for p in
                     glob.glob(os.path.join(OUT_DIR, '*', '*', 'stats.json'))}):
        refs = [('r50 current', os.path.join('data', 'results', 'continuix',
                                             'EXP01', g)),
                ('r25 current', os.path.join('data', 'results', 'continuix',
                                             'res25', 'EXP01', g))]
        for name, d in refs:
            out = os.path.join(d, 'Preprocess', 'outputs', 'output.nc')
            if os.path.exists(out):
                rows.append(dict(statistics(out), glacier=g, variant=name))
        for variant in VARIANTS:
            p = os.path.join(variant_dir(g, variant), 'stats.json')
            if os.path.exists(p):
                rows.append(json.load(open(p)))
    print(f"{'':4} {'variant':14} {'it':>5} {'falling':>8} {'log tau p5..p95':>16}"
          f" | grid: {'r':>5} {'RMS':>5} {'ratio':>5} | 50 m: {'r':>5} {'RMS':>5}"
          f" {'ratio':>5} | {'min':>5}")
    for s in rows:
        gr, b = s['grid'], s['at_50m']
        minutes = s.get('seconds', np.nan) / 60
        print(f"{s['glacier']:4} {s['variant']:14} {s['iterations']:>5}"
              f" {100 * s['still_falling']:>7.2f}% {s['log_tau_p5']:>7.2f}..{s['log_tau_p95']:<7.2f}"
              f" |       {gr['r']:>5.2f} {gr['rms']:>5.1f} {gr['ratio']:>5.2f}"
              f" |       {b['r']:>5.2f} {b['rms']:>5.1f} {b['ratio']:>5.2f} | {minutes:>5.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--glacier')
    parser.add_argument('--variant', choices=list(VARIANTS))
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summary()
    else:
        run(args.glacier, args.variant)


if __name__ == '__main__':
    main()
