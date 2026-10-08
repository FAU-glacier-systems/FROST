#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Where does the time of an ES-MDA forward round go? On copies of calibrated
ensemble members (input.nc and params.yaml as the last round left them):

  tf_import   python: import tensorflow and initialise the GPU
  cli_0yr     one igm_run with a zero-length period: start-up only
  cli_full    one igm_run over the dh/dt period
  cli_36_0yr  36 igm_run at once (as the pipeline), zero-length period
  cli_36_full 36 igm_run at once over the dh/dt period
  inproc_full one Python process calling igm's main() 3x in a row: the
              cost per run once TF is loaded (and whether repeated calls
              work)

Writes data/results/continuix/bench/<glacier>/bench.json. Needs a GPU.
Run from the repository root:
    sbatch experiments/continuix/bench_forward.sh data/results/continuix/res25/EXP01/G02
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

N_MEMBERS = 36


def copy_member(source, target):
    """Member dir with input.nc and params.yaml, without old outputs."""
    shutil.rmtree(target, ignore_errors=True)
    for sub in ('data', 'experiment'):
        shutil.copytree(os.path.join(source, sub), os.path.join(target, sub))
    return target


def period(member):
    with open(os.path.join(member, 'experiment', 'params.yaml')) as f:
        t = yaml.safe_load(f)['processes']['time']
    return t['start'], t['end']


def cli(members, end):
    """igm_run in every member dir at once; wall time of the slowest."""
    t0 = time.time()
    procs = [subprocess.Popen(
        ['igm_run', '+experiment=params', f'processes.time.end={end}'],
        cwd=m, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for m in members]
    codes = [p.wait() for p in procs]
    return time.time() - t0, codes


def inproc(member, end, repeats=3):
    """igm's main() called repeatedly in a fresh Python process."""
    code = f"""
import os, sys, time, json
t0 = time.time()
from igm.igm_run import main
times = [time.time() - t0]
for i in range({repeats}):
    os.chdir({member!r})
    sys.argv = ['igm_run', '+experiment=params', 'processes.time.end={end}']
    t = time.time()
    try:
        main()
    except SystemExit:
        pass
    times.append(time.time() - t)
print('INPROC', json.dumps(times))
"""
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True)
    for line in out.stdout.splitlines():
        if line.startswith('INPROC'):
            return json.loads(line[6:])
    return out.stderr[-2000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('glacier_dir', help='e.g. data/results/continuix/res25/EXP01/G02')
    args = parser.parse_args()
    glacier = os.path.basename(args.glacier_dir.rstrip('/'))
    out_dir = os.path.abspath(os.path.join('data', 'results', 'continuix',
                                           'bench', glacier))
    os.makedirs(out_dir, exist_ok=True)
    members = [copy_member(
        os.path.join(args.glacier_dir, 'Ensemble', f'Member_{i}'),
        os.path.join(out_dir, f'Member_{i}')) for i in range(N_MEMBERS)]
    start, end = period(members[0])
    result = dict(glacier_dir=args.glacier_dir, start=start, end=end,
                  job_cores=len(os.sched_getaffinity(0)))

    t0 = time.time()
    subprocess.run([sys.executable, '-c', 'import tensorflow as tf; '
                    'tf.config.list_physical_devices("GPU")'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result['tf_import'] = time.time() - t0
    for name, ms, e in [('cli_0yr', members[:1], start),
                        ('cli_full', members[:1], end),
                        ('cli_36_0yr', members, start),
                        ('cli_36_full', members, end)]:
        result[name], codes = cli(ms, e)
        if any(codes):
            result[name + '_returncodes'] = codes
        print(name, f'{result[name]:.1f} s', flush=True)
    result['inproc_full'] = inproc(members[0], end)
    print(json.dumps(result, indent=2))
    with open(os.path.join(out_dir, 'bench.json'), 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
