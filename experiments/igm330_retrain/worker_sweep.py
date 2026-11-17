#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
How many forward workers per GPU? One glacier, one worker count: the
calibration's parallel forward pass (32 members 2000-2020, igm_default
retraining, live IGM model, 2-D output), run PASSES times with the same kept
worker pool, as EnKF.forward does it. Pass 1 includes the start of the
workers (TensorFlow/CUDA, first plain IGM run per worker); later passes are
the steady state of a calibration.

Members: SMB drawn once (seed 1) from the igm330_live posterior of the
glacier, so every worker count runs the same 32 members. Input: Member_0 of
data/results/igm330_live.

Measured per pass: wall time, peak GPU memory and mean GPU utilisation
(nvidia-smi, every second), peak host memory of all workers (VmRSS from
/proc) and of the job (cgroup memory.peak, if readable). A failed pass
(worker killed, out of memory) is recorded as failed.

    python experiments/igm330_retrain/worker_sweep.py <glacier> <workers>
    sbatch --array=0-17 experiments/igm330_retrain/worker_sweep_sbatch.sh
    python experiments/igm330_retrain/worker_sweep.py summary

Writes experiments/igm330_retrain/tables/worker_sweep/<glacier>_<workers>.tsv;
summary prints and writes tables/worker_sweep.tsv.
"""

import glob
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.getcwd())
import frost.glacier_model.igm_wrapper as W  # noqa: E402

GLACIERS = {  # small, medium, large (igm330_live: 3.6, 11.3, 14.8 min)
    'Silvretta': 'RGI2000-v7.0-G-11-01949',
    'Rhone': 'RGI2000-v7.0-G-11-01706',
    'Aletsch': 'RGI2000-v7.0-G-11-02596',
}
WORKERS = [4, 8, 12, 16, 24, 32]
MEMBERS = 32
PASSES = 3
YEARS = (2000, 2020)
SRC = 'data/results/igm330_live'
TARGET = 'data/results/igm330_worker_sweep'
TABLES = 'experiments/igm330_retrain/tables/worker_sweep'


class Monitor(threading.Thread):
    """Samples GPU memory/utilisation and the workers' host memory."""

    def __init__(self, executor):
        super().__init__(daemon=True)
        self.executor, self.stop, self.samples = executor, threading.Event(), []

    def host_mib(self):
        total = 0
        for p in list(getattr(self.executor, '_processes', {}) or {}):
            try:
                with open(f'/proc/{p}/status') as f:
                    total += next(int(l.split()[1]) for l in f if l.startswith('VmRSS'))
            except (OSError, StopIteration):
                pass
        return total / 1024

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10).stdout.split('\n')[0]
                mem, util = (float(x) for x in out.split(','))
            except Exception:
                mem = util = np.nan
            self.samples.append((mem, util, self.host_mib()))
            self.stop.wait(1.0)

    def summary(self):
        s = np.array(self.samples) if self.samples else np.full((1, 3), np.nan)
        return dict(gpu_mem_peak_mib=np.nanmax(s[:, 0]), gpu_util_mean=np.nanmean(s[:, 1]),
                    host_workers_peak_mib=np.nanmax(s[:, 2]))


def cgroup_mib(name='memory.peak'):
    """memory.peak (or memory.max: limit) of this job's cgroup (cgroup v2), NaN if not readable."""
    try:
        with open('/proc/self/cgroup') as f:
            path = f.read().strip().split('\n')[0].split(':', 2)[2]
        for d in [path, os.path.dirname(path), os.path.dirname(os.path.dirname(path))]:
            p = os.path.join('/sys/fs/cgroup' + d, name)
            if os.path.exists(p):
                v = open(p).read().strip()
                if v != 'max':
                    return int(v) / 2**20
    except Exception:
        pass
    return np.nan


def members(name):
    rgi = GLACIERS[name]
    with open(os.path.join(SRC, rgi, 'calibration_results.json')) as f:
        d = json.load(f)
    mean, std = np.array(d['final_mean']), np.array(d['final_std'])
    rng = np.random.default_rng(1)
    draws = mean + std * rng.standard_normal((MEMBERS, len(mean)))
    draws[:, 1:] = np.maximum(draws[:, 1:], 0.1)  # gradients positive
    return [{'ela': float(e), 'abl_grad': float(a), 'acc_grad': float(c)}
            for e, a, c in draws]


def sweep(name, workers):
    rgi = GLACIERS[name]
    root = os.path.abspath(os.path.join(TARGET, f'{name}_{workers}'))
    shutil.rmtree(root, ignore_errors=True)
    src_input = os.path.join(SRC, rgi, 'Ensemble', 'Member_0', 'data', 'input.nc')
    with Dataset(src_input) as ds:
        usurf = np.array(ds['usurf'][:])
    for i in range(MEMBERS):
        os.makedirs(os.path.join(root, f'Member_{i}', 'experiment'))
        os.makedirs(os.path.join(root, f'Member_{i}', 'data'))
        shutil.copy2(src_input, os.path.join(root, f'Member_{i}', 'data', 'input.nc'))
    emulator = os.path.abspath(os.path.join(SRC, rgi, 'Preprocess', 'outputs', 'emulator.keras'))
    smb = members(name)
    cores = len(os.sched_getaffinity(0))
    print(f'SWEEP {name} {workers} workers, {cores} cores, live IGM {W.LIVE_IGM}', flush=True)

    rows = []
    executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                   initializer=W.warm_up)
    try:
        for p in range(1, PASSES + 1):
            mon = Monitor(executor)
            mon.start()
            t0, ok = time.time(), True
            try:
                futures = [executor.submit(
                    W.forward, 'Calibration', False, True, i, 'ELA', usurf, smb[i],
                    *YEARS, os.path.join(root, f'Member_{i}'), None, emulator, 1000,
                    True, 'igm_default', None) for i in range(MEMBERS)]
                for f in as_completed(futures):
                    f.result()
            except Exception as e:  # worker killed, out of memory, ...
                ok = False
                print(f'PASS {p} FAILED: {type(e).__name__}: {e}', flush=True)
            dt = time.time() - t0
            mon.stop.set()
            mon.join()
            row = dict(glacier=name, workers=workers, cores=cores, pass_=p, ok=ok,
                       seconds=dt, **mon.summary(), job_peak_mib=cgroup_mib(),
                       job_limit_mib=cgroup_mib('memory.max'))
            rows.append(row)
            print('PASS ' + ' '.join(f'{k}={v:.1f}' if isinstance(v, float) else f'{k}={v}'
                                     for k, v in row.items()), flush=True)
            if not ok:
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, f'{name}_{workers}.tsv'), 'w') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for r in rows:
            f.write('\t'.join(f'{v:.2f}' if isinstance(v, float) else str(v)
                              for v in r.values()) + '\n')
    shutil.rmtree(root, ignore_errors=True)  # 32 member folders per case


def summary():
    import pandas as pd
    files = sorted(glob.glob(os.path.join(TABLES, '*.tsv')))
    if not files:
        print('no results in', TABLES)
        return
    t = pd.concat([pd.read_csv(f, sep='\t') for f in files])
    warm = t[t.pass_ > 1]
    s = warm.groupby(['glacier', 'workers']).agg(
        ok=('ok', 'all'), warm_pass_s=('seconds', 'mean'),
        gpu_mem_peak_gib=('gpu_mem_peak_mib', lambda x: x.max() / 1024),
        gpu_util=('gpu_util_mean', 'mean'),
        host_workers_gib=('host_workers_peak_mib', lambda x: x.max() / 1024),
        job_peak_gib=('job_peak_mib', lambda x: x.max() / 1024),
        job_limit_gib=('job_limit_mib', lambda x: x.max() / 1024)).reset_index()
    first = t[t.pass_ == 1].set_index(['glacier', 'workers'])['seconds']
    s['first_pass_s'] = [first.get((g, w), np.nan) for g, w in zip(s.glacier, s.workers)]
    # failed cases (no warm pass) still listed
    for (g, w), grp in t.groupby(['glacier', 'workers']):
        if not ((s.glacier == g) & (s.workers == w)).any():
            s.loc[len(s)] = dict(glacier=g, workers=w, ok=False,
                                 first_pass_s=grp.seconds.iloc[0])
    s = s.sort_values(['glacier', 'workers'])
    # speed-up of the warm pass against 8 workers (the current setting)
    ref = s[s.workers == 8].set_index('glacier')['warm_pass_s']
    s['vs_8'] = s.glacier.map(ref) / s.warm_pass_s
    pd.set_option('display.width', 200)
    print(s.round(2).to_string(index=False))
    s.round(3).to_csv(os.path.join(os.path.dirname(TABLES), 'worker_sweep.tsv'),
                      sep='\t', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary()
    else:
        sweep(sys.argv[1], int(sys.argv[2]))
