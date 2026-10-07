#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Assemble the ContinuIX upload folder GROUP_<group>/ from the submission
files of the task lists: EXP##/EXP##_G##_method01.nc (with the method
description attribute), log_GROUP_<group>.txt (computing times from
timings.json), README_GROUP_<group>_method01.txt, the filled checklist and
the ContinuIX instruction files.

Run from the repository root:
    python experiments/continuix/package_submission.py
"""

import argparse
import json
import os
import shutil

import yaml
from netCDF4 import Dataset

EXPERIMENT_DIR = 'experiments/continuix'
TASK_LISTS = ['tasks_exp01.txt', 'tasks_exp03-15.txt']
CONTINUIX_FILES = ['README_submission_template.txt',
                   'FILE_NAMING_INSTRUCTIONS.txt']
# 'raw' only for EXP02
STEPS = ['raw', 'prepare', 'inversion', 'calibrate', 'submit']

# 'description' attribute of every result file
DESCRIPTION = (
    'FROST (Herrmann et al., 2025, doi:10.1017/aog.2025.10020): IGM 3.2.0 '
    'with the provided thickness and the sliding inverted from the surface '
    'velocities; a three-parameter SMB model (ELA, ablation and '
    'accumulation gradient) calibrated with ES-MDA (36 members, 6 '
    'iterations) against the observed dh/dt in 50 m elevation bands. SMB: '
    'ensemble mean of the calibrated SMB model on the surface at the middle '
    'of the period; FDIV: time-mean flux divergence of the calibrated '
    'ensemble forward runs; UNCT_*: ensemble standard deviation. Ice '
    'equivalent with 910 kg m-3. See README_GROUP_FAU_method01.txt.')

LOG_HEADER = """\
ContinuIX computation log, GROUP_{group}, method01 (FROST)

Resources: Alex cluster of NHR@FAU (Erlangen). Each experiment and glacier
ran as one Slurm job on one NVIDIA A40 GPU (48 GB) with 16 CPU cores
(1/8 of a node: 2x AMD EPYC 7713, 8x A40, 512 GB RAM). The 36 ensemble
members of the forward runs share the GPU.
Software: IGM 3.2.0 (TensorFlow), FROST
(https://github.com/FAU-glacier-systems/FROST), Python 3.10.

Steps per experiment and glacier:
  raw        EXP02 only: GeoTIFFs and shapefiles -> netCDF on the EXP01 grid
  prepare    ContinuIX netCDF -> model grid (resampling, gap filling, ice mask)
  inversion  IGM inversion of the sliding (tau_ref) from the surface velocities
  calibrate  ES-MDA: 6 iterations with 36 IGM forward runs each over the
             dh/dt period, plus the forward runs of the calibrated ensemble
  submit     result file on the provided grid

Wall-clock times in seconds; dx: data / model grid spacing (m); area: ice
area on the model grid; period: dh/dt period (synthetic glaciers: 2000-2010).

"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--group', default='FAU')
    parser.add_argument('--continuix_repo', default='../ContinuIX-1',
                        help='clone of github.com/ContinuIX/ContinuIX-1')
    args = parser.parse_args()
    with open(os.path.join(EXPERIMENT_DIR, 'config.yml')) as f:
        cfg = yaml.safe_load(f)
    group = f'GROUP_{args.group}'
    results_dir = cfg['results_dir']
    out_dir = os.path.join(results_dir, group)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    tasks = []
    for task_list in TASK_LISTS:
        with open(os.path.join(EXPERIMENT_DIR, task_list)) as f:
            tasks += [tuple(line.split()) for line in f if line.strip()]

    # Result files
    rows = []
    for exp, glacier in tasks:
        name = f"{exp}_{glacier}_{cfg['method']}.nc"
        target = os.path.join(out_dir, exp, name)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(os.path.join(results_dir, 'submission', exp, name), target)
        with Dataset(target, 'a') as nc:
            nc.setncattr('description', DESCRIPTION)

        rgi_id_dir = os.path.join(results_dir, exp, glacier)
        with open(os.path.join(rgi_id_dir, 'timings.json')) as f:
            timings = json.load(f)
        with open(os.path.join(rgi_id_dir, 'continuix.json')) as f:
            meta = json.load(f)
        rows.append((exp, glacier, meta, timings))

    # Computation log
    lines = [LOG_HEADER.format(group=args.group)]
    lines.append(f"{'exp':6}{'glacier':8}{'dx':>10}{'area km2':>10}"
                 f"{'period':>11}" + ''.join(f'{s:>11}' for s in STEPS)
                 + f"{'total':>8}  job / node")
    total = 0.0
    for exp, glacier, meta, timings in rows:
        seconds = [timings.get(s, 0.0) for s in STEPS]
        total += sum(seconds)
        dx = f"{meta['resolution_data']:g}/{meta['resolution_model']:g}"
        lines.append(
            f"{exp:6}{glacier:8}{dx:>10}{meta['ice_area_km2']:>10.1f}"
            f"{meta['year_start']:>6}-{meta['year_end']}"
            + ''.join(f'{t:>11.0f}' for t in seconds)
            + f"{sum(seconds):>8.0f}  {timings['job']} / {timings['node']}")
    lines.append('')
    lines.append(f'{len(rows)} runs, {total / 3600:.1f} GPU hours in total.')
    per_glacier = {}
    for _, glacier, _, timings in rows:
        per_glacier.setdefault(glacier, []).append(
            sum(timings.get(s, 0.0) for s in STEPS))
    lines.append('Mean wall-clock time per run (min): ' + ', '.join(
        f'{g} {sum(t) / len(t) / 60:.0f}'
        for g, t in sorted(per_glacier.items())) + '.')
    lines.append('Not included: development and tuning runs (L-curve of the '
                 'sliding regularisation, earlier calibration variants).')
    with open(os.path.join(out_dir, f'log_{group}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # README, checklist, ContinuIX instructions
    submission_docs = os.path.join(EXPERIMENT_DIR, 'submission')
    for name in os.listdir(submission_docs):
        shutil.copy(os.path.join(submission_docs, name), out_dir)
    for name in CONTINUIX_FILES:
        shutil.copy(os.path.join(args.continuix_repo, 'Files', name), out_dir)
    print(f'Wrote {out_dir}: {len(rows)} result files, '
          f'{total / 3600:.1f} GPU hours')


if __name__ == '__main__':
    main()
