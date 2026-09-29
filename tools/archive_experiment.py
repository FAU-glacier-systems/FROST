#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Collect the results of a published FROST experiment into one archive.

Keeps per glacier what the results and figures are based on (inputs,
observations, inversion result, calibration result, the configs actually
used and the final monitor plots) and leaves out what can be regenerated
(hydra date/time run folders except the inversion's metadata, inversion
iteration histories, individual ensemble-member runs, logs). Adds the
experiment folder (including the git-ignored tables and plots) and a
PROVENANCE.md.

Run from the repository root:
    python tools/archive_experiment.py central_europe_submit \\
        --experiment_dir experiments/central_europe --dry_run
    python tools/archive_experiment.py central_europe_submit \\
        --experiment_dir experiments/central_europe --dest $HPCVAULT
"""

import argparse
import glob
import os
import shutil
import subprocess
import tarfile
from datetime import datetime

# Per glacier, relative to data/results/<experiment>/glaciers/<rgi_id>
GLACIER_FILES = [
    'calibration_results.json',
    'observations.nc',
    'Preprocess/data/input.nc',
    'Preprocess/outputs/output.nc',
    'Preprocess/outputs/emulator.keras',
    'Preprocess/outputs/iceflow-model',
    'Preprocess/experiment',
    'Ensemble/Member_0/experiment',
]
# Metadata of the IGM inversion run (resolved config, cost history), from
# the latest run folder that contains optimize.nc; optimize.nc itself is
# left out
INVERSION_META = ['.hydra', 'costs.dat', 'rms_std_vol.dat', 'convergence.png']


def latest_inversion_run(outputs_dir):
    runs = [os.path.dirname(p) for p in
            glob.glob(os.path.join(outputs_dir, '*', '*', 'optimize.nc'))
            + glob.glob(os.path.join(outputs_dir, 'inversion', 'optimize.nc'))
            + glob.glob(os.path.join(outputs_dir, 'igm', 'inversion',
                                     'optimize.nc'))]
    return max(runs, key=os.path.getmtime) if runs else None


def final_monitor_plots(monitor_dir):
    """Monitor plots of the last EnKF iteration: of <name>_<iteration>_<year>.png
    the highest iteration; plots without an iteration (Monitor plots 'latest'
    or 'final') are the last ones already."""
    plots = sorted(glob.glob(os.path.join(monitor_dir, '*.png')))
    numbered = [p for p in plots
                if os.path.basename(p).rsplit('_', 2)[-2:][0].isdigit()]
    if not numbered:
        return plots
    last = max(p.rsplit('_', 2)[-2] for p in numbered)
    return [p for p in numbered if p.rsplit('_', 2)[-2] == last]


def selection(results_dir):
    """(source, path in archive) pairs for all glaciers."""
    pairs = []
    for glacier_dir in sorted(glob.glob(os.path.join(results_dir, 'glaciers',
                                                     '*'))):
        rgi_id = os.path.basename(glacier_dir)
        target = os.path.join('glaciers', rgi_id)
        for rel in GLACIER_FILES:
            src = os.path.join(glacier_dir, rel)
            if os.path.exists(src):
                pairs.append((src, os.path.join(target, rel)))
        run = latest_inversion_run(os.path.join(glacier_dir, 'Preprocess',
                                                'outputs'))
        if run is not None:
            for rel in INVERSION_META:
                src = os.path.join(run, rel)
                if os.path.exists(src):
                    pairs.append((src, os.path.join(
                        target, 'Preprocess', 'inversion_run', rel)))
        for src in final_monitor_plots(os.path.join(glacier_dir, 'Monitor')):
            pairs.append((src, os.path.join(target, 'Monitor',
                                            os.path.basename(src))))
    return pairs


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(root, f))
               for root, _, files in os.walk(path) for f in files)


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def provenance(experiment, results_dir, experiment_dir, n_glaciers):
    calibrations = glob.glob(os.path.join(results_dir, 'glaciers', '*',
                                          'calibration_results.json'))
    times = sorted(os.path.getmtime(p) for p in calibrations)
    first, last = (datetime.fromtimestamp(t).isoformat(timespec='minutes')
                   for t in (times[0], times[-1])) if times else ('?', '?')
    commit_before = run(['git', 'log', '-1', '--format=%h %ci %s',
                         f'--before={first}'])
    tags = run(['git', 'tag', '--sort=creatordate', '--format',
                '%(refname:short) %(creatordate:short)'])
    return f"""# {experiment}: archived FROST results

Archived {datetime.now():%Y-%m-%d} from `data/results/{experiment}`
({n_glaciers} glaciers, {len(calibrations)} with calibration results).

## Provenance

- Calibration results written: {first} .. {last}
- Last FROST commit before the run: {commit_before}
- FROST tags: {', '.join(tags.splitlines()) or 'none'}
- FROST commit at archiving: {run(['git', 'log', '-1', '--format=%h %ci'])}
- IGM: version not recorded by the run; the resolved IGM configs are in
  `glaciers/*/Preprocess/inversion_run/.hydra/config.yaml` and
  `glaciers/*/Preprocess/experiment/`
- Python environment at archiving: `environment.yml` (may differ from the
  environment used for the run)

## Contents

- `experiment/`: `{experiment_dir}` as on disk at archiving, including the
  git-ignored result tables and plots
- `glaciers/<rgi_id>/`
  - `calibration_results.json`: EnKF ensemble and final SMB parameters
  - `observations.nc`: observations used by the EnKF
  - `Preprocess/data/input.nc`: model input (OGGM shop)
  - `Preprocess/outputs/output.nc`: inversion result; `iceflow-model/`
    or `emulator.keras`: ice-flow network
  - `Preprocess/experiment/`, `Ensemble/Member_0/experiment/`: IGM configs
    of the inversion, download and forward runs
  - `Preprocess/inversion_run/`: resolved config and cost history of the
    inversion (optimize.nc left out)
  - `Monitor/`: plots of the last EnKF iteration

Left out (regenerable from code, configs and inputs): hydra date/time run
folders, inversion iteration histories (optimize.nc), individual ensemble
member runs, logs.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('experiment', help='Folder name in data/results')
    parser.add_argument('--experiment_dir', required=True,
                        help='Experiment config folder, e.g. '
                             'experiments/central_europe')
    parser.add_argument('--dest', help='Folder for the .tar archive')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only report contents and size')
    args = parser.parse_args()

    results_dir = os.path.join('data', 'results', args.experiment)
    pairs = selection(results_dir)
    pairs.append((args.experiment_dir, 'experiment'))
    n_glaciers = len(glob.glob(os.path.join(results_dir, 'glaciers', '*')))

    sizes = {}
    for src, dst in pairs:
        # summed over glaciers: path without glaciers/<rgi_id>/, plot and
        # inversion-run files grouped by folder
        key = dst.split(os.sep, 2)[-1]
        if key.startswith(('Monitor', 'Preprocess/inversion_run')):
            key = os.path.dirname(key) + '/'
        sizes[key] = sizes.get(key, 0) + size_of(src)
    print(f"{args.experiment}: {n_glaciers} glaciers")
    for key, size in sorted(sizes.items(), key=lambda kv: -kv[1]):
        print(f"  {size / 1e6:10.1f} MB  {key}")
    print(f"  {sum(sizes.values()) / 1e9:10.2f} GB  total")
    if args.dry_run:
        return

    name = f"{args.experiment}_{datetime.now():%Y%m%d}"
    archive = os.path.join(args.dest, f"{name}.tar")
    with tarfile.open(archive, 'w') as tar:
        text = provenance(args.experiment, results_dir, args.experiment_dir,
                          n_glaciers)
        env = run(['conda', 'env', 'export', '--no-builds'])
        for filename, content in [('PROVENANCE.md', text),
                                  ('environment.yml', env)]:
            path = os.path.join(args.dest, f'.{name}_{filename}')
            with open(path, 'w') as f:
                f.write(content)
            tar.add(path, arcname=os.path.join(name, filename))
            os.remove(path)
        for src, dst in pairs:
            tar.add(src, arcname=os.path.join(name, dst))
    print(f"Wrote {archive} ({os.path.getsize(archive) / 1e9:.2f} GB)")


if __name__ == '__main__':
    main()
