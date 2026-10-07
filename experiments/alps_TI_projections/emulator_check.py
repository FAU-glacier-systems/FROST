#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

"""
Is the iceflow emulator right on the inverted glaciers? For the inverted state
of a glacier (thickness fixed), the surface velocity from
  A the frozen emulator saved by the inversion (what the forward runs use),
  C the pretrained dahunet_mini fine-tuned on this state, and
  B the direct solver (identity mapping, L-BFGS), the reference,
compared with each other and with the observations. Results in
data/results/alps_TI_projections/emulator_check/<rgi_id>/.

Run from the repository root (GPU job, see emulator_check_sbatch.sh):
    python experiments/alps_TI_projections/emulator_check.py <rgi_id> [nbit_solver]
"""
import os, sys, subprocess, shutil, yaml, numpy as np
from netCDF4 import Dataset
rgi = sys.argv[1]; d = os.path.abspath(f'data/results/alps_TI_projections/{rgi}')
base = os.path.abspath(f'data/results/alps_TI_projections/emulator_check/{rgi}')
cases = {
 'A_frozen_saved': dict(mapping='network', nbit_init=0, network=dict(pretrained=True, pretrained_path=f'{d}/Preprocess/outputs/emulator.keras')),
 'C_finetuned_1000': dict(mapping='network', nbit_init=1000, optimizer='adam', network=dict(pretrained=True, pretrained_path='dahunet_mini.keras')),
 'B_solver_lbfgs': dict(mapping='identity', nbit_init=int(sys.argv[2]) if len(sys.argv) > 2 else 3000, optimizer='lbfgs', line_search='hager-zhang'),
}
res = {}
for name, u in cases.items():
    w = os.path.join(base, name); shutil.rmtree(w, ignore_errors=True)
    os.makedirs(f'{w}/experiment'); os.makedirs(f'{w}/data')
    shutil.copy(f'{d}/Preprocess/outputs/output.nc', f'{w}/data/input.nc')
    unified = dict(mapping=u['mapping'], inputs=['thk', 'usurf', 'arrhenius', 'tau_ref', 'dX'], nbit_init=u['nbit_init'], nbit=0, retrain_freq=0)
    if 'optimizer' in u: unified['optimizer'] = u['optimizer']
    if 'line_search' in u: unified['line_search'] = u['line_search']
    if 'network' in u: unified['network'] = u['network']
    p = {'hydra': {'run': {'dir': 'outputs/run'}}, 'core': {'url_data': ''},
         'defaults': [{'override /inputs': ['local']}, {'override /processes': ['iceflow', 'time']}, {'override /outputs': ['write_ncdf']}],
         'inputs': {'local': {'filename': 'input.nc'}},
         'processes': {'iceflow': {'method': 'unified', 'physics': {'sliding': {'tau_ref': 0.213, 'u_ref': 100.0}},
                                   'numerics': {'Nz': 2, 'basis_horizontal': 'q1', 'basis_vertical': 'molho'}, 'unified': unified},
                       'time': {'start': 2000.0, 'end': 2000.5, 'save': 1.0}},
         'outputs': {'write_ncdf': {'output_file': '../velocity.nc', 'vars_to_save': ['thk', 'velsurf_mag', 'velbar_mag']}}}
    with open(f'{w}/experiment/params.yaml', 'w') as f:
        f.write('# @package _global_\n'); yaml.dump(p, f)
    r = subprocess.run(['igm_run', '+experiment=params'], cwd=w, capture_output=True, text=True)
    if r.returncode or not os.path.exists(f'{w}/outputs/velocity.nc'):
        print(name, 'FAILED'); print(r.stderr[-1500:]); continue
    with Dataset(f'{w}/outputs/velocity.nc') as ds: res[name] = np.array(ds['velsurf_mag'][0])
with Dataset(f'{d}/Preprocess/outputs/output.nc') as ds:
    m = np.array(ds['icemask']) > 0.5; obs = np.array(ds['velsurfobs_mag']); thk = np.array(ds['thk'])
ok = m & np.isfinite(obs) & (obs > 0)
print(f'{rgi}: ice px {m.sum()}, mean thk {thk[m].mean():.0f} m, obs vel median {np.median(obs[ok]):.1f} m/yr')
ref = res.get('B_solver_lbfgs')
for k, v in res.items():
    s = f'  {k:18s} median {np.median(v[ok]):6.1f}  mean {v[ok].mean():6.1f} m/yr  ratio to obs {np.median(v[ok]) / np.median(obs[ok]):5.2f}'
    if ref is not None and k != 'B_solver_lbfgs':
        s += f'  | vs solver: median ratio {np.median(v[ok]) / np.median(ref[ok]):5.2f}, RMS diff {np.sqrt(np.mean((v[ok] - ref[ok])**2)):6.1f}'
    print(s)
