#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import shutil
import json
import yaml
import os
import subprocess
from netCDF4 import Dataset
import numpy as np
from frost.glacier_model.no_flow import SNOW_SPINUP_YEARS

# Suppress warnings and optimize TensorFlow execution

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = info, 2 = warning, 3 = error

FROST_USER_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'igm_user')


def install_user_processes(workdir, processes):
    """Copy FROST's IGM user processes into <workdir>/user, where igm_run
    picks them up, instead of into the installed igm package (which is
    shared by all jobs using the environment)."""
    for process in processes:
        shutil.copytree(
            os.path.join(FROST_USER_DIR, 'code', 'processes', process),
            os.path.join(workdir, 'user', 'code', 'processes', process),
            dirs_exist_ok=True)
        os.makedirs(os.path.join(workdir, 'user', 'conf', 'processes'),
                    exist_ok=True)
        shutil.copy(
            os.path.join(FROST_USER_DIR, 'conf', 'processes', process + '.yaml'),
            os.path.join(workdir, 'user', 'conf', 'processes'))


def run_igm(workdir, in_process=False):
    """igm_run +experiment=params in workdir. in_process: call IGM's main()
    in this process instead of starting igm_run, so TensorFlow, CUDA and
    IGM are loaded once per process, not once per run (G02: 18.9 s per
    igm_run, 2.6 s per call after the first). For worker processes that run
    one member at a time: changes the working directory while it runs."""
    if not in_process:
        subprocess.run(["igm_run", "+experiment=params"], cwd=workdir)
        return
    import sys
    from igm.igm_run import main as igm_main
    cwd, argv = os.getcwd(), sys.argv
    try:
        os.chdir(workdir)
        sys.argv = ["igm_run", "+experiment=params"]
        igm_main()
    except SystemExit as e:
        # hydra exits on errors; a failed member must not leave the previous
        # round's output.nc to be read as its result
        if e.code:
            raise RuntimeError(f"igm_run failed in {workdir}") from e
    finally:
        os.chdir(cwd)
        sys.argv = argv


def warm_up():
    """Pool initializer: import IGM (and TensorFlow) once per worker."""
    import igm.igm_run  # noqa: F401


# IGM time step: min(CFL * dx / fastest speed, STEP_MAX) (IGM time defaults)
CFL, STEP_MAX = 0.3, 1.0
# longest time between two retrainings of the emulator in the forward runs
MAX_RETRAIN_INTERVAL = 0.9  # yr
# retraining iterations at the start of a forward run (IGM default); on
# Corbassiere they leave the speed of 2000 low (8.9 m/yr, solver 12.4),
# from the first step on it follows the solver (emulator_test)
RETRAIN_START = 300


def retrain_schedule(inversion_output):
    """(retrain_freq, nbit) of the forward runs: IGM's 5 iterations every 5
    time steps where 5 steps take less than MAX_RETRAIN_INTERVAL (fast
    glaciers, short steps; every-step retraining costs 4-6x there), else 20
    every step: on slow glaciers with long steps the speed jumped at each
    rare retraining (Corbassiere, experiments/emulator_test). The step is
    estimated with IGM's CFL rule from the fastest surface speed on the ice
    of the inversion."""
    with Dataset(inversion_output) as ds:
        dx = abs(float(ds['x'][1] - ds['x'][0]))
        ice = np.array(ds['thk'][:]) > 1
        fastest = float(np.nanmax(np.array(ds['velsurf_mag'][:])[ice]))
    step = min(CFL * dx / max(fastest, 1e-6), STEP_MAX)
    return (5, 5) if 5 * step < MAX_RETRAIN_INTERVAL else (1, 20)


def forward(exp, output1D, output2D_3D, member_id, smb_model, usurf, smb,
            year_start, year_end, workdir, climate_file,
            emulator_path="dahunet_mini.keras", max_velbar=0.0,
            in_process=False, emulator_retrain=False, snow_ratio=None):
    '''
    Runs a single forward model simulation for an ensemble member.

    Authors: Oskar Herrmann, Johannes J. Fürst

    Args:
        exp(str)              - intended experiment ('Calibration', 'Projection', ...)
        output1D(bool)        - if TRUE: standard time series output from IGM is provided
        output2D_3D(bool)     - if TRUE: 2D/3D output from IGM is provided
        member_id (int)       - ID of the ensemble member
        rgi_dir (str)         - Path to the glacier directory
        smb_model(str)        - chosen SMB model (ELA, TI, ...)
        usurf (ndarray)       - Initial surface elevation array
        smb (dict)            - Surface mass balance parameters:
                                  * 'ela' (float)      - Equilibrium line altitude
                                  * 'gradabl' (float)  - Ablation gradient (per km)
                                  * 'gradacc' (float)  - Accumulation gradient (per km)
        start_year (int)      - Year that simulation starts
        end_year (int)        - Year that simulaiton ends
        emulator_path (str)   - Iceflow network; pass the one saved by the
                                inversion (igm_inversion.emulator_path) to
                                avoid an initialisation shock
        max_velbar (float)    - Cap of the depth-averaged speed (m/yr), 0 for
                                none; keeps single cells at a retreating ice
                                cliff from forcing tiny time steps
        in_process (bool)     - run IGM in this process (run_igm)
        emulator_retrain (tuple) - (retrain_freq, nbit) of retrain_schedule:
                                retrain the emulator on the glacier's
                                physics (RETRAIN_START iterations at the
                                start, then nbit every retrain_freq steps) instead
                                of keeping it frozen (False); removes the
                                start-up speed shock on geometries the
                                pretrained network extrapolates
                                (emulator_test)
        snow_ratio (float)    - TI: melt factor of snow and firn over that of
                                ice (no_flow.ti_month); None: one melt
                                factor. smb['temp_gradient'] (K/km), if
                                there, replaces the -6.5 K/km lapse rate

    Returns:
        member_id (int)         - Ensemble member ID
        new_usurf (ndarray)     - Updated surface elevation
        new_smb (ndarray)       - Updated SMB values
    '''

    # Extract SMB parameters and convert gradients from m/km to m/m
    if str(smb_model) == 'ELA':
        # Scalars give a constant SMB; sequences give one value per year
        # from year_start to year_end
        ela = np.asarray(smb['ela'])
        abl_grad = np.asarray(smb['abl_grad']) / 1000
        acc_grad = np.asarray(smb['acc_grad']) / 1000
    elif str(smb_model) == 'TI':
        melt_f = smb['melt_f']
        prcp_fac = smb['prcp_fac']
        temp_bias = smb['temp_bias']

    print(f'{exp}')
    print(f'Forward ensemble member {member_id} with SMB: {smb}')

    # Define input parameters for the ice flow model (IGM)
    igm_params = {
        # Fixed run dir so reruns overwrite instead of creating
        # outputs/<date>/<time>. The output paths below ('../output.nc') and
        # clim_1D_3D's climate file are relative to this depth.
        "hydra": {
            "run": {
                "dir": "outputs/forward",
            }
        },
        "core": {
            "url_data": "",
        },
        "inputs": {
            "local": {
                "filename": "input.nc"
            }
        },
        "processes": {
            "iceflow": {
                # Frozen emulator, as igm-examples aletsch
                # common/iceflow_offline.yaml. tau_ref comes from input.nc.
                "method": "unified",
                "force_max_velbar": float(max_velbar),
                "numerics": {
                    "Nz": 2,
                    "basis_horizontal": "q1",
                    "basis_vertical": "molho",
                },
                "unified": {
                    "mapping": "network",
                    "inputs": ["thk", "usurf", "arrhenius", "tau_ref", "dX"],
                    "nbit_init": RETRAIN_START if emulator_retrain else 0,
                    "nbit": emulator_retrain[1] if emulator_retrain else 0,
                    "retrain_freq": emulator_retrain[0] if emulator_retrain else 0,
                    "network": {
                        "pretrained_path": str(emulator_path),
                        "pretrained": True,
                    }
                },
            },
            "time": {
                "start": year_start,
                "end": year_end,
                "save": 1.,
            },
        }
    }

    if str(smb_model) == 'TI':
        install_user_processes(workdir, ['clim_1D_3D', 'smb_1D_3D'])

        igm_params['defaults'] = [{"override /inputs": ["local"]}]
        igm_params['defaults'] += [{'override /processes': ["clim_1D_3D", "smb_1D_3D", "iceflow", "time", "thk"]}]

        igm_params['processes']["clim_1D_3D"] = {
            "clim_trend_array": []
        }
        igm_params['processes']['clim_1D_3D']['clim_trend_array'] = [
            ['time', 'delta_temp', 'prec_scal'],
            [1900, 0.0, 1.0],
            [2020, 0.0, 1.0],
        ]
        igm_params['processes']['clim_1D_3D']['ref_period'] = [2000, 2019]
        igm_params['processes']['clim_1D_3D']['temp_bias'] = temp_bias
        igm_params['processes']['clim_1D_3D']['prcp_fac'] = prcp_fac
        igm_params['processes']['clim_1D_3D'][
            'prcp_gradient'] = 0.00035  # https://hess.copernicus.org/articles/24/5355/2020/
        igm_params['processes']['clim_1D_3D']['temp_default_gradient'] = \
            float(smb.get('temp_gradient', -6.5)) / 1000
        igm_params['processes']['clim_1D_3D']['update_freq'] = 1
        igm_params['processes']['clim_1D_3D']['climate_file'] = os.path.abspath(climate_file)
        # IGM >= 3.2 does not merge user/conf defaults, so set every key
        igm_params['processes']['clim_1D_3D']['seed_par'] = 123

        igm_params['processes']["smb_1D_3D"] = {
            "temp_all_solid": 0.0
        }

        igm_params['processes']['smb_1D_3D']['temp_all_liquid'] = 2.0
        igm_params['processes']['smb_1D_3D']['temp_melt'] = -1.0
        igm_params['processes']['smb_1D_3D']['melt_f'] = melt_f
        igm_params['processes']['smb_1D_3D']['wat_density'] = 1000.0
        igm_params['processes']['smb_1D_3D']['ice_density'] = 910.0
        igm_params['processes']['smb_1D_3D']['melt_enhancer'] = 1
        igm_params['processes']['smb_1D_3D']['update_freq'] = 1
        igm_params['processes']['smb_1D_3D']['snow_ratio'] = float(snow_ratio or 0.0)
        igm_params['processes']['smb_1D_3D']['snow_spinup'] = SNOW_SPINUP_YEARS

    elif str(smb_model) == 'ELA':

        igm_params['defaults'] = [{"override /inputs": ["local"]}]
        igm_params['defaults'] += [{'override /processes': ["smb", "iceflow", "time", "thk"]}]

        # smb_simple became the `smb` umbrella with method: simple (IGM >= 3.2)
        igm_params['processes']["smb"] = {
            "method": "simple",
            "simple": {"array": []},
        }
        years = np.arange(year_start, year_end + 1)
        abl_grads, acc_grads, elas = (np.broadcast_to(v, years.shape)
                                      for v in (abl_grad, acc_grad, ela))
        igm_params['processes']['smb']['simple']['array'] = [
            ['time', 'gradabl', 'gradacc', 'ela', 'accmax']] + [
            [int(y), float(ab), float(ac), float(e), 100]
            for y, ab, ac, e in zip(years, abl_grads, acc_grads, elas)]

    if output1D and output2D_3D:
        igm_params['defaults'] += [{'override /outputs': ["write_ts", "write_ncdf"]}]
    elif output1D:
        igm_params['defaults'] += [{'override /outputs': ["write_ts"]}]
    elif output2D_3D:
        igm_params['defaults'] += [{'override /outputs': ["write_ncdf"]}]

    if output1D:
        igm_params['outputs'] = {
            "write_ts": {
                "output_file": []
            }
        }
        igm_params['outputs']['write_ts']['output_file'] = '../output_ts.nc'
    if output2D_3D:
        igm_params['outputs'] = {
            "write_ncdf": {
                "output_file": [],
                "vars_to_save": [],
            }
        }
        if output1D:
            igm_params['outputs']['write_ts'] = {
                "output_file": []
            }
            igm_params['outputs']['write_ts']['output_file'] = '../output_ts.nc'

        igm_params['outputs']['write_ncdf']['output_file'] = '../output.nc'

        if str(smb_model) == 'TI':
            igm_params['outputs']['write_ncdf']['vars_to_save'] = ['topg',
                                                                   'usurf',
                                                                   'thk',
                                                                   'smb',
                                                                   'velbar_mag',
                                                                   'velsurf_mag',
                                                                   'slidingco',
                                                                   'divflux',
                                                                   'mean_temp',
                                                                   'sum_prec',
                                                                   'icemask',
                                                                   'icemask_init'
                                                                   ]
        else:
            igm_params['outputs']['write_ncdf']['vars_to_save'] = ['topg',
                                                                   'usurf',
                                                                   'thk',
                                                                   'smb',
                                                                   'velbar_mag',
                                                                   'velsurf_mag',
                                                                   'divflux',
                                                                   'slidingco',
                                                                   'icemask',
                                                                   'icemask_init'
                                                                   ]

    # Save simulation parameters as JSON

    def sanitize(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # convert np.float32 → float
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        return obj

    params_path = os.path.join(workdir, 'experiment', 'params.yaml')
    if os.path.exists(params_path):
        os.remove(params_path)
    with open(params_path, 'w') as file:
        file.write('# @package _global_\n')
        yaml.dump(sanitize(igm_params), file)

    # Load input NetCDF file and update elevation values
    print("start igm run here", workdir)
    input_file = os.path.join(workdir, 'data', 'input.nc')
    with Dataset(input_file, 'r+') as input_dataset:
        if not "icemask_init" in input_dataset.variables:
            bedrock = input_dataset.variables['topg'][:]  # Read bedrock elevation
            thickness = usurf - bedrock  # Compute ice thickness

            # Update surface elevation and thickness
            input_dataset.variables['usurf'][:] = usurf
            input_dataset.variables['thk'][:] = thickness

            # Create init icemask
            var_in = input_dataset.variables['icemask']

            var_out = input_dataset.createVariable("icemask_init", var_in.datatype, ("y", "x",))
            var_out.setncatts({k: var_in.getncattr(k) for k in var_in.ncattrs()})
            var_out[:] = input_dataset.variables['icemask'][:]

    # Run the Iceflow Glacier Model (IGM)
    run_igm(workdir, in_process)

    if str(exp) == 'Calibration':
        # Read updated results from the output file
        output_file = os.path.join(workdir, 'outputs', 'output.nc')
        with Dataset(output_file, 'r') as new_ds:
            new_usurf = np.array(new_ds['usurf'][-1])  # Final surface elevation
            init_usurf = np.array(new_ds['usurf'][0])
            new_velsurf_mag = np.array(new_ds['velsurf_mag'][0])
            # SMB and flux divergence averaged over the same yearly outputs
            # after the start (the first is the initial state), so that
            # dh/dt = smb - divflux over the period
            new_smb = np.mean(np.array(new_ds['smb'][1:]), axis=0)
            new_divflux = np.mean(np.array(new_ds['divflux'][1:]), axis=0)

        return member_id, new_usurf, new_smb, init_usurf, new_velsurf_mag, new_divflux
    else:
        return

