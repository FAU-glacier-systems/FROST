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


def reset_igm_tf_functions():
    """Replace every module-level tf.function of the loaded IGM modules by a
    fresh one (same Python function and options), as in a new process.

    IGM >= 3.3.0 fails on its second run in one process: compute_node_ice_mask
    (thk/masks.py, jit), traced inside the jit evaluator_iceflow of the first
    run, keeps references to that run's variables, and the next run's
    evaluator (retraced for the new network) calls the stale trace:
    "AssertionError: Called a function referencing variables which have been
    deleted" (experiments/igm330_retrain, 2026-11-16). Fresh functions lose
    nothing that was reused: the evaluator retraces for every run anyway.
    Returns the number of functions replaced."""
    import sys
    import tensorflow as tf
    function_type = getattr(tf.types.experimental, 'PolymorphicFunction', None) or \
        tf.types.experimental.GenericFunction
    fresh = {}  # id -> (old, new); the old one stays referenced while swapping
    modules = [m for name, m in list(sys.modules.items())
               if m is not None and (name == 'igm' or name.startswith('igm.'))]
    for mod in modules:
        for obj in list(vars(mod).values()):
            if (isinstance(obj, function_type)
                    and id(obj) not in fresh and hasattr(obj, 'python_function')):
                fresh[id(obj)] = obj, tf.function(
                    obj.python_function,
                    input_signature=getattr(obj, 'input_signature', None),
                    autograph=getattr(obj, '_autograph', True),
                    jit_compile=getattr(obj, '_jit_compile', None),
                    reduce_retracing=getattr(obj, '_reduce_retracing', False))
    # also where a module imported the function by name
    for mod in modules:
        for attr, obj in list(vars(mod).items()):
            if id(obj) in fresh and fresh[id(obj)][0] is obj:
                setattr(mod, attr, fresh[id(obj)][1])
    return len(fresh)


# In-process runs reuse one live IGM model per process (LiveIGM): ~5x faster
# per member, bit-identical to fresh runs (experiments/igm330_retrain,
# 2026-11-17: Rhone 5.5 vs 27.5 s). FROST_LIVE_IGM=0 switches it off.
LIVE_IGM = os.environ.get('FROST_LIVE_IGM', '1') != '0'


class LiveIGM:
    """Runs IGM members one after the other in one process, keeping IGM's
    compiled ice-flow objects, without IGM source changes.

    The first member of a glacier is a plain in-process igm_run; its module
    lists, cfg and state are captured, and the network weights right before
    the initial training. Every next member with the same setup (same
    ensemble folder, same params.yaml apart from SMB and time) gets a new
    State, initialised as igm_run does it, except iceflow: the kept mapping,
    optimizer (traced minimize), evaluator parameters and discretisation are
    put back, the weights reset to the captured ones, the Adam moments
    zeroed, then IGM's own initial solve (solve_iceflow(init=True)) and
    evaluation, and IGM's update/finalize loop. cfg = the first cfg merged
    with the member's params.yaml. Another setup starts again with a plain
    run. Nothing is retraced, so no compiled graphs pile up either."""

    def __init__(self):
        self.base = None  # dict: key, cfg, modules, state, w0

    @staticmethod
    def _key(workdir, params):
        """What the kept objects depend on: the glacier (ensemble folder,
        hence the grid), the module lists and the iceflow, input, output and
        core settings. SMB/climate and time settings (member parameters) are
        read again by every member's initialize."""
        kept = {k: params.get(k) for k in ('defaults', 'inputs', 'outputs', 'core')}
        kept['iceflow'] = params.get('processes', {}).get('iceflow')
        return (os.path.dirname(os.path.abspath(workdir)),
                json.dumps(kept, sort_keys=True, default=str))

    def run(self, workdir):
        with open(os.path.join(workdir, 'experiment', 'params.yaml')) as f:
            params = yaml.safe_load(f)
        key = self._key(workdir, params)
        if self.base is None or self.base['key'] != key:
            self.base = None  # free the old glacier's model before the new one
            self._capture_run(workdir, key)
            return
        try:
            self._live_run(workdir, params)
        except Exception:
            self.base = None  # next member starts again with a plain run
            raise

    def _capture_run(self, workdir, key):
        import igm.igm_run as R
        import igm.processes.iceflow.unified.unified as U
        cap = {}
        orig = R.setup_igm_modules, R.finalize_modules, U.solve_iceflow

        def setup(cfg, state):
            cap['modules'] = orig[0](cfg, state)
            return cap['modules']

        def final(modules, cfg, state):
            cap['cfg'], cap['state'] = cfg, state
            return orig[1](modules, cfg, state)

        def solve(cfg, state, init=False):
            if init and 'w0' not in cap:
                cap['w0'] = [v.numpy() for v in state.iceflow_model.variables]
            return orig[2](cfg, state, init=init)

        R.setup_igm_modules, R.finalize_modules, U.solve_iceflow = setup, final, solve
        try:
            _run_igm_main(workdir)
        finally:
            R.setup_igm_modules, R.finalize_modules, U.solve_iceflow = orig
        if {'modules', 'cfg', 'state', 'w0'} <= set(cap):
            self.base = dict(cap, key=key)

    def _reset_iceflow(self, cfg, state):
        from igm.processes.iceflow.utils.fields import initialize_iceflow_fields
        from igm.processes.iceflow.unified.solver import solve_iceflow
        from igm.processes.iceflow.unified.evaluator import evaluate_iceflow
        state0 = self.base['state']
        state.iceflow_initialized = True
        state.iceflow = state0.iceflow
        state.iceflow_model = state0.iceflow_model
        state.vert_weight = state0.vert_weight
        initialize_iceflow_fields(cfg, state)
        for v, w in zip(state.iceflow_model.variables, self.base['w0']):
            v.assign(w)
        adam = getattr(state.iceflow.optimizer, 'optim_adam', None)
        if adam is not None:
            for v in adam.variables:
                if 'learning_rate' not in v.name:
                    v.assign(np.zeros(v.shape, v.dtype.as_numpy_dtype
                                      if hasattr(v.dtype, 'as_numpy_dtype') else v.dtype))
        solve_iceflow(cfg, state, init=True)
        evaluate_iceflow(cfg, state)

    def _live_run(self, workdir, params):
        from datetime import datetime
        from pathlib import Path
        from omegaconf import OmegaConf
        from igm.common import State, update_modules, finalize_modules
        params = {k: v for k, v in params.items() if k not in ('hydra', 'defaults')}
        cfg = OmegaConf.merge(self.base['cfg'], OmegaConf.create(params))
        inputs, processes, assimilations, outputs = self.base['modules']
        workdir = os.path.abspath(workdir)
        rundir = os.path.join(workdir, 'outputs', 'forward')
        os.makedirs(rundir, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(rundir)
        try:
            state = State()
            state.original_cwd = Path(workdir)
            state.saveresult = True
            state.start_time = datetime.now()
            for m in inputs:
                m.run(cfg, state)
            for m in outputs:
                m.initialize(cfg, state)
            combined = assimilations + processes
            for m in combined:
                if m.__name__.split('.')[-1] == 'iceflow':
                    self._reset_iceflow(cfg, state)
                else:
                    m.initialize(cfg, state)
            update_modules(combined, outputs, cfg, state)
            finalize_modules(combined, cfg, state)
        finally:
            os.chdir(cwd)


_LIVE = LiveIGM()


def run_igm(workdir, in_process=False):
    """igm_run +experiment=params in workdir. in_process: run IGM in this
    process instead of starting igm_run, so TensorFlow, CUDA and IGM are
    loaded once per process, not once per run (G02: 18.9 s per igm_run,
    2.6 s per call after the first), and with LIVE_IGM the compiled ice-flow
    model is kept between members of a glacier (LiveIGM). For worker
    processes that run one member at a time: changes the working directory
    while it runs."""
    if not in_process:
        subprocess.run(["igm_run", "+experiment=params"], cwd=workdir)
        return
    if LIVE_IGM:
        _LIVE.run(workdir)
    else:
        _run_igm_main(workdir)


def _run_igm_main(workdir):
    """IGM's main() in this process, in workdir."""
    import sys
    from igm.igm_run import main as igm_main
    cwd, argv = os.getcwd(), sys.argv
    reset_igm_tf_functions()
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


# retraining iterations at the start of a forward run (IGM default); on
# Corbassiere they leave the speed of 2000 low (8.9 m/yr, solver 12.4),
# from the first step on it follows the solver (emulator_test)
RETRAIN_START = 300
# When the emulator is retrained after the start (EnKF emulator_retrain),
# with plain IGM >= 3.3.0 options (no IGM fork). The igm32 fork's rule of
# 2026-11-12 (20 every year of model time, unified.retrain_interval) is
# dropped with the update to IGM 3.3.0 (experiments/igm330_retrain):
#   igm_default  5 iterations every 5 time steps (IGM default; on slow
#                glaciers with ~1-yr steps that is every ~5 years)
#   credit       5 iterations whenever the inputs (thk, usurf, ...) have
#                changed by kappa (normalised, accumulated over steps),
#                at the latest every 100 steps (IGM 3.3.0 adaptive_time)
RETRAIN_RULES = {
    'igm_default': {'nbit': 5, 'retrain_freq': 5,
                    'adaptive_time': {'method': 'none'}},
    'credit': {'nbit': 5, 'retrain_freq': 5,
               'adaptive_time': {'method': 'credit'}},
}
# emulator_retrain: true picks this rule. credit (2026-11-20): the same
# calibration as igm_default (experiments/igm330_retrain) and the same speeds
# (17 flow glaciers: forward / ITS_LIVE 1.04 against 1.09), but on slow
# glaciers with long time steps igm_default lowers the speed in steps (every
# 5 steps), credit smoothly (experiments/igm330_TI/README.md); igm_default
# was faster (254 against 307 min for the 17 glaciers, before the live IGM
# model)
DEFAULT_RETRAIN_RULE = 'credit'


def retrain_rule(emulator_retrain):
    """Name of the retraining rule for EnKF emulator_retrain (False/None:
    frozen emulator, True: DEFAULT_RETRAIN_RULE, or a key of RETRAIN_RULES)."""
    if emulator_retrain in (None, False, 'false', 'none'):
        return None
    rule = DEFAULT_RETRAIN_RULE if emulator_retrain is True else str(emulator_retrain)
    if rule not in RETRAIN_RULES:
        raise ValueError(f"emulator_retrain '{emulator_retrain}': use true, false or one "
                         f"of {list(RETRAIN_RULES)}")
    return rule


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
        emulator_retrain (bool or str) - retrain the emulator on the
                                glacier's physics (RETRAIN_START iterations
                                at the start, then by a rule of
                                RETRAIN_RULES; True: DEFAULT_RETRAIN_RULE)
                                instead of keeping it frozen (False); removes the
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

    rule = retrain_rule(emulator_retrain)
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
                    "nbit_init": RETRAIN_START if rule else 0,
                    "nbit": 0,
                    "retrain_freq": 0,
                    **(RETRAIN_RULES[rule] if rule else {}),
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

