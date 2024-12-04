import json
import os
import subprocess
from netCDF4 import Dataset
import numpy as np
import time
import threading

os.environ['PYTHONWARNINGS'] = "ignore"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'


def forward(member_id, rgi_dir, usurf, smb, year_interval):
    # Define the logic for a single forward computation
    ela = smb['ela']
    grad_abl = smb['gradabl']/1000
    grad_acc = smb['gradacc']/1000
    print("SMB: "+str(smb))
    igm_params = {"modules_preproc": ["load_ncdf"],
                  "modules_process": ["smb_simple", "iceflow", "time",
                                      "thk"],
                  "modules_postproc": ["write_ncdf", "print_info"],
                  "smb_simple_array": [
                      ["time", "gradabl", "gradacc", "ela", "accmax"],
                      [0, grad_abl, grad_acc, ela, 2],
                      [year_interval, grad_abl, grad_acc, ela, 2]],
                  "iflo_emulator": "iceflow-model",
                  "lncd_input_file": f'input.nc',
                  "wncd_output_file": f'output.nc',
                  "time_start": 0,
                  "time_end": year_interval,
                  "iflo_retrain_emulator_freq": 0,
                  # "time_step_max": 0.2,
                  }
    member_dir = os.path.join(rgi_dir, 'Ensemble', f'Member_{member_id}')
    with open(os.path.join(member_dir, "params.json"), 'w') as file:
        json.dump(igm_params, file, indent=4, separators=(',', ': '))

    input_file = os.path.join(member_dir, "input.nc")

    with Dataset(input_file, 'r+') as input_dataset:
        # Open in read/write mode
        # Update 'usurf'
        input_dataset.variables['usurf'][:] = usurf

        # Update 'thk' based on 'topg'
        bedrock = input_dataset.variables['topg'][:]
        thickness = usurf - bedrock

        input_dataset.variables['thk'][:] = thickness

    ################################## RUN IGM ######################################
    subprocess.run(["igm_run"], cwd=member_dir,)
    #################################################################################

    output_file = os.path.join(member_dir, "output.nc")


    with Dataset(output_file, 'r') as new_ds:
        new_usurf = np.array(new_ds['usurf'][-1])
        new_smb = np.array(new_ds['smb'][-1])

    return new_usurf, new_smb
