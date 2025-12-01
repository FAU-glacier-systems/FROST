import copy
import glob
import os
import frost.glacier_model.igm_wrapper as igm_wrapper
from select_SMB_member import select_SMB_member
from extract_CMIP_climate_historical import extract_and_save
from extract_CMIP_representative import extract_repo
from compile_output_ts import compile_output_ts
from netCDF4 import Dataset
import numpy as np
import json

# settings
exec_script = "frost_pipeline.py"
list_fname = "todo_codeskel.txt"
PATH_exp_config = "alps_TI_projections/eALPS"
PATH_results = "../../data/results/alps_TI_projections/glaciers"
FNAME_exp_config = "config_TI"

countit = 0


def prepare_forward(rgi_id, member_id, start_year, end_year, workdir):
    # Projection run setting

    rgi_id_dir = os.path.join('..', '..', 'data', 'results', 'alps_TI_projections', 'glaciers', rgi_id)

    member_dir = workdir

    # Load initial surface elevation

    inversion_dir = os.path.join(rgi_id_dir, 'Preprocess', 'outputs')
    geology_file = os.path.join(inversion_dir, 'output.nc')
    with Dataset(geology_file, 'r') as geology_dataset:
        usurf_init = np.array(geology_dataset['usurf'])

    # Copy input files
    # Copy geology file as the initial input.nc
    os.makedirs(os.path.join(member_dir, 'data'), exist_ok=True)
    if start_year == 2000:
        shutil.copy2(geology_file, os.path.join(member_dir, 'data', 'input.nc'))
    if start_year == 2020:
        if str(member_id) == "mean":
            shutil.copy2(
                os.path.join(member_dir, '..', 'past', 'outputs', 'output_W5E5_' + str(member_id) + '.nc'),
                os.path.join(member_dir, 'data', 'input.nc'))
        else:
            shutil.copy2(os.path.join(member_dir, '..', 'past', 'outputs', 'output_W5E5_best.nc'),
                         os.path.join(member_dir, 'data', 'input.nc'))

    # Copy iceflow-model directory
    member_iceflow_dir = os.path.join(member_dir, 'iceflow-model')
    shutil.rmtree(member_iceflow_dir, ignore_errors=True)
    shutil.copytree(os.path.join(inversion_dir, 'iceflow-model'), member_iceflow_dir)

    # Create experiment folder
    os.makedirs(os.path.join(member_dir, 'experiment'), exist_ok=True)

    # Load glacier-specific parameters
    # calibration version
    # cal_version = 'v01'
    # params_file_path = os.path.join('./Experiments',rgi_id,'Ensemble_'+cal_version+'_'+rgi_id+'_50.0_6_6_50m',
    #                                    'result.json')
    params_file_path = JSON_cal_path

    with open(params_file_path, 'r') as file:
        params = json.load(file)
        final_mean = params['final_mean']
        final_std = params['final_std']

        final_ensemble = params['final_ensemble']

    if str(member_id) == 'mean':
        # Create SMB dictionary
        new_smb = {
            'temp_bias': final_mean[2],
            'melt_f': final_mean[0],
            'prcp_fac': final_mean[1]
        }
    else:
        new_smb = {
            'temp_bias': final_ensemble[int(member_id)][2],
            'melt_f': final_ensemble[int(member_id)][0],
            'prcp_fac': final_ensemble[int(member_id)][1]
        }

    # Saving JSON file
    SMB = {
        "member_id": member_id,
        "temp_bias": new_smb["temp_bias"],
        "melt_f": new_smb["melt_f"],
        "prcp_fac": new_smb["prcp_fac"]
    }

    # Write to file
    outputJSON = os.path.join(rgi_id_dir, "selectedSMBmember.json")
    with open(outputJSON, "w") as f:
        json.dump(SMB, f, indent=4)

    print('IGM_wrapper.forward')
    print('member_id', member_id)
    print('rgi_id_dir', rgi_id_dir)
    print('usurf', np.shape(usurf_init))
    print('smb', new_smb)
    print('start/end year :', start_year, end_year)
    return copy.copy(usurf_init), copy.copy(new_smb)


def run_single_member(workdir, rgi_id, m3D, member_id, IGM_flag, experiment, usurf_init, new_smb, mm, threeD):
    if IGM_flag != 1:
        return None

    print(f"Starting projection of: {rgi_id}")

    start_year = 2020
    end_year = 2100

    climate_file = os.path.join("..", "..", "..", "climate_historical.nc")
    output1D = True

    if start_year == 2000:
        output2D_3D = True
    else:
        output2D_3D = False

    if threeD:
        output2D_3D = True

    # Define SMB model
    smb_model = 'TI'

    member_id = igm_wrapper.forward(experiment, output1D, output2D_3D, member_id, smb_model,
                                    usurf_init, new_smb, start_year, end_year, workdir, climate_file)

    return workdir


with open(list_fname, "r") as f:
    for line in f:
        rgi_id = line.strip()  # remove newline, spaces

        # do whatever you want with rgi_id
        print(f"Processing RGI ID: {rgi_id}")

        # variables assumed to be defined earlier:
        # PATH_results, rgi_id

        # set calibration directory
        cal_path = os.path.join(PATH_results, rgi_id)
        JSON_cal_path = os.path.join(cal_path, "calibration_results.json")

        # ensemble output directory
        OUT_cal_dir = os.path.join(PATH_results, rgi_id, "Ensemble")

        # retrieve ensemble size (count Member_* folders)
        member_dirs = glob.glob(os.path.join(OUT_cal_dir, "Member_*"))
        EnKF_ensemble_size = len(member_dirs)

        print("ensemble size", EnKF_ensemble_size)

        # select SMB member
        SMBoption = "best"  # "best" or "mean"

        # calibration results
        cal_fpath = os.path.join(PATH_results, rgi_id)
        cal_fname = "calibration_results.json"

        # observations
        obs_fpath = os.path.join(PATH_results, rgi_id)
        obs_fname = "observations.nc"

        # model output directory (Ensemble directory from earlier)
        mod_fpath = OUT_cal_dir
        mod_fname = "output.nc"

        member_id = select_SMB_member(
            cal_fpath=cal_fpath,
            cal_fname=cal_fname,
            mod_fpath=mod_fpath,
            mod_fname=mod_fname,
            obs_fpath=obs_fpath,
            obs_fname=obs_fname,
            option=SMBoption
        )

        import os
        import shutil
        import subprocess

        # TIME_PERIOD: 2000–2020
        print(f"Simulating reference period for: {rgi_id} {member_id}")
        start_year = 2000
        end_year = 2020

        # Paths for climate forcing files
        in_fpath = cal_path
        in_fname = "climate_historical_W5E5.nc"

        out_fpath = cal_path
        out_fname = "climate_historical.nc"

        # Full paths
        in_file = os.path.join(in_fpath, in_fname)
        out_file = os.path.join(out_fpath, out_fname)

        # Copy W5E5 climate forcing
        if os.path.isfile(in_file):
            # in_file exists → copy in_file → out_file
            shutil.copy(in_file, out_file)
        else:
            # fallback: copy out_file → in_file
            shutil.copy(out_file, in_file)

        flag_3D_output = True
        print("Start projections")

        # Run IGM model for past
        workdir = os.path.join(out_fpath, "Projection", f"Member_{member_id}", "past")
        climate_file = os.path.join("../", "../", "../", "../", "../", "../", "climate_historical.nc")

        experiment = "past"
        usurf_init, new_smb = prepare_forward(rgi_id, member_id, start_year, end_year, workdir)

        igm_wrapper.forward(experiment, True, flag_3D_output, member_id, "TI", usurf_init, new_smb,
                            start_year, end_year, workdir, climate_file)

        # Base directories
        proj_member_out = os.path.join(
            PATH_results, rgi_id, "Projection", f"Member_{member_id}", "outputs"
        )

        proj_member = os.path.join(
            PATH_results, rgi_id, "Projection", f"Member_{member_id}"
        )

        proj_root = os.path.join(PATH_results, rgi_id, "Projection")

        # --- Copy output.nc → output_W5E5_<SMBoption>.nc ---

        src_output_nc = os.path.join(workdir, "outputs", "output.nc")
        dst_output_nc = os.path.join(workdir, "outputs", f"output_W5E5_{SMBoption}.nc")

        shutil.copy(src_output_nc, dst_output_nc)

        # --- Copy output_ts.nc → Projection directory ---

        src_output_ts = os.path.join(workdir, "outputs", "output_ts.nc")
        dst_output_ts_root = os.path.join(proj_root, f"output_ts_W5E5_{SMBoption}.nc")

        shutil.copy(src_output_ts, dst_output_ts_root)

        # Also copy output_ts.nc → outputs/ directory with new name
        dst_output_ts_local = os.path.join(workdir, "outputs", f"output_ts_W5E5_{SMBoption}.nc")

        shutil.copy(src_output_ts, dst_output_ts_local)

        for climate_scenario in [1, 2, 3]:

            # Build paths
            in_fpath = f"/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/{rgi_id[-5:]}/climate/CORDEX_unbiased"
            # in_fpath = f"/home/oskar/Desktop/HPC_fragile/input/11/1/glacier_ids/{rgi_id[-5:]}/climate/CORDEX_unbiased"
            in_fname = "CORDEX_merged.nc"

            print(
                f"python alps_TI_projections/extract_CMIP_representative.py --in_fpath {in_fpath} --in_fname {in_fname} --experiment {climate_scenario}")

            #
            m3D, experiment3D, member3D = extract_repo(
                in_fpath=in_fpath,
                in_fname=in_fname,
                experiment=climate_scenario
            )

            rgi_id_dir = f"../../data/results/alps_TI_projections/glaciers/{rgi_id}/"
            base_workdir = os.path.join(rgi_id_dir, "Projection")  # choose your base folder
            os.makedirs(base_workdir, exist_ok=True)

            args_list = []

            for mm in range(70):
                workdir = os.path.join(base_workdir, f"Member_{member_id}", f"worker_{mm:03d}")
                os.makedirs(workdir, exist_ok=True)
                print(f"mm: {mm}")

                out_fpath = workdir
                out_fname = "climate_historical.nc"
                in_fpath = f"/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/{rgi_id[-5:]}/climate/CORDEX_unbiased"
                # in_fpath = f"/home/oskar/Desktop/HPC_fragile/input/11/1/glacier_ids/{rgi_id[-5:]}/climate/CORDEX_unbiased"
                in_fname = "CORDEX_merged.nc"

                IGM_flag, experiment, member = extract_and_save(
                    in_fpath, in_fname, out_fpath, out_fname, climate_scenario, mm, rgi_id_dir
                )

                if IGM_flag == 1:
                    threeD_flag = mm == m3D
                    usurf_init, new_smb = prepare_forward(rgi_id, member_id, start_year, end_year, workdir)
                    args_list.append(
                        (workdir, rgi_id, m3D, member_id, IGM_flag, experiment, usurf_init, new_smb, mm, threeD_flag))

            # number of parallel workers
            N = os.cpu_count()  # or use os.cpu_count()

            from concurrent.futures import ProcessPoolExecutor, as_completed

            workers_with_igm_output = []
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_single_member, *args) for args in args_list]

                for future in as_completed(futures):
                    workers_with_igm_output.append(future.result())

            bpr_fpath = f"/home/vault/gwgi/gwgifu1h/data/input/11/1/glacier_ids/{rgi_id[-5:]}/climate/CORDEX_unbiased"
            bpr_fname = "CORDEX_merged.nc"

            # for workdir in workers_with_igm_output:
            for args in args_list:
                workdir, rgi_id, m3D, member_id, IGM_flag, experiment, usurf_init, new_smb, mm, threeD_flag = args
                if IGM_flag == 1:
                    # input paths
                    in_fpath = os.path.join(workdir, "outputs")
                    in_fname = "output_ts.nc"

                    # output paths
                    out_fpath = f"{PATH_results}/{rgi_id}/Projection/"
                    out_fname = f"CORDEX_{SMBoption}_output_ts.nc"

                    compile_output_ts(countit, bpr_fpath, bpr_fname, in_fpath, in_fname,
                                      out_fpath, out_fname, climate_scenario, mm)
                    countit += 1

                    if threeD_flag == 1:
                        src = os.path.join(in_fpath, "output.nc")
                        dst = os.path.join(out_fpath, f"output_{experiment}_{member}.nc")
                        shutil.copy(src, dst)
                        os.remove(src)
