import os.path
import shutil

import yaml
import argparse
import frost.preprocess.download_data as download_data
import frost.preprocess.igm_inversion as igm_inversion
import frost.preprocess.create_observation as create_observation
import frost_calibration

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hides all GPUs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Optional for JAX
os.environ[
    "XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"  # Makes JAX use only CPU


def run_frost_pipeline(cfg):
    print(f"Running FROST for RGI ID: {cfg['rgi_id']}")
    print(f"Steps: download={cfg['pipeline_steps']['download']}, "
          f"inversion={cfg['pipeline_steps']['inversion']}, "
          f"calibration={cfg['pipeline_steps']['calibrate']}")

    # create experiment folder
    experiment_path = os.path.join('data', 'results', cfg['experiment_name'])
    params_inversion_path = os.path.join('experiments', cfg['experiment_name'],
                                         'params_inversion.yaml')
    os.makedirs(experiment_path, exist_ok=True)
    rgi_id_dir = os.path.join(experiment_path, 'glaciers', cfg['rgi_id'])

    if cfg['pipeline_steps']['download']:
        #####################################
        #                                   #
        # STEP 1: SETUP GENERATION          #
        #         (input data, domain, ...) #
        #                                   #
        #####################################

        download_data.main(rgi_id=cfg['rgi_id'],
                           rgi_id_dir=rgi_id_dir,
                           smb_model=cfg['smb_model'],
                           target_resolution=cfg['download']['target_resolution'],
                           oggm_shop=cfg['download']['oggm_shop'],)

    if cfg['pipeline_steps']['inversion']:
        #####################################
        #                                   #
        # STEP 2: STATIONARY INVERSION      #
        #         (ice dynamics, ...)       #
        #                                   #
        #####################################
        igm_inversion.main(rgi_id_dir=rgi_id_dir,
                           params_inversion_path=params_inversion_path)


    if cfg['pipeline_steps']['calibrate']:
        create_observation.main(rgi_id_dir=rgi_id_dir,
                                year_interval=cfg['download']['year_interval'],
                                hugonnet_directory=cfg['download'][
                                    'hugonnet_directory'],
                                target_resolution=cfg['download'][
                                    'target_resolution'])
        #####################################
        #                                   #
        # STEP 3: TRANSIENT ASSIMILATION    #
        #         (ELAs, melt params., ...) #
        #                                   #
        #####################################
        frost_calibration.main(
            rgi_id=cfg['rgi_id'],
            rgi_id_dir=rgi_id_dir,
            smb_model=cfg['smb_model'],
            **cfg['EnKF']
        )

    print("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FROST pipeline with a config file")
    parser.add_argument("--config", type=str,
                        default='experiments/test_default/pipeline_config.yml',
                        help="Path to YAML config file")
    parser.add_argument("--rgi_id", type=str,
                        help="RGI ID to override config file")

    args = parser.parse_args()
    print(os.getcwd())
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.rgi_id is not None:
        config['rgi_id'] = args.rgi_id

    run_frost_pipeline(config)
