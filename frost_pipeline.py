import yaml
import subprocess
from pathlib import Path
import argparse
import frost.Preprocess.download_data as download_data
import frost.Preprocess.igm_inversion as igm_inversion
import frost_calibration

def run_frost_pipeline(cfg):
    print(f"Running FROST for RGI ID: {cfg['rgi_id']}")
    print(f"Steps: download={cfg['pipeline_steps']['download']}, "
          f"inversion={cfg['pipeline_steps']['inversion']}, "
          f"calibration={cfg['pipeline_steps']['calibrate']}")

    if cfg['pipeline_steps']['download']:
        #####################################
        #                                   #
        # STEP 1: SETUP GENERATION          #
        #         (input data, domain, ...) #
        #                                   #
        #####################################
        download_data.main(rgi_id=cfg['rgi_id'],
                           smb_model=cfg['smb_model'],
                           **cfg['download'], )
        # download_data.main(rgi_id=cfg['rgi_id'],
        #                    download_oggm_shop_flag=cfg['download']['oggm_shop'],
        #                    download_hugonnet_flag=cfg['download']['hugonnet'],
        #                    hugonnet_directory=cfg['download']['hugonnet_directory'],
        #                    year_interval=cfg['download']['year_interval'],
        #                    target_resolution=cfg['download']['target_resolution'],
        #                    )

    if cfg['pipeline_steps']['inversion']:
        #####################################
        #                                   #
        # STEP 2: STATIONARY INVERSION      #
        #         (ice dynamics, ...)       #
        #                                   #
        #####################################
        igm_inversion.main(rgi_id=cfg['rgi_id'])

    if cfg['pipeline_steps']['calibrate']:
        #####################################
        #                                   #
        # STEP 3: TRANSIENT ASSIMILATION    #
        #         (ELAs, melt params., ...) #
        #                                   #
        #####################################
        FROST_calibration.main(
            rgi_id=cfg['rgi_id'],
            ensemble_size=cfg['EnKF']['ensemble_size'],
            forward_parallel=cfg['EnKF']['forward_parallel'],
            iterations=cfg['EnKF']['iterations'],
            seed=cfg['EnKF']['seed'],
            inflation=cfg['EnKF']['inflation'],
            results_dir=cfg['experiment_name'],
            init_offset=cfg['EnKF']['init_offset'],
            elevation_step=cfg['EnKF']['elev_band_height'],
            synthetic=cfg['EnKF']['synthetic'],
            SMB_model=cfg['smb_model']
        )

    print("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FROST pipeline with a config file")
    parser.add_argument("--config", type=str, default='Experiments/Test/config.yml',
                        help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_frost_pipeline(config)
