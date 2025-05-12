#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
from Scripts.EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.ObservationProvider import ObservationProvider
from Scripts.Visualization.Monitor import Monitor
import os
import numpy as np


def main(rgi_id, synthetic, ensemble_size, inflation, iterations, seed, init_offset,
         elevation_step, forward_parallel, results_dir, obs_uncertainty):
    """
    main function to run the calibration, handles the interaction between
    observation, ensemble and visualization. It saves the results in the experiment
    folder as a json file that contain the final ensemble, its mean and standard
    deviation.

    Authors: Oskar Herrmann

    Args:
           rgi_id(str)           - glacier ID
           synthetic(bool)       - switch to synthetic observations
           obs_uncertainty(float)- factor of synthetic observation uncertainty
           ensemble_size(int)    - ensemble size
           inflation(float)      - inflation factor
           iterations(int)       - number of iterations
           seed(int)             - random seed
           elevation_step(int)   - elevation step
           forward_parallel(int) - forward parallel
           results_dir(str)      - results directory

    Returns:
           none
    """

    print(f'Running calibration for glacier: {rgi_id}')
    print(f'Ensemble size: {ensemble_size}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Elevation step: {elevation_step}',
          f'Observation uncertainty: {obs_uncertainty}',
          f'Synthetic: {synthetic}',
          f'Initial offset: {init_offset}'
          f'Results directory: {results_dir}'
          f'Forward parallel: {forward_parallel}')

    # Initialise the Observation provider
    print("Initializing Observation Provider")
    obs_provider = ObservationProvider(rgi_id=rgi_id,
                                       elevation_step=int(elevation_step),
                                       obs_uncertainty=obs_uncertainty,
                                       synthetic=synthetic)
    print("Initializing Usurf 2000")
    year, usurf_ensemble, binned_usurf, init_surf_bin = obs_provider.initial_usurf(
        num_samples=ensemble_size)

    # Initialise an ensemble kalman filter object
    print("Initializing Ensemble Kalman Filter")
    ensembleKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                      ensemble_size=ensemble_size,
                                      inflation=inflation,
                                      seed=seed,
                                      start_year=year,
                                      usurf_ensemble=usurf_ensemble,
                                      init_offset=init_offset,
                                      output_dir=results_dir)

    # Initialise a monitor for visualising the process
    monitor = Monitor(EnKF_object=ensembleKF,
                      ObsProvider=obs_provider,
                      output_dir=results_dir,
                      max_iterations=iterations,
                      synthetic=synthetic,
                      binned_usurf_init=binned_usurf,
                      plot_dhdt=False)

    ################# MAIN LOOP #####################################################
    for i in range(1, iterations + 1):
        # get new observation
        year, new_observation, noise_matrix, noise_samples, obs_dhdt_raster, obs_velsurf_mag_raster \
            = obs_provider.get_next_observation(
            current_year=ensembleKF.current_year,
            num_samples=ensembleKF.ensemble_size)

        print(f'Forward pass ensemble to {year}')
         #ensembleKF.forward(year=year, forward_parallel=forward_parallel)

        ensemble_observables = obs_provider.get_ensemble_observables(
            EnKF_object=ensembleKF)

        print("Update")
        ensembleKF.update(new_observation=new_observation,
                          noise_matrix=noise_matrix,
                          noise_samples=noise_samples,
                          modeled_observables=ensemble_observables)

        print("Visualise")
        monitor.plot_iteration(
            ensemble_smb_log=ensembleKF.ensemble_smb_log,
            new_observation=new_observation,
            uncertainty=noise_matrix,
            iteration=i,
            year=year,
            ensemble_observables=ensemble_observables,
            noise_samples=noise_samples)
        monitor.plot_maps_prognostic(ensembleKF.ensemble_usurf,
                                     ensembleKF.ensemble_smb_raster,
                                     ensembleKF.ensemble_init_surf_raster,
                                     ensembleKF.ensemble_velsurf_mag_raster,
                                     ensembleKF.ensemble_divflux_raster,
                                     obs_dhdt_raster, obs_velsurf_mag_raster,
                                     init_surf_bin,
                                     new_observation,
                                     ensemble_observables,
                                     uncertainty=noise_matrix,
                                     iteration=i, year=year,
                                     bedrock=ensembleKF.bedrock)

        # monitor.visualise_3d(obs_provider.ensemble_usurf[0],
        #                      ensembleKF.ensemble_usurf[0], ensembleKF.bedrock, 2000,
        #                      obs_provider.x, obs_provider.y)

        ensembleKF.reset_time()

    #################################################################################

    ensembleKF.save_results(elevation_step=elevation_step,
                            iterations=iterations,
                            obs_uncertainty=obs_uncertainty,
                            synthetic=synthetic)
    print('Done')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str,
                        default="RGI2000-v7.0-G-11-01706",
                        help='RGI ID of the glacier for the model.')

    parser.add_argument("--synthetic", type=str, default="false",
                        help="Change to synthetic observations.")

    parser.add_argument("--forward_parallel", type=str, default="false",
                        help="Enable forward parallel processing")

    parser.add_argument('--ensemble_size', type=int, default=64,
                        help='number of ensemble members for the model.')

    parser.add_argument('--inflation', type=float, default=1.0,
                        help='Inflation rate for the model.')

    parser.add_argument('--iterations', type=int, default=6,
                        help='Number of iterations')

    parser.add_argument('--elevation_step', type=int, default=50,
                        help='Elevation step for observations.')

    parser.add_argument("--obs_uncertainty", type=int, default="1",
                        help="Factor for the synthetic observation uncertainty")

    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for the model.')

    parser.add_argument('--init_offset', type=int, default=0,
                        help='Random seed for the model.')

    parser.add_argument('--results_dir', type=str, default='',
                        help='path to the results directory.', required=True)

    # Parse arguments
    args = parser.parse_args()
    forward_parallel = False if args.forward_parallel == "false" else True
    synthetic = False if args.synthetic == "false" or args.synthetic == "False" \
        else True

    # Call the main function with the parsed arguments
    main(rgi_id=args.rgi_id,
         synthetic=synthetic,
         ensemble_size=args.ensemble_size,
         inflation=args.inflation,
         iterations=args.iterations,
         seed=args.seed,
         forward_parallel=forward_parallel,
         elevation_step=args.elevation_step,
         obs_uncertainty=args.obs_uncertainty,
         results_dir=args.results_dir,
         init_offset=args.init_offset
         )
