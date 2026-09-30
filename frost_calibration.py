#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import os.path

import numpy as np
import yaml

from frost.calibration.ensemble_kalman_filter import EnsembleKalmanFilter
from frost.calibration.observation_provider import ObservationProvider
from frost.visualization.monitor import Monitor


def main(rgi_id, rgi_id_dir, smb_model, synthetic, ensemble_size, inflation,
         smb_prior_mean, smb_prior_std,
         iterations, seed, init_offset, elev_band_height, forward_parallel,
         synth_obs_std=None, smb_reference_mean=None, smb_reference_std=None,
         dark_monitor=True, monitor_plots='latest', method='esmda'):
    """
    main function to run the calibration, handles the interaction between
    observation, ensemble and visualization. It saves the results in the experiment
    folder as a json file that contain the final ensemble, its mean and standard
    deviation.

    Authors: Oskar Herrmann, Johannes J. Fürst

    Args:
           rgi_id(str)           - glacier ID
           smb_model(str)        - chosen SMB model (ELA, TI, ...)
           synthetic(bool)       - switch to synthetic observations
           synth_obs_std(float)- factor of synthetic observation uncertainty
           ensemble_size(int)    - ensemble size
           inflation(float)      - inflation factor
           iterations(int)       - number of iterations
           seed(int)             - random seed
           elev_band_height(int)   - elevation band height
           forward_parallel(int) - forward parallel
           dark_monitor(bool)    - Monitor plots on black (True) or white
           monitor_plots(str)    - 'all': one png per iteration; 'latest':
                                   one png, overwritten every iteration;
                                   'final': only the last (posterior) run
           method(str)           - 'esmda' (default): ES-MDA (Emerick &
                                   Reynolds 2013), observation error x
                                   iterations in every iteration, so the
                                   data count once in total;
                                   'enkf': every iteration assimilates the
                                   observations with their full error
                                   (posterior too narrow by ~sqrt(iterations))

    Returns:
           none
    """

    if not os.path.exists(rgi_id_dir):
        os.makedirs(rgi_id_dir, exist_ok=True)

    print(f'Running calibration for glacier: {rgi_id}')
    print(f'SMB model: {smb_model}')
    print(f'Ensemble size: {ensemble_size}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Elevation step: {elev_band_height}',
          f'Observation uncertainty: {synth_obs_std}',
          f'Synthetic: {synthetic}',
          f'Initial offset: {init_offset}',
          f'Results directory: {rgi_id_dir}',
          f'Forward parallel: {forward_parallel}')

    # Initialise the Observation provider
    print("Initializing Observation Provider")
    obs_provider = ObservationProvider(rgi_id=rgi_id,
                                       rgi_id_dir=rgi_id_dir,
                                       elevation_step=int(elev_band_height),
                                       obs_uncertainty=synth_obs_std,
                                       synthetic=synthetic)
    print("Initializing start surface")
    year, usurf_ensemble = obs_provider.initial_usurf(num_samples=ensemble_size)

    # Initialise an ensemble kalman filter object
    print("Initializing Ensemble Kalman Filter")
    ensembleKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                      rgi_id_dir=rgi_id_dir,
                                      smb_model=smb_model,
                                      ensemble_size=ensemble_size,
                                      inflation=inflation,
                                      seed=seed,
                                      start_year=year,
                                      usurf_ensemble=usurf_ensemble,
                                      init_offset=init_offset,
                                      smb_prior_mean=smb_prior_mean,
                                      smb_prior_std=smb_prior_std,
                                      smb_reference_mean=smb_reference_mean,
                                      smb_reference_std=smb_reference_std,
                                      obs_provider=obs_provider)

    # Initialise a monitor for visualising the process
    monitor = Monitor(EnKF_object=ensembleKF,
                      ObsProvider=obs_provider,
                      output_dir=rgi_id_dir,
                      max_iterations=iterations + 1,
                      synthetic=synthetic,
                      dark=dark_monitor,
                      plots=monitor_plots)

    ################# MAIN LOOP #####################################################
    if method not in ('enkf', 'esmda'):
        raise ValueError(f'Unknown calibration method {method}')
    rng = np.random.default_rng(seed)
    # ES-MDA: every iteration assimilates the data with its error inflated by
    # alpha = iterations (sum of 1/alpha = 1)
    alpha = iterations if method == 'esmda' else 1.0
    diagnostics = []

    # iterations updates; the last pass only runs the posterior ensemble, so
    # the final parameters come with their own forward run
    for i in range(1, iterations + 2):
        posterior = i == iterations + 1
        # get new observation
        year, new_observation, noise_matrix, noise_samples, obs_dhdt_raster, obs_velsurf_mag_raster \
            = obs_provider.get_next_observation(
            current_year=ensembleKF.current_year,
            num_samples=ensembleKF.ensemble_size)
        if method == 'esmda':
            update_noise = rng.multivariate_normal(
                np.zeros_like(new_observation), alpha * noise_matrix,
                size=ensembleKF.ensemble_size)
        else:
            update_noise = noise_samples

        print(f'Forward pass ensemble to {year}')
        ensembleKF.forward(year=year, forward_parallel=forward_parallel)

        ensemble_observables = obs_provider.get_ensemble_observables(
            EnKF_object=ensembleKF)

        # Diagnostics of the ensemble just run
        parameters = ensembleKF.smb_array()
        misfit = normalised_misfit(new_observation, ensemble_observables,
                                   noise_matrix)
        record = {'iteration': i, 'posterior': posterior, 'misfit': misfit,
                  'parameter_mean': parameters.mean(0).tolist(),
                  'parameter_std': parameters.std(0).tolist()}
        if diagnostics:
            previous = diagnostics[-1]
            record['parameter_change'] = (
                np.abs(parameters.mean(0) - previous['parameter_mean'])
                / np.maximum(parameters.std(0), 1e-12)).tolist()
        diagnostics.append(record)
        print(f"{'Posterior' if posterior else f'Iteration {i}'}: "
              f"misfit {misfit:.3f}")

        if not posterior:
            print("Update")
            ensembleKF.update(new_observation=new_observation,
                              noise_matrix=noise_matrix,
                              noise_samples=update_noise,
                              modeled_observables=ensemble_observables,
                              obs_error_factor=alpha)

        print("Visualise")
        monitor.plot_iteration(
            ensemble_smb_log=ensembleKF.ensemble_smb_log,
            new_observation=new_observation,
            uncertainty=noise_matrix,
            iteration=i,
            year=year,
            ensemble_observables=ensemble_observables,
            noise_samples=noise_samples)

        monitor.plot_maps_prognostic(ensembleKF,
                                     obs_dhdt_raster,
                                     obs_velsurf_mag_raster,
                                     new_observation,
                                     noise_samples,
                                     ensemble_observables,
                                     uncertainty=noise_matrix,
                                     iteration=i, year=year,
                                     write_json=posterior)

        ensembleKF.reset_time()

    #################################################################################

    ensembleKF.save_results(elevation_step=elev_band_height,
                            iterations=iterations,
                            obs_uncertainty=synth_obs_std,
                            synthetic=synthetic,
                            method=method,
                            diagnostics=diagnostics)

    print('Done')


def normalised_misfit(observation, ensemble_observables, noise_matrix):
    """chi^2 / n of the ensemble mean against the observation; about 1
    when the misfit matches the observation error."""
    residual = observation - np.mean(ensemble_observables, axis=0)
    return float(residual @ np.linalg.solve(noise_matrix, residual)
                 / len(residual))


if __name__ == '__main__':
    # Calibration step alone, with the settings of a pipeline config; the
    # download, inversion and observation outputs (create_observation) must
    # already be in the glacier folder
    parser = argparse.ArgumentParser(
        description='Run the EnKF calibration for one glacier.')
    parser.add_argument('--config', type=str,
                        default='experiments/test_default/pipeline_config.yml',
                        help='Path to the pipeline YAML config file')
    parser.add_argument('--rgi_id', type=str,
                        help='RGI ID to override the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    if args.rgi_id is not None:
        cfg['rgi_id'] = args.rgi_id

    main(rgi_id=cfg['rgi_id'],
         rgi_id_dir=os.path.join('data', 'results', cfg['experiment_name'],
                                 cfg['rgi_id']),
         smb_model=cfg['smb_model'],
         **cfg['EnKF'])
