import argparse
from Scripts.EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.ObservationProvider import ObservationProvider
from Scripts.Visualization.Monitor import Monitor
import os
import numpy as np


def main(rgi_id, ensemble_size, inflation, iterations, seed, num_bins,
         forward_parallel):
    print(f'Running calibration for glacier: {rgi_id}')
    print(f'Ensemble size: {ensemble_size}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Forward parallel: {forward_parallel}')
    output_dir = os.path.join('Experiments', rgi_id, 'geometry_update',
                              f'Experiment_{ensemble_size}_{num_bins}_{inflation}_{seed}')

    # Initialise the Observation provider
    obs_provider = ObservationProvider(rgi_id=rgi_id,
                                       num_bins=int(num_bins))

    # Initialise an ensemble kalman filter object
    ensembleKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                      ensemble_size=ensemble_size,
                                      inflation=inflation,
                                      seed=seed,
                                      start_year=2000,
                                      output_dir=output_dir)

    # Initialise a monitor for visualising the process
    monitor = Monitor(EnKF_object=ensembleKF,
                      ObsProvider=obs_provider,
                      output_dir=output_dir,
                      max_iterations=iterations)

    ################# MAIN LOOP #####################################################
    total_iteration = 0
    while True:
        # get new observation
        year, new_observation, noise_matrix, noise_samples \
            = obs_provider.get_next_observation(
            current_year=ensembleKF.current_year,
            num_samples=ensembleKF.ensemble_size)
        if year is None: break

        start_year = ensembleKF.current_year
        start_usurf = obs_provider.inital_usurf(num_samples=ensemble_size,
                                                year=start_year)

        for i in range(1, iterations + 1):
            total_iteration += 1

            ensembleKF.reset_time(ensemble_usurf=start_usurf, year=start_year)

            print(
                f'Forward pass ensemble from {ensembleKF.current_year} to {year} iteration: {i}')
            ensembleKF.forward(year=year, forward_parallel=forward_parallel)

            ensemble_observables = obs_provider.get_ensemble_observables(
                EnKF_object=ensembleKF)

            np.random.shuffle(noise_samples)
            print("Update")
            ensembleKF.update(new_observation=new_observation,
                              noise_matrix=noise_matrix,
                              noise_samples=noise_samples,
                              modeled_observables=ensemble_observables)

            print("Visualise")
            monitor.plot_iteration(
                ensemble_smb_log=ensembleKF.ensemble_smb_log,
                ensemble_smb_raster=ensembleKF.ensemble_smb_raster,
                new_observation=new_observation,
                uncertainty=noise_matrix,
                iteration=total_iteration,
                year=year,
                ensemble_observables=ensemble_observables)

    #################################################################################

    ensembleKF.save_results(num_bins)
    print('Done')


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str, required=True,
                        help='RGI ID of the glacier for the model.')

    parser.add_argument('--ensemble_size', type=int, default=50,
                        help='number of ensemble members for the model.')

    parser.add_argument('--inflation', type=float, default=1.0,
                        help='Inflation rate for the model.')

    parser.add_argument('--iterations', type=int, default=6,
                        help='Number of iterations')

    parser.add_argument("--forward_parallel", type=str, default="false",
                        help="Enable forward parallel processing")

    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for the model.')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='Elevation bin for observations.')

    # Parse arguments
    args = parser.parse_args()
    if args.forward_parallel == "false":
        forward_parallel = False
    else:
        forward_parallel = True

    # Call the main function with the parsed arguments
    main(rgi_id=args.rgi_id,
         ensemble_size=args.ensemble_size,
         inflation=args.inflation,
         iterations=args.iterations,
         seed=args.seed,
         forward_parallel=forward_parallel,
         num_bins=args.num_bins)
