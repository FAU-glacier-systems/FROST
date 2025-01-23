import argparse
from Scripts.EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.ObservationHandler import ObservationProvider
from Scripts.Visualization.Monitor import Monitor
import os


def main(rgi_id, ensemble_size, inflation, iterations, seed, num_bins,
         forward_parallel):
    print(f'Running calibration for glacier: {rgi_id}')
    print(f'Ensemble size: {ensemble_size}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Forward parallel: {forward_parallel}')
    output_dir = os.path.join('Experiments', rgi_id, 'Bin_sensitivity',
                              f'Experiment_{seed}_{num_bins}')

    # Initialise the Observation provider
    ObsProvider = ObservationProvider(rgi_id=rgi_id,
                                      num_bins=int(num_bins))

    year, usurf_ensemble =  ObsProvider.inital_usurf_ensemble(
        num_samples=ensemble_size)

    # Initialise an ensemble kalman filter object
    ensemble_kalman_filter = EnsembleKalmanFilter(rgi_id=rgi_id,
                                                  ensemble_size=ensemble_size,
                                                  inflation=inflation,
                                                  seed=seed,
                                                  start_year=year,
                                                  usurf_ensemble=usurf_ensemble,
                                                  output_dir=output_dir)



    # Initialise a monitor for visualising the process
    monitor = Monitor(EnKF_object=ensemble_kalman_filter, ObsProvider=ObsProvider,
                      output_dir=output_dir)

    ################# MAIN LOOP #####################################################
    for i in range(1, iterations + 1):
        # get new observation

        year, new_observation, noise_matrix, noise_samples \
            = ObsProvider.get_next_observation(
            ensemble_kalman_filter.current_year,
            ensemble_kalman_filter.ensemble_size)
        print(noise_matrix)
        print(f'Forward pass ensemble to {year}')
        ensemble_kalman_filter.forward(year=year, forward_parallel=forward_parallel)
        ensemble_observables = ObsProvider.get_observables_from_ensemble(
            ensemble_kalman_filter)

        print("Update")
        ensemble_kalman_filter.update(new_observation, noise_matrix, noise_samples,
                                      ensemble_observables)

        print("Visualise")
        monitor.plot_iteration(
            ensemble_smb_log=ensemble_kalman_filter.ensemble_smb_log,
            ensemble_smb_raster=ensemble_kalman_filter.ensemble_smb_raster,
            new_observation=new_observation,
            uncertainty=noise_matrix,
            iteration=i,
            year=year,
            ensemble_observables=ensemble_observables)
        ensemble_kalman_filter.reset_time()
    #################################################################################

    ensemble_kalman_filter.save_results(num_bins)
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

    parser.add_argument('--iterations', type=int, default=5,
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
