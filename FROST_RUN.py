import argparse
from Scripts.EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.ObservationHandler import ObservationProvider
from Scripts.Visualization.monitor import Monitor


def main(rgi_id, ensemble_size, covered_area, inflation, iterations, seed,
         forward_parallel):
    print(f'Running calibration for glacier: {rgi_id}')
    print(f'Ensemble size: {ensemble_size}',
          f'Covered area: {covered_area}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Forward parallel: {forward_parallel}')

    # TODO save params
    # Initialise an ensemble kalman filter object
    ENKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                ensemble_size=ensemble_size,
                                inflation=inflation,
                                seed=seed,
                                start_year=2000)

    # Initialise the Observation provider
    ObsProvider = ObservationProvider(rgi_id=rgi_id,
                                      covered_area=covered_area)

    # Initialise a monitor for visualising the process
    monitor = Monitor(EnKF_object=ENKF, ObsProvider=ObsProvider)

    ################# MAIN LOOP #####################################################
    for i in range(iterations):
        # get new observation
        year, new_observation, noise_matrix, noise_samples \
            = ObsProvider.get_next_observation(
            ENKF.current_year, ENKF.ensemble_size)

        while new_observation is not None:
            print(f'Forward pass ensemble to {year}')
            ENKF.forward(year=year, forward_parallel=forward_parallel)
            modeled_observables = ObsProvider.get_observables_from_ensemble(ENKF)

            print("Update")
            ENKF.update(new_observation, noise_matrix, noise_samples,
                        modeled_observables)

            # update geometries
            new_geometry = ObsProvider.get_new_geometrie()
            ENKF.update_geometries(new_geometry)

            print("Visualise")
            monitor.plot_status(ENKF, new_observation, i, year)

        ENKF.reset_time()
    #################################################################################

    ENKF.save_results()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str, required=True,
                        help='RGI ID of the glacier for the model.')

    parser.add_argument('--ensemble_size', type=int, default=50,
                        help='number of ensemble members for the model.')

    parser.add_argument('--covered_area', type=float, default=50,
                        help='Fraction of the area of the glacier that is covered'
                             'by the observations')

    parser.add_argument('--inflation', type=float, default=1.0,
                        help='Inflation rate for the model.')

    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of iterations')

    parser.add_argument("--forward_parallel", type=str, default="false",
                        help="Enable forward parallel processing")

    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for the model.')

    # Parse arguments
    args = parser.parse_args()
    if args.forward_parallel == "false":
        forward_parallel = False
    else:
        forward_parallel = True

    # Call the main function with the parsed arguments
    main(rgi_id=args.rgi_id,
         ensemble_size=args.ensemble_size,
         covered_area=args.covered_area,
         inflation=args.inflation,
         iterations=args.iterations,
         seed=args.seed,
         forward_parallel=forward_parallel)
