import argparse
import os
import json
from EnsembleKalmanFilter import EnsembleKalmanFilter


def main(rgi_id, ensemble_size, covered_area, year_interval, inflation, iterations,
         seed):
    print(f'Running calibration for glacier: {rgi_id}')
    params_file_path = os.path.join('..', 'Experiments', rgi_id,
                                    'params_calibration.json')
    with open(params_file_path, 'r') as file:
        params = json.load(file)
        initial_smb = params['initial_smb']
        initial_spread = params['initial_spread']

    # TODO save params

    ENKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                ensemble_size=ensemble_size,
                                initial_smb=initial_smb,
                                initial_spread=initial_spread,
                                covered_area=covered_area,
                                year_interval=year_interval,
                                inflation=inflation,
                                seed=seed)

    for i in range(iterations):
        for year in ENKF.years[::year_interval]:
            print(year)
            # TODO implement forward
            ENKF.forward(year_interval=year_interval, parallel_cpu=True)

            # TODO implement update
            ENKF.update()

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

    parser.add_argument('--year_interval', type=int, default=20,
                        help='Select between 5-year or 20-year dhdt (5, 20)')

    parser.add_argument('--inflation', type=float, default=1.0,
                        help='Inflation rate for the model.')

    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations')

    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for the model.')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(rgi_id=args.rgi_id,
         ensemble_size=args.ensemble_size,
         covered_area=args.covered_area,
         year_interval=args.year_interval,
         inflation=args.inflation,
         iterations=args.iterations,
         seed=args.seed)
