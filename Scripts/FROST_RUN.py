import argparse
import os
import json
from EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.Visualization.monitor import Monitor
from netCDF4 import Dataset
import numpy as np


def get_observations(rgi_id, year):
    # load observations
    observation_file = (os.path.join('..', 'Data', 'Glaciers', rgi_id,
                                     'observations.nc'))
    with Dataset(observation_file, 'r') as ds:
        start_year = ds['time'][0]
        dhdt = ds['dhdt'][year-start_year]

    return dhdt


def main(rgi_id, ensemble_size, covered_area, year_interval, inflation, iterations,
         seed):
    print(f'Running calibration for glacier: {rgi_id}')
    params_file_path = os.path.join('..', 'Experiments', rgi_id,
                                    'params_calibration.json')
    with open(params_file_path, 'r') as file:
        params = json.load(file)
        initial_smb = params['initial_smb']
        initial_spread = params['initial_spread']

    observation_file = (os.path.join('..', 'Data', 'Glaciers', rgi_id,
                                     'observations.nc'))
    with Dataset(observation_file, 'r') as ds:
        years = np.array(ds.variables['time'])

    # TODO save params
    ENKF = EnsembleKalmanFilter(rgi_id=rgi_id,
                                ensemble_size=ensemble_size,
                                initial_smb=initial_smb,
                                initial_spread=initial_spread,
                                covered_area=covered_area,
                                years=years,
                                year_interval=year_interval,
                                inflation=inflation,
                                seed=seed)

    monitor_dir = os.path.join('..', 'Experiments', rgi_id, 'Monitor')
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)


    monitor = Monitor(EnKF_object=ENKF,
                      monitor_dir=monitor_dir, )


    for i in range(iterations):
        for year in ENKF.years[::year_interval]:
            print(year)
            print('Forward pass...')
            ENKF.forward(year_interval=year_interval, parallel_cpu=False)

            # TODO implement update
            observation = get_observations(rgi_id, year+year_interval)
            print("Update")
            ENKF.update()
            monitor.plot_status(ENKF, observation, i, year+year_interval,)


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
