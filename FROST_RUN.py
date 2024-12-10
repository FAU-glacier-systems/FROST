import argparse
import os
import json
from Scripts.EnsembleKalmanFilter import EnsembleKalmanFilter
from Scripts.Visualization.monitor import Monitor
from netCDF4 import Dataset
import numpy as np
import gstools as gs
from Scripts.Tools import utils


def get_observations(rgi_id, year, ensemble_size, year_interval):
    # load observations
    observation_file = (os.path.join( 'Data', 'Glaciers', rgi_id,
                                     'observations.nc'))
    with Dataset(observation_file, 'r') as ds:
        start_year = ds['time'][0]
        dhdt = ds['dhdt'][year - start_year]
        dhdt_err = ds['dhdt_err'][year - start_year]
        x = ds['x'][:]
        y = ds['y'][:]

    model = utils.Variogram_hugonnet(dim=2)
    srf = gs.SRF(model, mode_no=100)
    srf.set_pos([y, x], "structured")
    samples_srf = [srf() for i in range(ensemble_size)]
    noise = samples_srf * dhdt_err * np.sqrt(year_interval)

    return dhdt, dhdt_err, noise


def main(rgi_id, ensemble_size, covered_area, year_interval, inflation, iterations,
         seed, forward_parallel):
    print(f'Running calibration for glacier: {rgi_id}')
    print(f'Ensemble size: {ensemble_size}',
          f'Covered area: {covered_area}',
          f'Year interval: {year_interval}',
          f'Inflation: {inflation}',
          f'Iterations: {iterations}',
          f'Seed: {seed}',
          f'Forward parallel: {forward_parallel}')

    params_file_path = os.path.join('Experiments', rgi_id,
                                    'params_calibration.json')

    with open(params_file_path, 'r') as file:
        params = json.load(file)
        initial_smb = params['initial_smb']
        initial_spread = params['initial_spread']

    observation_file = (os.path.join( 'Data', 'Glaciers', rgi_id,
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

    monitor_dir = os.path.join( 'Experiments', rgi_id, 'Monitor')
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    monitor = Monitor(EnKF_object=ENKF,
                      monitor_dir=monitor_dir, )

    ################# MAIN LOOP #####################################################
    for i in range(iterations):


        for year in ENKF.years[:-1:year_interval]:
            print(year)
            print('Forward pass...')
            ENKF.forward(year_interval=year_interval,
                         forward_parallel=forward_parallel)

            observation, uncertainty, noise = get_observations(rgi_id,
                                                               year + year_interval,
                                                               ensemble_size,
                                                               year_interval)
            print("Update")
            ENKF.update(observation, uncertainty, noise)
            monitor.plot_status(ENKF, observation, i, year + year_interval)

        ENKF.reset()
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

    parser.add_argument('--year_interval', type=int, default=5,
                        help='Select between 5-year or 20-year dhdt (5, 20)')

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
         year_interval=args.year_interval,
         inflation=args.inflation,
         iterations=args.iterations,
         seed=args.seed,
         forward_parallel=forward_parallel)
