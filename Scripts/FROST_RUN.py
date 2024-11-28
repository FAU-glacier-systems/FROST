import argparse
import os
from netCDF4 import Dataset
import json
import copy
import numpy as np

class EnsembleKalmanFilter:
    def __init__(self, rgi_id, ensemble_size, initial_smb, initial_spread,
                 covered_area, year_interval, inflation, seed):
        self.rgi_id = rgi_id
        self.ensemble_size = ensemble_size
        self.covered_area = covered_area
        self.year_interval = year_interval
        self.inflation = inflation
        self.seed = seed

        # Create a random generator object
        rng = np.random.default_rng(seed)

        self.ensemble_usurf = []
        self.ensemble_smb = []

        # load inversion data
        rgi_id_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id)
        inversion_dir = os.path.join(rgi_id_dir, 'Inversion')
        ensemble_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id, 'Ensemble')
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)

        geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
        geology_dataset = Dataset(geology_file, 'r')

        for i in range(self.ensemble_size):
            member_usurf = copy.copy(np.array(geology_dataset['usurf']))
            self.ensemble_usurf.append(member_usurf)

            # Generate ensemble using the random generator
            member_smb = rng.normal(np.array(initial_smb), np.array(initial_spread))
            self.ensemble_smb.append(member_smb)


    def forward(self, parallel_cpu=True):
        pass

    def update(self):
        pass

    def save_results(self):
        pass



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
                                inflation= inflation,
                                seed=seed)
    
    for i in range(iterations):
        #TODO implement forward
        ENKF.forward()

        #TODO implement update
        ENKF.update()
        
    ENKF.save_results()
    
    


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run glacier calibration experiments.')

    # Add arguments for parameters
    parser.add_argument('--rgi_id', type=str, required=True,
                        help='RGI ID of the glacier for the model.')
    
    parser.add_argument('--ensemble_size', type=str, default=50,
                        help='number of ensemble members for the model.')

    parser.add_argument('--covered_area', type=float, default=50,
                        help='Fraction of the area of the glacier that is covered'
                             'by the observations')

    parser.add_argument('--year_interval', type=int, default=20,
                        help='Select between 5-year or 20-year dhdt (5, 20)')

    parser.add_argument( '--inflation', type=float, default= 1.0,
                        help='Inflation rate for the model.' )
    
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
