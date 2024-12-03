import argparse
import os
from netCDF4 import Dataset
import json
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import IGM_wrapper
import shutil

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
        self.rgi_id_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id)
        inversion_dir = os.path.join(self.rgi_id_dir, 'Inversion')
        ensemble_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id, 'Ensemble')
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)

        #TODO transfer necessary data and close file
        geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
        geology_dataset = Dataset(geology_file, 'r')

        observation_file = os.path.join(self.rgi_id_dir, 'observations.nc')
        observation_dataset = Dataset(observation_file, 'r')
        self.years = observation_dataset.variables['time']

        # Initialise the Ensemble and create directories for each member to
        # parallize the forward simulation
        for i in range(self.ensemble_size):
            print('Initialise ensemble member', i )

            # Copy the initial surface elevation
            #TODO create different starting geometries
            member_usurf = copy.copy(np.array(geology_dataset['usurf']))
            self.ensemble_usurf.append(member_usurf)

            # Generate ensemble using the random generator
            member_smb = rng.normal(np.array(initial_smb), np.array(initial_spread))
            self.ensemble_smb.append(member_smb)

            # Create directory for folder if it does not exist
            member_dir = os.path.join(ensemble_dir, f'Member_{i}')
            if not os.path.exists(member_dir):
                os.makedirs(member_dir)

            # copy geology file as initial input.nc
            member_input_file = os.path.join(member_dir, "input.nc")
            shutil.copy2(geology_file, member_input_file)

            # copy iceflow-model
            member_iceflow_dir = os.path.join(member_dir, "iceflow-model")
            if os.path.exists(member_iceflow_dir):
                shutil.rmtree(member_iceflow_dir)
            shutil.copytree(os.path.join(inversion_dir, "iceflow-model"),
                            member_iceflow_dir)




    def forward(self, year_interval, parallel_cpu=True):

        if parallel_cpu:

            # Create a thread pool
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool
                new_usurf = [
                    executor.submit(IGM_wrapper.forward,
                                    member_id,
                                    self.rgi_id_dir,
                                    usurf,
                                    smb,
                                    year_interval)
                    for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                                    self.ensemble_smb))
                ]

                # Wait for all tasks to complete
                for usurf in new_usurf:
                    try:
                        usurf.result(timeout=3600)  # Timeout after 1 hour
                    except Exception as e:
                        print(f"Task failed with error: {e}")

        else:

            for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                             self.ensemble_smb)):
                new_usurf = IGM_wrapper.forward(member_id,
                                                self.rgi_id_dir,
                                                usurf,
                                                smb,
                                                year_interval)

        pass



    def update(self):
        pass

    def save_results(self):
        pass