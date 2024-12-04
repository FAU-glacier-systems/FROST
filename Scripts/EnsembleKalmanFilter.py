import os
from netCDF4 import Dataset
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import IGM_wrapper
import shutil

from Scripts.Tools.utils import get_observation_point_locations


class EnsembleKalmanFilter:
    def __init__(self, rgi_id, ensemble_size, initial_smb, initial_spread,
                 covered_area, years, year_interval, inflation, seed):
        self.rgi_id = rgi_id
        self.ensemble_size = ensemble_size
        self.covered_area = covered_area

        self.years = years
        self.year_interval = year_interval
        self.inflation = inflation
        self.seed = seed

        # Create a random generator object
        rng = np.random.default_rng(seed)

        self.ensemble_usurf = []
        self.ensemble_usurf_log = []

        self.ensemble_smb_raster = []

        self.ensemble_smb = []
        self.ensemble_smb_log = {'ela':[[] for e in range(self.ensemble_size)],
                                 'gradabl':[[] for e in range(self.ensemble_size)],
                                 'gradacc':[[] for e in range(self.ensemble_size)]}

        # load inversion data
        self.rgi_id_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id)
        inversion_dir = os.path.join(self.rgi_id_dir, 'Inversion')
        ensemble_dir = os.path.join('..', 'Data', 'Glaciers', rgi_id, 'Ensemble')
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)

        # load geology file
        geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
        with Dataset(geology_file, 'r') as geology_dataset:
            icemask = np.array(geology_dataset['icemask'])
            usurf = np.array(geology_dataset['usurf'])


        # sample observation points to reduce computational costs
        self.observation_point_location = get_observation_point_locations(icemask,
                                                                          usurf,
                                                                          covered_area)

        # Initialise the Ensemble and create directories for each member to
        # parallize the forward simulation
        for i in range(self.ensemble_size):
            print('Initialise ensemble member', i)

            # Copy the initial surface elevation
            # TODO create different starting geometries
            member_usurf = copy.copy(usurf)
            self.ensemble_usurf.append(member_usurf)

            # Generate ensemble using the random generator
            member_smb = {
                key: rng.normal(initial_smb[key], initial_spread[key])
                for key in initial_smb
            }
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

        self.ensemble_usurf_log.append(self.ensemble_usurf)

        for key in self.ensemble_smb_log:
            for e in range(self.ensemble_size):
                self.ensemble_smb_log[key][e].append(self.ensemble_smb[e][key])


    def forward(self, year_interval, parallel_cpu=True):

        if parallel_cpu:

            # Create a thread pool
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool
                (new_usurf_ensemble, new_smb_raster_ensemble) = [
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
                for new_usurf in new_usurf_ensemble:
                    try:
                        new_usurf.result(timeout=3600)  # Timeout after 1 hour
                    except Exception as e:
                        print(f"Task failed with error: {e}")

        else:

            new_usurf_ensemble = []
            new_smb_raster_ensemble = []
            for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                                         self.ensemble_smb)):
                new_usurf, new_smb_raster = IGM_wrapper.forward(member_id,
                                                self.rgi_id_dir,
                                                usurf,
                                                smb,
                                                year_interval)
                new_usurf_ensemble.append(new_usurf)
                new_smb_raster_ensemble.append(new_smb_raster)

        self.ensemble_usurf = new_usurf_ensemble
        self.ensemble_smb_raster = new_smb_raster_ensemble
        self.ensemble_usurf_log.append(new_usurf_ensemble)

    def update(self):

        for key in self.ensemble_smb_log:
            for e in range(self.ensemble_size):
                self.ensemble_smb_log[key][e].append(self.ensemble_smb[e][key])

        pass

    def save_results(self):
        pass
