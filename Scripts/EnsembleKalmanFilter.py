import os
from netCDF4 import Dataset
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import Scripts.IGM_wrapper as IGM_wrapper
import shutil
import concurrent.futures
from Scripts.Tools.utils import get_observation_point_locations
import matplotlib.pyplot as plt


class EnsembleKalmanFilter:
    def __init__(self, rgi_id, ensemble_size, initial_smb, initial_spread,
                 covered_area, years, year_interval, inflation, seed):
        self.rgi_id = rgi_id
        self.ensemble_size = ensemble_size
        self.covered_area = covered_area
        self.initial_smb = initial_smb

        self.years = years
        self.year_interval = year_interval
        self.inflation = inflation
        self.seed = seed

        # Create a random generator object
        rng = np.random.default_rng(seed)

        self.ensemble_smb = []
        self.ensemble_smb_log = {'ela': [[] for e in range(self.ensemble_size)],
                                 'gradabl': [[] for e in range(self.ensemble_size)],
                                 'gradacc': [[] for e in range(self.ensemble_size)]}

        # load inversion data
        self.rgi_id_dir = os.path.join('Data', 'Glaciers', rgi_id)
        inversion_dir = os.path.join(self.rgi_id_dir, 'Inversion')
        ensemble_dir = os.path.join('Data', 'Glaciers', rgi_id, 'Ensemble')
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)

        # load geology file
        geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
        with Dataset(geology_file, 'r') as geology_dataset:
            self.icemask_init = np.array(geology_dataset['icemask'])
            self.usurf_init = np.array(geology_dataset['usurf'])

        # sample observation points to reduce computational costs
        self.ensemble_usurf = np.empty((self.ensemble_size,) + self.usurf_init.shape)
        self.ensemble_usurf_log = []

        self.ensemble_smb_raster = np.empty(
            (self.ensemble_size,) + self.usurf_init.shape)

        self.observation_point_location = get_observation_point_locations(
            self.icemask_init,
            self.usurf_init,
            covered_area)

        # Initialise the Ensemble and create directories for each member to
        # parallize the forward simulation
        for e in range(self.ensemble_size):
            print('Initialise ensemble member', e)

            # Copy the initial surface elevation
            # TODO create different starting geometries
            member_usurf = copy.copy(self.usurf_init)
            self.ensemble_usurf[e] = member_usurf

            # Generate ensemble using the random generator
            member_smb = {
                key: rng.normal(initial_smb[key], initial_spread[key])
                for key in initial_smb
            }
            self.ensemble_smb.append(member_smb)

            # Create directory for folder if it does not exist
            member_dir = os.path.join(ensemble_dir, f'Member_{e}')
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

    def reset(self):

        for e in range(self.ensemble_size):
            self.ensemble_usurf[e] = self.usurf_init

        self.ensemble_usurf_log = [self.ensemble_usurf]
        for key in self.ensemble_smb_log:
            for e in range(self.ensemble_size):
                self.ensemble_smb_log[key][e] = [self.ensemble_smb[e][key]]

    def forward(self, year_interval, forward_parallel):

        if forward_parallel:

            # Create a thread pool
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool
                futures = [
                    executor.submit(IGM_wrapper.forward,
                                    member_id,
                                    self.rgi_id_dir,
                                    usurf,
                                    smb,
                                    year_interval)
                    for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                                                 self.ensemble_smb))
                ]

                # Initialize storage for results
                new_usurf_ensemble = []
                new_smb_raster_ensemble = []
                # Wait for all tasks to complete
                for future in concurrent.futures.as_completed(futures):
                    new_usurf, new_smb_raster = future.result()  # Unpack the returned values
                    new_usurf_ensemble.append(new_usurf)
                    new_smb_raster_ensemble.append(new_smb_raster)

        else:

            new_usurf_ensemble = np.empty_like(self.ensemble_usurf)
            new_smb_raster_ensemble = np.empty_like(self.ensemble_smb_raster)
            for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                                         self.ensemble_smb)):
                new_usurf, new_smb_raster = IGM_wrapper.forward(member_id,
                                                                self.rgi_id_dir,
                                                                usurf,
                                                                smb,
                                                                year_interval)
                new_usurf_ensemble[member_id] = new_usurf
                new_smb_raster_ensemble[member_id] = new_smb_raster

        self.ensemble_usurf = new_usurf_ensemble
        self.ensemble_smb_raster = new_smb_raster_ensemble
        self.ensemble_usurf_log.append(new_usurf_ensemble)

    def update(self, observation, uncertainty, noise):

        observation_sampled = observation[self.observation_point_location[:, 0],
        self.observation_point_location[:, 1]]

        uncertainty_sampled = uncertainty[self.observation_point_location[:, 0],
        self.observation_point_location[:, 1]]

        noise_sampled = noise[:, self.observation_point_location[:, 0],
                        self.observation_point_location[:, 1]]

        uncertainty_R = np.diag(uncertainty_sampled ** 2)

        indices = tuple(self.observation_point_location.T)
        ensemble_usurf = np.array(self.ensemble_usurf)
        ensemble_usurf_previous = np.array(self.ensemble_usurf_log)[-2]
        ensemble_dhdt = (
                                ensemble_usurf - ensemble_usurf_previous) / self.year_interval
        ensemble_dhdt_sampled = ensemble_dhdt[:, indices[0], indices[1]]

        ensemble_usurf_mean = np.mean(ensemble_dhdt_sampled, axis=0)
        deviations_usurf = ensemble_dhdt_sampled - ensemble_usurf_mean
        ensemble_cov = (np.dot(deviations_usurf.T, deviations_usurf) / (
                self.ensemble_size - 1)
                        + uncertainty_R
                        )

        # Convert self.ensemble_smb from list of dict into np.array
        keys = self.initial_smb.keys()
        ensemble_smb = np.array([
            [member_smb[key] for key in keys]
            for member_smb in self.ensemble_smb
        ])
        ensemble_smb_mean = np.mean(ensemble_smb, axis=0)
        deviations_smb = ensemble_smb - ensemble_smb_mean

        cross_covariance = np.dot(deviations_usurf.T, deviations_smb) / (
                self.ensemble_size - 1)

        kalman_gain = np.dot(cross_covariance.T, np.linalg.inv(ensemble_cov))

        new_ensemble_smb = []
        for e, (member_smb, member_usurf, member_noise) in enumerate(zip(
                self.ensemble_smb,
                ensemble_dhdt_sampled,
                noise_sampled)):
            member_update = kalman_gain.dot(observation_sampled
                                            + member_noise
                                            - member_usurf)

            new_member_smb = {}
            for i, key in enumerate(member_smb.keys()):
                new_member_smb[key] = member_smb[key] + member_update[i]
                self.ensemble_smb_log[key][e].append(new_member_smb[key])

            new_ensemble_smb.append(new_member_smb)

        self.ensemble_smb = new_ensemble_smb
        self.ensemble_usurf = (self.ensemble_usurf_log[0]
                               + observation * self.year_interval)
        self.ensemble_usurf_log.append(self.ensemble_usurf)

    def save_results(self):
        pass
