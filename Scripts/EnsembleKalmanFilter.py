import os
from netCDF4 import Dataset
import copy
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import Scripts.IGM_wrapper as IGM_wrapper
import shutil
import concurrent.futures
import json


class EnsembleKalmanFilter:
    def __init__(self, rgi_id, ensemble_size, inflation, seed, start_year,
                 output_dir, usurf_ensemble):

        # save arguments
        self.rgi_id = rgi_id
        self.rgi_id_dir = os.path.join('Data', 'Glaciers', rgi_id)
        self.ensemble_size = ensemble_size
        self.inflation = inflation
        self.seed = seed
        self.start_year = start_year
        self.current_year = start_year
        self.output_dir = output_dir

        # create a folder to store the in- and output of the ensemble members
        ensemble_dir = os.path.join('Data', 'Glaciers', rgi_id, 'Ensemble')
        if not os.path.exists(ensemble_dir):
            os.makedirs(ensemble_dir)

        # load geology file
        inversion_dir = os.path.join(self.rgi_id_dir, 'Inversion')
        geology_file = os.path.join(inversion_dir, 'geology-optimized.nc')
        with Dataset(geology_file, 'r') as geology_dataset:
            self.icemask_init = np.array(geology_dataset['icemask'])

        # create placeholder for observable and hidden variables
        self.ensemble_usurf = np.empty(
            (self.ensemble_size,) + self.icemask_init.shape)
        self.ensemble_smb_raster = np.empty(
            (self.ensemble_size,) + self.icemask_init.shape)

        # Load parameter file with glacier specific values
        params_file_path = os.path.join('Experiments', rgi_id,
                                        'params_calibration.json')
        # Load properties of initial ensemble
        with open(params_file_path, 'r') as file:
            params = json.load(file)
            self.initial_smb = params['initial_smb']
            self.initial_spread = params['initial_spread']
            self.reference_smb = params['reference_smb']

        rng = np.random.default_rng(seed=seed)
        self.ensemble_smb = []

        # Initialise the Ensemble and create directories for each member to
        # parallize the forward simulation
        for e, usurf in enumerate(usurf_ensemble):
            print('Initialise ensemble member', e)

            # Copy the initial surface elevation
            # TODO create different starting geometries
            self.ensemble_usurf[e] = usurf

            # Generate ensemble using the random generator
            member_smb = {
                key: rng.normal(self.initial_smb[key], self.initial_spread[key])
                for key in self.initial_smb
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

        ####################### logging #############################################
        self.ensemble_usurf_log = []
        self.ensemble_usurf_log.append(self.ensemble_usurf)

        self.ensemble_smb_log = {'ela': [[] for _ in range(self.ensemble_size)],
                                 'gradabl': [[] for _ in range(self.ensemble_size)],
                                 'gradacc': [[] for _ in range(self.ensemble_size)]}

        for key in self.ensemble_smb_log:
            for e in range(self.ensemble_size):
                self.ensemble_smb_log[key][e].append(self.ensemble_smb[e][key])

    def reset_time(self):
        # resets the ensemble usurf
        for e in range(self.ensemble_size):
            self.ensemble_usurf[e] = self.ensemble_usurf_log[0][e]

        # self.ensemble_usurf_log = [self.ensemble_usurf]
        self.current_year = self.start_year
        # for key in self.ensemble_smb_log:
        #    for e in range(self.ensemble_size):
        #        self.ensemble_smb_log[key][e] = [self.ensemble_smb[e][key]]

    def forward(self, year, forward_parallel):
        # forwards the ensemble to the given year
        year_interval = int(year) - self.current_year
        if forward_parallel:

            # Create a thread pool
            with ThreadPoolExecutor() as executor:
                # Submit tasks to the thread pool
                # Prepare the list of futures for parallel execution
                futures = [
                    executor.submit(
                        IGM_wrapper.forward,  # The function to call
                        member_id,  # Member ID
                        self.rgi_id_dir,  # Directory for RGI ID
                        usurf,  # Surface elevation
                        smb,  # Surface mass balance
                        year_interval  # Time interval
                    )
                    for member_id, (usurf, smb) in enumerate(
                        zip(self.ensemble_usurf, self.ensemble_smb)
                        # Pair usurf and smb for each ensemble member
                    )
                ]

                # Initialize storage for results
                id_order = []
                new_usurf_ensemble = np.empty_like(self.ensemble_usurf)
                new_smb_raster_ensemble = np.empty_like(self.ensemble_smb_raster)
                # Wait for all tasks to complete
                for future in concurrent.futures.as_completed(futures):
                    member_id, new_usurf, new_smb_raster = future.result()  # Unpack
                    # the returned values
                    id_order.append(member_id)
                    new_usurf_ensemble[member_id] = new_usurf
                    new_smb_raster_ensemble[member_id] = new_smb_raster
                print("################# id Order")
                print(id_order)

        else:

            new_usurf_ensemble = np.empty_like(self.ensemble_usurf)
            new_smb_raster_ensemble = np.empty_like(self.ensemble_smb_raster)
            for member_id, (usurf, smb) in enumerate(zip(self.ensemble_usurf,
                                                         self.ensemble_smb)):
                member_id, new_usurf, new_smb_raster = IGM_wrapper.forward(member_id,
                                                                           self.rgi_id_dir,
                                                                           usurf,
                                                                           smb,
                                                                           year_interval)
                new_usurf_ensemble[member_id] = new_usurf
                new_smb_raster_ensemble[member_id] = new_smb_raster

        self.ensemble_usurf = new_usurf_ensemble
        self.ensemble_smb_raster = new_smb_raster_ensemble
        self.ensemble_usurf_log.append(new_usurf_ensemble)
        self.current_year = int(year)

    def update(self, new_observation, noise_matrix, noise_samples,
               modeled_observables):

        ensemble_obs_mean = np.mean(modeled_observables, axis=0)
        ensemble_deviations_obs = modeled_observables - ensemble_obs_mean
        ensemble_cov = (
                np.dot(ensemble_deviations_obs.T, ensemble_deviations_obs) / (
                self.ensemble_size - 1) + noise_matrix)

        # Convert self.ensemble_smb from list of dict into np.array
        keys = self.initial_smb.keys()
        ensemble_smb = np.array([
            [member_smb[key] for key in keys]
            for member_smb in self.ensemble_smb
        ])
        ensemble_smb_mean = np.mean(ensemble_smb, axis=0)
        deviations_smb = ensemble_smb - ensemble_smb_mean

        cross_covariance = np.dot(ensemble_deviations_obs.T, deviations_smb) / (
                self.ensemble_size - 1)

        kalman_gain = np.dot(cross_covariance.T, np.linalg.inv(ensemble_cov))

        new_ensemble_smb = []
        for e, (member_smb, member_observable, member_noise) in enumerate(zip(
                self.ensemble_smb,
                modeled_observables,
                noise_samples)):

            member_update = kalman_gain.dot(new_observation
                                            + member_noise
                                            - member_observable)

            new_member_smb = {}
            for i, key in enumerate(member_smb.keys()):
                new_member_smb[key] = member_smb[key] + member_update[i]

            new_ensemble_smb.append(new_member_smb)

        ### INFLATION ###
        # Compute the ensemble mean for each key
        ensemble_mean = {
            key: sum(member[key] for member in new_ensemble_smb) / len(
                new_ensemble_smb)
            for key in new_ensemble_smb[0].keys()}

        # Apply multiplicative inflation to each member
        inflated_ensemble_smb = []
        for e, member in enumerate(new_ensemble_smb):
            inflated_member = {}
            for key in member.keys():
                deviation = member[key] - ensemble_mean[key]
                inflated_member[key] = (ensemble_mean[key] + self.inflation *
                                        deviation)
                # logging
                self.ensemble_smb_log[key][e].append(inflated_member[key])

            inflated_ensemble_smb.append(inflated_member)

        self.ensemble_smb = inflated_ensemble_smb

    def save_results(self, num_bins):
        self.params = dict()

        keys = self.initial_smb.keys()
        ensemble_smb = np.array([
            [member_smb[key] for key in keys]
            for member_smb in self.ensemble_smb
        ])

        self.params['final_mean'] = list(ensemble_smb.mean(axis=0))
        self.params['final_std'] = list(ensemble_smb.std(axis=0))
        self.params['final_ensemble'] = [list(sigma) for sigma in ensemble_smb]

        # information
        self.params['initial_smb'] = self.initial_smb
        self.params['initial_spread'] = self.initial_spread
        self.params['reference_smb'] = self.reference_smb
        self.params['ensemble_size'] = self.ensemble_size
        self.params['inflation'] = self.inflation
        self.params['num_bins'] = num_bins
        self.params['seed'] = self.seed

        from pathlib import Path

        # Ensure self.output_dir is a Path object
        self.output_dir = Path(self.output_dir)

        # Use / operator to join paths
        output_path = self.output_dir / (
            f"result_seed_{self.seed}_{self.inflation}_{num_bins}.json"

            # Write to the file
        )
        with open(output_path, 'w') as f:
            json.dump(self.params, f, indent=4, separators=(',', ': '))
