#!/usr/bin/env python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
from netCDF4 import Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import frost.glacier_model.igm_wrapper as IGM_wrapper
import shutil
import concurrent.futures
import json
from pathlib import Path
import copy


class EnsembleKalmanFilter:
    """
    Implements an Ensemble Kalman Filter (EnKF) for glacier surface mass balance (SMB)
    calibration using forward modeling and observations.

    Authors: Oskar Herrmann

    Args:
        rgi_id (str)          - Glacier ID
        smb_model(str)        - chosen SMB model (ELA, TI, ...)
        ensemble_size (int)   - Number of ensemble members
        inflation (float)     - Inflation factor for Kalman updates
        seed (int)            - Random seed for reproducibility
        start_year (int)      - Start year for the simulation
        output_dir (str)      - Path to output directory
        usurf_ensemble (list) - Initial surface elevation ensemble

    Attributes:
        ensemble_usurf (ndarray)      - Ensemble of surface elevations
        ensemble_smb_raster (ndarray) - SMB raster for each ensemble member
        ensemble_smb (list)           - List of SMB parameters for each member
        ensemble_usurf_log (list)     - Log of surface elevation updates
        ensemble_smb_log (dict)       - Log of SMB updates per key
        current_year (int)            - Current simulation year
        reference_smb (dict)          - Reference SMB values
    """

    def __init__(self, rgi_id, rgi_id_dir, smb_model, ensemble_size, inflation,
                 seed, start_year, smb_prior_mean, smb_prior_std,
                 smb_reference_mean, smb_reference_std, usurf_ensemble, obs_provider,
                 init_offset=0):
        """
        Initializes the Ensemble Kalman Filter by loading required data and setting up
        the initial ensemble.

        Args:
            rgi_id (str)         - Glacier ID
            smb_model(str)        - chosen SMB model (ELA, TI, ...)
            ensemble_size (int)  - Number of ensemble members
            inflation (float)    - Inflation factor for Kalman updates
            seed (int)           - Random seed for reproducibility
            start_year (int)     - Start year for the simulation
            output_dir (str)     - Path to output directory
            usurf_ensemble (list) - Initial surface elevation ensemble

        Returns:
            None
        """

        # Store arguments
        self.rgi_id = rgi_id
        self.smb_model = smb_model
        self.rgi_id_dir = rgi_id_dir
        self.ensemble_size = ensemble_size
        self.inflation = inflation
        self.seed = seed
        self.start_year = start_year
        self.current_year = start_year

        # Create ensemble directory if not existing
        ensemble_dir = os.path.join(self.rgi_id_dir, 'Ensemble')
        os.makedirs(ensemble_dir, exist_ok=True)

        # Load geology file (bedrock and initial icemask)
        inversion_dir = os.path.join(self.rgi_id_dir, 'Preprocess', 'outputs')
        inversion_file = os.path.join(inversion_dir, 'output.nc')
        with Dataset(inversion_file, 'r') as geology_dataset:
            self.icemask_init = np.array(geology_dataset['icemask'])
            self.bedrock = np.array(geology_dataset['topg'])
            init_divflux = np.array(geology_dataset['divflux'])
            init_velsurf_mag = np.array(geology_dataset['velsurf_mag'])

        # average the surface elevation into bins
        self.bedrock_binned = obs_provider.average_elevation_bin(self.bedrock)

        # Initialize placeholders for observable and hidden variables
        self.ensemble_usurf = np.empty((ensemble_size,) + self.icemask_init.shape)
        self.ensemble_smb_raster = np.empty(
            (ensemble_size,) + self.icemask_init.shape)

        self.ensemble_smb_raster = np.zeros_like(self.ensemble_smb_raster)
        self.ensemble_velsurf_mag_raster = np.zeros_like(self.ensemble_smb_raster)
        self.ensemble_divflux_raster = np.zeros_like(self.ensemble_smb_raster)

        self.initial_smb = smb_prior_mean
        self.initial_spread = smb_prior_std
        self.reference_smb = smb_reference_mean
        self.reference_variability = smb_reference_std

        self.init_offset = init_offset
        if not init_offset == 0:
            sign = np.random.choice([-1, 1], 3)
            if str(smb_model) == "ELA":
                self.initial_smb['ela'] = self.reference_smb['ela'] + (
                        500 * init_offset / 100 * sign[0])
                self.initial_smb['gradabl'] = self.reference_smb[
                                                  'gradabl'] + 5 * init_offset / 100 * \
                                              sign[1]
                self.initial_smb['gradacc'] = self.reference_smb[
                                                  'gradacc'] + 5 * init_offset / 100 * \
                                              sign[2]

                self.initial_spread['ela'] = 500
                self.initial_spread['gradabl'] = 5
                self.initial_spread['gradacc'] = 5
            elif str(smb_model) == "TI":
                self.initial_smb['melt_f'] = self.reference_smb['melt_f'] + (
                        4 * init_offset
                        / 100 *
                        sign[0])
                self.initial_spread['melt_f'] = 4
                self.initial_smb['prcp_fac'] = self.reference_smb['prcp_fac'] + (
                        0.25 * init_offset
                        / 100 *
                        sign[1])
                self.initial_spread['prcp_fac'] = 1.0
                self.initial_smb['temp_bias'] = self.reference_smb['temp_bias'] + (
                        2.00 * init_offset
                        / 100 *
                        sign[1])
                self.initial_spread['temp_bias'] = 2.0

        # Initialize random generator and SMB ensemble
        rng = np.random.default_rng(seed=seed)
        self.ensemble_smb = []

        # Initialize each ensemble member
        for e, usurf in enumerate(usurf_ensemble):
            print(f'Initializing ensemble member {e}')

            self.ensemble_usurf[e] = usurf  # Copy initial surface elevation
            self.ensemble_divflux = copy.copy(init_divflux)
            self.ensemble_velsurf_mag = copy.copy(init_velsurf_mag)
            # Sample SMB parameters for each member
            member_smb = {
                key: rng.normal(self.initial_smb[key], self.initial_spread[key])
                for key in self.initial_smb}
            self.ensemble_smb.append(member_smb)

            # Create member directory
            member_dir = os.path.join(ensemble_dir, f'Member_{e}')
            exp_dir = os.path.join(member_dir, 'experiment')
            os.makedirs(exp_dir, exist_ok=True)

            # Copy geology file as the initial input.nc
            data_dir = os.path.join(member_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            shutil.copy2(inversion_file, os.path.join(data_dir, 'input.nc'))

            # Copy iceflow-model directory
            member_iceflow_dir = os.path.join(member_dir, "iceflow-model")
            shutil.rmtree(member_iceflow_dir, ignore_errors=True)
            shutil.copytree(os.path.join(inversion_dir, "iceflow-model"),
                            member_iceflow_dir)

        # Logging initialization
        self.ensemble_init_surf_raster = copy.copy(self.ensemble_usurf)

        self.ensemble_usurf_log = [self.ensemble_usurf]
        self.ensemble_smb_log = {key: [[] for _ in range(self.ensemble_size)] for key
                                 in self.initial_smb}
        for key in self.ensemble_smb_log:
            for e in range(self.ensemble_size):
                self.ensemble_smb_log[key][e].append(self.ensemble_smb[e][key])

    def reset_time(self):
        """
        Resets the ensemble surface elevation to its initial state.
        """
        self.ensemble_usurf = np.copy(self.ensemble_usurf_log[0])
        self.current_year = self.start_year

    def forward(self, year, forward_parallel):
        """
        Advances the ensemble members forward in time.

        Args:
            year (int)              - Target year to advance to
            forward_parallel (bool) - Whether to use parallel execution

        Returns:
            None
        """
        #year_interval = year - self.current_year
        year_start = self.current_year
        year_end   = year
        workers = os.cpu_count()  # Default worker count
        print(f"Default max workers: {workers}")
        new_usurf_ensemble = np.empty_like(self.ensemble_usurf)
        new_smb_raster_ensemble = np.empty_like(self.ensemble_smb_raster)
        new_init_surf_ensemble = np.empty_like(self.ensemble_smb_raster)
        new_velsurf_mag_ensemble = np.empty_like(self.ensemble_smb_raster)
        new_divflux_ensemble = np.empty_like(self.ensemble_smb_raster)

        # Calibration run setting
        exp = "Calibration"
        output1D = False
        output2D_3D = True

        if forward_parallel:

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(IGM_wrapper.forward, exp, output1D, output2D_3D,
                                    member_id, self.rgi_id_dir,
                                    self.smb_model, usurf, smb, year_start, year_end)
                    for member_id, (usurf, smb) in
                    enumerate(zip(self.ensemble_usurf, self.ensemble_smb))
                ]

                for future in concurrent.futures.as_completed(futures):
                    member_id, new_usurf, new_smb_raster, init_usurf, new_velsurf_mag, new_divflux = future.result()
                    new_usurf_ensemble[member_id] = new_usurf
                    new_smb_raster_ensemble[member_id] = new_smb_raster
                    new_init_surf_ensemble[member_id] = init_usurf
                    new_velsurf_mag_ensemble[member_id] = new_velsurf_mag
                    new_divflux_ensemble[member_id] = new_divflux


        else:

            for member_id, (usurf, smb) in enumerate(
                    zip(self.ensemble_usurf, self.ensemble_smb)):
                member_id, new_usurf, new_smb_raster, init_usurf, new_velsurf_mag, new_divflux = IGM_wrapper.forward(
                    exp, output1D, output2D_3D,
                    member_id,
                    self.rgi_id_dir,
                    self.smb_model,
                    usurf,
                    smb,
                    year_start, year_end)
                new_usurf_ensemble[member_id] = new_usurf
                new_smb_raster_ensemble[member_id] = new_smb_raster
                new_init_surf_ensemble[member_id] = init_usurf
                new_velsurf_mag_ensemble[member_id] = new_velsurf_mag
                new_divflux_ensemble[member_id] = new_divflux

        self.ensemble_usurf = new_usurf_ensemble
        self.ensemble_smb_raster = new_smb_raster_ensemble
        self.ensemble_init_surf_raster = new_init_surf_ensemble
        self.ensemble_velsurf_mag_raster = new_velsurf_mag_ensemble
        self.ensemble_divflux_raster = new_divflux_ensemble
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

            noisy_observation = new_observation + member_noise

            noisy_observation = np.maximum(noisy_observation, self.bedrock_binned)

            member_update = kalman_gain.dot(noisy_observation - member_observable)

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

    def save_results(self, elevation_step, iterations, obs_uncertainty, synthetic):
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

        self.params['init_offset'] = self.init_offset
        self.params['ensemble_size'] = self.ensemble_size
        self.params['elevation_step'] = elevation_step
        self.params['iterations'] = iterations
        self.params['obs_uncertainty'] = obs_uncertainty

        self.params['synthetic'] = synthetic
        self.params['seed'] = self.seed
        self.params['inflation'] = self.inflation

        with open(os.path.join(self.rgi_id_dir, 'calibration_results.json'),
                  'w') as f:
            json.dump(self.params, f, indent=4, separators=(',', ': '))
