#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file
import numpy as np
import os
from netCDF4 import Dataset
import gstools as gs
import copy


# Variogram from Hugonnet
class Variogram_hugonnet(gs.CovModel):
    def cor(self, d: np.ndarray):
        """
        Spatial correlation of error in mean elevation change (Hugonnet et al., 2021).

        :param d: Distance between two glaciers (meters).

        :return: Spatial correlation function (input = distance in meters,
         output = correlation between 0 and 1).
        """

        # About 50% correlation of error until 150m, 80% until 2km, etc...
        ranges = [150, 2000, 5000, 20000, 50000, 500000]
        psills = [0.47741896, 0.34238422, 0.06662273, 0.06900394, 0.01602816,
                  0.02854199]

        # Spatial correlation at a given distance (using sum of exponential models)
        return 1 - np.sum((psills[i] * (1 - np.exp(- 3 * d / ranges[i])) for i in
                           range(len(ranges))))


class ObservationProvider:
    """
    The ObservationProvider manages the observations stored in the `observations.nc`
    file and provides the data in a structured format. It processes surface elevation
    data, assigns elevation bins, and computes uncertainty in glacier elevation changes.

    Authors: Oskar Herrmann

    Args:
        rgi_id (str)          - Specify glacier ID
        elevation_step (int)   - Specify elevation step size for binning

    Attributes:
        dhdt (ndarray)        - Surface elevation change rate (dH/dt)
        dhdt_err (ndarray)    - Uncertainty of dH/dt measurements
        icemask (ndarray)     - Binary mask indicating glacier coverage (1 = glacier)
        usurf (ndarray)       - Observed surface elevation
        usurf_err (ndarray)   - Uncertainty in surface elevation
        time_period (ndarray) - Array indicating time periods of observations
        x (ndarray)           - X-coordinates (longitude)
        y (ndarray)           - Y-coordinates (latitude)
        resolution (int)      - Spatial resolution (grid cell size)
        bin_edges (ndarray)   - Computed elevation bin boundaries
        bin_map (ndarray)     - Map assigning grid cells to elevation bins
        bins (list)           - List of spatial locations grouped by elevation bins
        num_bins (int)        - Total number of elevation bins
        variogram_model       - Variogram model for spatial correlation analysis

    Returns:
        None
    """

    def __init__(self, rgi_id, elevation_step, obs_uncertainty, synthetic):
        """
        Initializes the ObservationProvider by loading glacier observation data
        from a NetCDF file and computing elevation bins.

        Args:
            rgi_id (str)          - Specify glacier ID
            elevation_step (int)   - Specify elevation step size for binning

        Returns:
            None
        """

        # Path to the NetCDF file containing glacier observation data
        if synthetic:
            observation_file = os.path.join('Data', 'Glaciers', rgi_id,
                                            'SyntheticData', 'observations_synth.nc')
        else:
            observation_file = os.path.join('Data', 'Glaciers', rgi_id,
                                            'observations.nc')

        # Load important values from the observation file
        with Dataset(observation_file, 'r') as ds:
            self.dhdt = ds['dhdt'][:]  # Surface elevation change rate (dH/dt)
            self.dhdt_err = ds['dhdt_err'][:]  # Uncertainty of dH/dt
            self.icemask = np.array(ds['icemask'][:][0])  # Binary glacier mask
            self.usurf = ds['usurf'][:]  # Surface elevation
            self.usurf_err = ds['usurf_err'][:]
            self.velsurf_mag = ds['velsurf_mag'][:]  #
            self.time_period = np.array(ds['time'][:]).astype(
                int)  # Time period array
            self.x = ds['x'][:]  # X-coordinates (longitude)
            self.y = ds['y'][:]  # Y-coordinates (latitude)


        # Compute the spatial resolution from the coordinate grid
        self.resolution = int(self.x[1] - self.x[0])
        self.elevation_step = elevation_step
        self.obs_uncertainty = obs_uncertainty
        self.synthetic = synthetic

        # Masked surface elevation for glacier areas (year 2000)
        usurf2000_masked = self.usurf[0][self.icemask == 1]

        # Define bin edges based on a fixed elevation step size
        min_elev = np.floor(
            usurf2000_masked.min() / self.elevation_step) * self.elevation_step
        max_elev = np.ceil(
            usurf2000_masked.max() / self.elevation_step) * self.elevation_step
        self.bin_edges = np.arange(min_elev, max_elev + self.elevation_step,
                                   self.elevation_step)

        # Compute bin indices for the surface elevation of 2000
        self.bin_map = np.digitize(self.usurf[0], self.bin_edges)
        self.bin_map[self.icemask == 0] = 0  # Mask out non-glacier areas

        # Initialize bins to store grid locations for each elevation bin
        self.bins = []
        for bin_id in range(1, len(self.bin_edges)):  # Bins are indexed from 1
            index_x, index_y = np.where(
                self.bin_map == bin_id)  # Get indices for this bin
            loc_x, loc_y = self.y[index_x], self.x[
                index_y]  # Convert indices to coordinates
            locations = np.column_stack(
                (loc_x, loc_y))  # Store as an array of (lat, lon) pairs
            self.bins.append(locations)  # Append bin locations to the list

        # Store the total number of bins
        self.num_bins = len(self.bins)

        # Initialize the variogram model for spatial correlation analysis
        self.variogram_model = Variogram_hugonnet(dim=2)

    def get_next_observation(self, current_year, num_samples):
        # load observations
        next_index = np.where(self.time_period == current_year)[0][0] + 1
        # next_index = 1  # TODO
        if next_index >= len(self.time_period):
            return None, None, None, None

        # get data of next year
        year = int(self.time_period[next_index])
        usurf_raster = self.usurf[next_index]
        usurf_err_raster = self.usurf_err[next_index]
        self.nan_mask = np.isnan(usurf_raster)

        # average the surface elevation into bins
        usurf_line = self.average_elevation_bin(usurf_raster, self.nan_mask)

        # compute uncertainty of the average bin value and standard deviation of
        # the bins
        bin_variance = self.compute_bin_variance(usurf_raster, usurf_err_raster,
                                                 self.nan_mask)

        noise_matrix = self.compute_covariance_matrix(bin_variance)

        print("Bin variance len", len(bin_variance))

        noise_samples = np.random.multivariate_normal(np.zeros_like(usurf_line),
                                                      noise_matrix,
                                                      size=num_samples)
        dhdt = self.dhdt[next_index]
        velocity = self.velsurf_mag[next_index]
        return year, usurf_line, noise_matrix, noise_samples, dhdt, velocity

    def initial_usurf(self, num_samples, sample=False):
        index = 0

        # get data of next year
        year = self.time_period[index]
        usurf_raster = self.usurf[index]
        usurf_err_raster = self.usurf_err[index]
        usurf_err_masked = usurf_err_raster[self.icemask == 1]

        ensemble_usurf = np.empty((num_samples,) + usurf_raster.shape)

        if sample:
            index_x, index_y = np.where(self.icemask)
            loc_x, loc_y = self.y[index_x], self.x[index_y]
            locations = np.column_stack((loc_x, loc_y))

            # Compute pairwise distances between all points in bin1 and bin2
            distances = np.linalg.norm(
                locations[:, None, :] - locations[None, :, :], axis=2
            )  # Shape: (n1, n2)

            # Apply the correlation function to each distance
            correlations = self.variogram_model.cor(distances)
            pixel_uncertainties = usurf_err_masked[:, np.newaxis] * usurf_err_masked[
                                                                    np.newaxis, :]
            covariance_matrix = correlations * pixel_uncertainties

            noise_samples = np.random.multivariate_normal(
                np.zeros_like(usurf_err_masked),
                covariance_matrix, size=num_samples)

            for e, noise_sample in enumerate(noise_samples):
                usurf_sample = copy.deepcopy(usurf_raster)
                usurf_sample[self.icemask == 1] += noise_sample
                ensemble_usurf[e] = usurf_sample

        else:
            for e in range(num_samples):
                usurf_sample = copy.deepcopy(usurf_raster)
                ensemble_usurf[e] = usurf_sample

        binned_usurf = []
        for usurf in ensemble_usurf:
            binned_usurf.append(self.average_elevation_bin(usurf))

        return int(year), ensemble_usurf, np.array(
            binned_usurf), self.average_elevation_bin(usurf_raster)

    def compute_bin_variance(self, usurf_raster, usurf_err_raster, nan_mask):
        bin_variance = []

        for bin_id in range(1, self.num_bins + 1):
            # Filter pixels belonging to the current bin and not masked

            mask = np.logical_and(self.bin_map == bin_id, ~nan_mask)
            err_bin = usurf_err_raster[mask]
            usurf_bin = usurf_raster[mask]
            usurf_bin_var = np.var(usurf_bin)  # delat/2

            index_x, index_y = np.where(mask)
            loc_x, loc_y = self.y[index_x], self.x[index_y]
            locations = np.column_stack((loc_x, loc_y))

            num_pixels = len(locations)
            if num_pixels == 0:  # Skip empty bins
                bin_variance.append(0.0)
                continue

            # Compute pairwise distances (vectorized)
            distances = np.linalg.norm(
                locations[:, np.newaxis, :] - locations[np.newaxis, :, :], axis=2
            )

            # Apply variogram model to compute correlations
            correlations = self.variogram_model.cor(distances)

            # Compute covariance matrix (vectorized)
            pixel_uncertainties = err_bin[:, np.newaxis] * err_bin[np.newaxis, :]
            if self.synthetic:
                covariance_matrix = (correlations * pixel_uncertainties *
                                     self.obs_uncertainty ** 2)
            else:
                covariance_matrix = (correlations * pixel_uncertainties)

            # Variance of the bin mean
            bin_var = np.sum(covariance_matrix) / (num_pixels ** 2)
            combined_var = bin_var + usurf_bin_var

            bin_variance.append(combined_var)

        return bin_variance

    def compute_pairwise_correlation(self, bin1_coords, bin2_coords):
        """
        Compute pairwise correlation between two bins.

        :param bin1_coords: Array of coordinates (x, y) for bin 1 (shape: [n1, 2]).
        :param bin2_coords: Array of coordinates (x, y) for bin 2 (shape: [n2, 2]).
        :param cor_function: Function that computes correlation given a distance.

        :return: Average pairwise correlation between the bins.
        """
        # Compute pairwise distances between all points in bin1 and bin2

        distances = np.linalg.norm(
            bin1_coords[:, None, :] - bin2_coords[None, :, :], axis=2
        )  # Shape: (n1, n2)

        # Apply the correlation function to each distance
        correlations = self.variogram_model.cor(distances)

        # Average the correlations
        return correlations.mean()


    def compute_covariance_matrix(self, bin_variance):
        """
        Compute the covariance matrix for multiple bins.
        """

        corr_matrix = np.zeros((self.num_bins, self.num_bins))

        # Compute pairwise correlations for all bins
        for i in range(self.num_bins):
            for j in range(i, self.num_bins):
                correlation = self.compute_pairwise_correlation(self.bins[i],
                                                                self.bins[j])
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation  # Symmetry
        # Set diagonal to 1 for correlation matrix
        np.fill_diagonal(corr_matrix, 1.0)
        # Convert correlation matrix to covariance matrix
        cov_matrix = np.zeros_like(corr_matrix)
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                cov_matrix[i, j] = corr_matrix[i, j] * np.sqrt(
                    bin_variance[i] * bin_variance[j])

        return cov_matrix


    def average_elevation_bin(self, usurf, nan_mask=None):
        # Apply mask to both surfaces
        # If no mask is provided, treat all values as valid
        if nan_mask is None:
            nan_mask = np.zeros_like(self.bin_map, dtype=bool)
        # Compute average 2020 surface for each bin
        average_usurf = []
        for i in range(1, self.num_bins + 1):
            # Mask for pixels in the current bin
            bin_pixels = usurf[np.logical_and(self.bin_map == i, ~ nan_mask)]
            avg = np.mean(bin_pixels)
            average_usurf.append(avg)

        # Convert results to a NumPy array
        return np.array(average_usurf)


    def get_ensemble_observables(self, EnKF_object):
        ensemble_usurf = EnKF_object.ensemble_usurf
        observables = []
        for usurf in ensemble_usurf:
            observables.append(self.average_elevation_bin(usurf, self.nan_mask))
            # observables.append(usurf[self.obs_locations[:, 0],
        # self.obs_locations[:, 1]])
        return np.array(observables)


    def get_new_geometrie(self, year):
        index = np.where(self.time_period == year)[0][0]
        return self.usurf[index]
