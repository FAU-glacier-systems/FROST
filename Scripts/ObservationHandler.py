import numpy as np
import os
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import gstools as gs
import copy


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
    def __init__(self, rgi_id):

        observation_file = (os.path.join('Data', 'Glaciers', rgi_id,
                                         'observations.nc'))

        with Dataset(observation_file, 'r') as ds:
            self.dhdt = ds['dhdt'][:]
            self.dhdt_err = ds['dhdt_err'][:]
            self.icemask = np.array(ds['icemask'][:][0])
            self.usurf = ds['usurf'][:]
            self.usurf_err = ds['usurf_err'][:]
            self.time_period = np.array(ds['time'][:]).astype(int)
            self.x = ds['x'][:]
            self.y = ds['y'][:]

        # self.observation_locations = self.sample_locations()
        self.num_obs_points = 10
        usurf2000_masked = self.usurf[0][self.icemask == 1]
        self.bin_edges = np.linspace(usurf2000_masked.min(), usurf2000_masked.max(),
                                     self.num_obs_points + 1)

        # Compute bin indices for 2000 surface
        self.bin_map = np.digitize(self.usurf[0], self.bin_edges)
        self.bin_map[self.icemask == 0] = 0

        # Compute bin indices for 2000 surface

        self.variogram_model = Variogram_hugonnet(dim=2)
        self.srf = gs.SRF(self.variogram_model, mode_no=100)
        self.srf.set_pos([self.y, self.x], "structured")

    def sample_locations(self):

        slim_icemask = skeletonize(self.icemask)
        print('Number of points: {}'.format(np.sum(slim_icemask)))

        gx, gy = np.where(slim_icemask)
        glacier_points = np.array(list(zip(gx, gy)))

        # We check for NaNs along the third dimension
        valid_mask = ~np.isnan(self.dhdt).any(axis=0)
        observation_points = glacier_points[valid_mask[gx, gy]]

        def get_pixel_value(point):
            x, y = point
            return self.usurf[0][x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        return np.array(sorted_observation_points)

    def compute_bin_uncertainty(self, usurf_err_raster, nan_mask):
        bin_variance = []

        for bin_id in range(1, self.num_obs_points + 1):
            # Filter pixels belonging to the current bin and not masked
            mask = np.logical_and(self.bin_map == bin_id, ~nan_mask)
            err_bin = usurf_err_raster[mask]
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
            covariance_matrix = correlations * pixel_uncertainties

            # Variance of the bin mean
            bin_var = np.sum(covariance_matrix) / (num_pixels ** 2)
            print(f'Bin {bin_id} variance: {bin_var}')
            bin_variance.append(bin_var)

        return bin_variance

    def get_next_observation(self, current_year, num_samples):
        # load observations
        next_index = np.where(self.time_period == current_year)[0][0] + 1
        if next_index >= len(self.time_period):
            return None, None, None, None

        year = self.time_period[next_index]
        usurf_raster = self.usurf[next_index]
        usurf_err_raster = self.usurf_err[next_index]
        self.nan_mask = np.isnan(usurf_raster)

        usurf_line = self.average_elevation_bin(usurf_raster, self.nan_mask)
        # usurf_err_line = self.average_elevation_bin(usurf_err_raster, self.nan_mask)

        # TODO
        noise_matrix = self.compute_bin_uncertainty(usurf_err_raster,
                                                       self.nan_mask)  #

        noise_matrix = np.diag(noise_matrix)
        # noise_matrix = np.eye(len(usurf_line)) * 10

        noise_samples = np.random.multivariate_normal(np.zeros_like(usurf_line),
                                                      noise_matrix, size=num_samples)

        return year, usurf_line, noise_matrix, noise_samples

    def average_elevation_bin(self, usurf, nan_mask):
        # Apply mask to both surfaces

        # Compute average 2020 surface for each bin
        average_usurf = []
        for i in range(1, len(self.bin_edges)):
            # Mask for pixels in the current bin
            bin_pixels = usurf[np.logical_and(self.bin_map == i, ~ nan_mask)]
            avg = np.mean(bin_pixels)
            average_usurf.append(avg)

        # Convert results to a NumPy array
        return np.array(average_usurf)

    def get_observables_from_ensemble(self, EnKF_object):

        ensemble_usurf = EnKF_object.ensemble_usurf
        observables = []
        for usurf in ensemble_usurf:
            observables.append(self.average_elevation_bin(usurf, self.nan_mask))

        return np.array(observables)

    def get_new_geometrie(self, year):
        index = np.where(self.time_period == year)[0][0]
        return self.usurf[index]
