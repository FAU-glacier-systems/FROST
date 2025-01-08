import numpy as np
import os

from matplotlib import pyplot as plt
from netCDF4 import Dataset
import gstools as gs


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
    def __init__(self, rgi_id, covered_area):

        observation_file = (os.path.join('Data', 'Glaciers', rgi_id,
                                         'observations.nc'))

        with Dataset(observation_file, 'r') as ds:
            self.dhdt = ds['dhdt'][:]
            self.dhdt_err = ds['dhdt_err'][:]
            self.icemask = ds['icemask'][:][0]
            self.usurf = ds['usurf'][:][0]
            self.time_period = np.array(ds['time'][:]).astype(int)
            self.x = ds['x'][:]
            self.y = ds['y'][:]

        self.observation_locations = self.sample_locations(covered_area)

        self.variogram_model = Variogram_hugonnet(dim=2)
        self.srf = gs.SRF(self.variogram_model, mode_no=100)
        self.srf.set_pos([self.y, self.x], "structured")

    def sample_locations(self, covered_area):
        num_sample_points = int((covered_area / 100) * np.sum(self.icemask))
        print('Number of points: {}'.format(num_sample_points))

        gx, gy = np.where(self.icemask)
        glacier_points = np.array(list(zip(gx, gy)))

        # We check for NaNs along the third dimension
        valid_mask = ~np.isnan(self.dhdt).any(axis=0)
        valid_points = glacier_points[valid_mask[gx, gy]]

        random_state = np.random.RandomState(seed=42)
        observation_index = random_state.choice(len(valid_points),
                                                num_sample_points,
                                                replace=False)
        observation_points = valid_points[observation_index]

        def get_pixel_value(point):
            x, y = point
            return self.usurf[x][y]

        sorted_observation_points = sorted(observation_points, key=get_pixel_value)
        return np.array(sorted_observation_points)

    def generate_covariance_matrix(self, uncertainties):
        num_pixels = len(self.observation_locations)
        covariance_matrix = np.zeros((num_pixels, num_pixels))

        # Compute pairwise covariance
        for i in range(num_pixels):
            for j in range(i,num_pixels): # use symmetry
                # Calculate distance between locations i and j (assuming 2D coordinates)
                distance = np.linalg.norm(
                    self.observation_locations[i] - self.observation_locations[j])

                # Get the correlation for this distance
                correlation = self.variogram_model.cor(distance)

                # Calculate the covariance: Corr * sigma_i * sigma_j
                covariance_matrix[i, j] = correlation * uncertainties[i] * \
                                          uncertainties[j]
                covariance_matrix[j, i] = covariance_matrix[i, j]  # Symmetry

        return covariance_matrix

    def get_next_observation(self, current_year, num_samples):
        # load observations
        next_index = np.where(self.time_period == current_year)[0][0] + 1
        if next_index >= len(self.time_period):
            return None, None, None

        year = self.time_period[next_index]
        dhdt_raster = self.dhdt[next_index]
        dhdt_err_raster = self.dhdt_err[next_index]
        dhdt = dhdt_raster[self.observation_locations[:,0],
                           self.observation_locations[:, 1]]

        dhdt_err = dhdt_err_raster[self.observation_locations[:, 0],
                                   self.observation_locations[:, 1]]

        noise_matrix = self.generate_covariance_matrix(dhdt_err)

        noise_samples = np.random.multivariate_normal(np.zeros_like(dhdt),
                                                      noise_matrix, size=num_samples)

        return year, dhdt, noise_matrix, noise_samples

    def get_observables_from_ensemble(self, EnKF_object):
        ensemble_usurf_log = EnKF_object.ensemble_usurf_log

        # Check that there are at least two time steps
        if len(ensemble_usurf_log) < 2:
            return None

        current_usurf_ensemble = np.array(ensemble_usurf_log[-1])
        previous_usurf_ensemble = np.array(ensemble_usurf_log[-2])

        # Compute the difference between the current and previous time step
        dhdt_ensemble = current_usurf_ensemble - previous_usurf_ensemble
        dhdt_ensemble_all.append(dhdt_ensemble)

        # Convert the differences to a numpy array
        dhdt_ensemble_all = np.array(dhdt_ensemble_all)

        # Compute mean dh/dt across the spatial dimensions (x, y)
        mean_dhdt = dhdt_ensemble_all.mean(axis=(2, 3))
        mean_dhdt /= EnKF_object.year_interval

        return observables
