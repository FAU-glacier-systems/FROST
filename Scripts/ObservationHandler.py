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

        #self.observation_locations = self.sample_locations()
        self.num_obs_points = 20
        usurf2000_masked = self.usurf[0][self.icemask == 1]
        self.bin_edges = np.linspace(usurf2000_masked.min(), usurf2000_masked.max(),
                                self.num_obs_points + 1)

        # Compute bin indices for 2000 surface
        self.bin_indices = np.digitize(usurf2000_masked, self.bin_edges)
        # Compute bin indices for 2000 surface

        # Create a grid for visualization
        bin_colored_image = np.zeros_like(self.usurf[0], dtype=int)
        bin_colored_image[self.icemask == 1] = self.bin_indices
        # Plot the bins colored by indices

        plt.figure(figsize=(10, 8))
        plt.imshow(bin_colored_image, cmap="viridis", origin="lower")
        plt.colorbar(label="Bin Index")
        plt.title("Pixels Colored by Elevation Bin")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show(dpi=300)
        self.variogram_model = Variogram_hugonnet(dim=2)
        self.srf = gs.SRF(self.variogram_model, mode_no=100)
        self.srf.set_pos([self.y, self.x], "structured")

    def sample_locations(self, ):

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

    def generate_covariance_matrix(self, uncertainties):
        num_pixels = len(self.observation_locations)
        covariance_matrix = np.zeros((num_pixels, num_pixels))

        # Compute pairwise covariance
        for i in range(num_pixels):
            for j in range(i, num_pixels):  # use symmetry
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
            return None, None, None, None

        year = self.time_period[next_index]
        usurf_raster = self.usurf[next_index]
        usurf_err_raster = self.usurf_err[next_index]

        usurf_line = self.average_elevation_bin(usurf_raster)

        #noise_matrix = self.generate_covariance_matrix(usurf_err_line) #TODO
        noise_matrix = np.eye(len(usurf_line))*10

        noise_samples = np.random.multivariate_normal(np.zeros_like(usurf_line),
                                                      noise_matrix, size=num_samples)

        return year, usurf_line, noise_matrix, noise_samples

    def average_elevation_bin(self, usurf):
        # Apply mask to both surfaces

        usurf_masked = usurf[self.icemask==1]

        # Compute average 2020 surface for each bin
        average_usurf = []
        for i in range(1, len(self.bin_edges)):
            # Mask for pixels in the current bin
            in_bin = self.bin_indices == i
            if np.any(in_bin):
                avg = np.mean(usurf_masked[in_bin])

            average_usurf.append(avg)

        # Convert results to a NumPy array
        return np.array(average_usurf)

    def get_observables_from_ensemble(self, EnKF_object):

        ensemble_usurf = EnKF_object.ensemble_usurf
        observables = []
        for usurf in ensemble_usurf:
            observables.append(self.average_elevation_bin(usurf))

        return np.array(observables)

    def get_new_geometrie(self, year):
        index = np.where(self.time_period == year)[0][0]
        return self.usurf[index]
