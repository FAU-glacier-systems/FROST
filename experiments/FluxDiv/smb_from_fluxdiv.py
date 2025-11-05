import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import copy
import frost.calibration.observation_provider as obs_provider

def main():
    # Path to the NetCDF file containing glacier observation data
    rgi_id = "RGI2000-v7.0-G-11-01706"
    geology_optimized_file = os.path.join('..', '..', 'Data', 'Glaciers', rgi_id,
                                          'Inversion', 'geology-optimized.nc')

    observation_file = os.path.join('..', '..', 'Data', 'Glaciers', rgi_id,
                                    'observations.nc')


    # Load important values from the observation file
    with Dataset(geology_optimized_file, 'r') as ds:
        divflux = ds['divflux'][:]

    with Dataset(observation_file, 'r') as ds:
        dhdt = ds['dhdt'][:][1]
        dhdt_err = ds['dhdt_err'][:][1]
        icemask = ds['icemask'][:][1]
        usurf = ds['usurf'][:][0]
        x = ds['x'][:]
        y = ds['y'][:]

    num_samples = 5
    ensemble_dhdt = sample_dhdt(dhdt, dhdt_err, icemask, x ,  y, num_samples)

    elevation_step = 50
    usurf_masked = usurf[icemask == 1]
    # Define bin edges based on a fixed elevation step size
    min_elev = np.floor(
        usurf_masked.min() / elevation_step) * elevation_step
    max_elev = np.ceil(
        usurf_masked.max() / elevation_step) * elevation_step
    bin_edges = np.arange(min_elev, max_elev + elevation_step,
                               elevation_step)

    # Compute bin indices for the surface elevation of 2000
    bin_map = np.digitize(usurf, bin_edges)
    bin_map[icemask == 0] = 0  # Mask out non-glacier areas

    # smb into bins

    smb_maps = []
    for dhdt in ensemble_dhdt:
        bin_smb = average_elevation_bin(len(bin_edges), bin_map, dhdt + divflux)

        bin_smb_map = np.full_like(bin_map, np.nan, dtype=np.float32)
        for bin_id, value in enumerate(bin_smb, start=1):
            bin_smb_map[bin_map == bin_id] = value
        bin_smb_map[bin_map == 0] = np.nan
        smb_maps.append(bin_smb_map)

    show_ensemble(smb_maps)


def average_elevation_bin(num_bins, bin_map, raster):
    # Apply mask to both surfaces

    # Compute average 2020 surface for each bin
    average_bin_value= []
    for i in range(1, num_bins + 1):
        # Mask for pixels in the current bin
        bin_pixels = raster[bin_map == i]
        avg = np.mean(bin_pixels)
        average_bin_value.append(avg)

    # Convert results to a NumPy array
    return np.array(average_bin_value)

def show_ensemble(ensemble):
    num_samples = len(ensemble)
    fig, axes = plt.subplots(1, num_samples,
                             figsize=(5 * num_samples, 5))  # Adjust grid if needed

    # If only one sample, `axes` is not iterable, handle it
    if num_samples == 1:
        axes = [axes]

    # Plot each sample
    for i, ax in enumerate(axes):
        ax.imshow(ensemble[i] , origin='lower', cmap='RdBu', vmin=-10,
                  vmax=10)
        ax.set_title(f"Sample {i + 1}")
        ax.axis("off")  # Hide axes for better visualization

    plt.tight_layout()
    plt.savefig("dhdt_samples.png")



def sample_dhdt(dhdt, dhdt_err, icemask, x, y, num_samples):

    # get data of next year
    dhdt_err_masked = dhdt_err[icemask == 1]

    index_x, index_y = np.where(icemask)
    loc_x, loc_y = y[index_x], x[index_y]
    locations = np.column_stack((loc_x, loc_y))

    # Compute pairwise distances between all points in bin1 and bin2
    distances = np.linalg.norm(
        locations[:, None, :] - locations[None, :, :], axis=2
    )  # Shape: (n1, n2)

    # Apply the correlation function to each distance
    variogram_model = obs_provider.Variogram_hugonnet()
    correlations = variogram_model.cor(distances)
    pixel_uncertainties = dhdt_err_masked[:, np.newaxis] * dhdt_err_masked[
                                                           np.newaxis, :]
    covariance_matrix = correlations * pixel_uncertainties

    noise_samples = np.random.multivariate_normal(
        np.zeros_like(dhdt_err_masked),
        covariance_matrix, size=num_samples)

    ensemble_dhdt = np.empty((num_samples,) + dhdt.shape)
    for e, noise_sample in enumerate(noise_samples):
        dhdt_sample = copy.deepcopy(dhdt)
        dhdt_sample[icemask == 1] += noise_sample
        ensemble_dhdt[e] = dhdt_sample

    return ensemble_dhdt


if __name__ == '__main__':
    main()
