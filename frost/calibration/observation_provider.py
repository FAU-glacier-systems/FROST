#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file
import numpy as np
import os
from netCDF4 import Dataset

# Spatial correlation of dh/dt errors (Hugonnet et al., 2021): sum of
# exponential models with these ranges (m) and partial sills (sum 1)
VARIOGRAM_RANGES = [150, 2000, 5000, 20000, 50000, 500000]
VARIOGRAM_SILLS = [0.47741896, 0.34238422, 0.06662273, 0.06900394,
                   0.01602816, 0.02854199]


def error_correlation(d):
    """
    Correlation of the dh/dt errors of two pixels at distance d (m):
    about 50% at 150 m, 20% at 2 km, a few percent at regional scales.
    """
    d = np.asarray(d, dtype=np.float64)
    return 1 - sum(sill * (1 - np.exp(-3 * d / r))
                   for r, sill in zip(VARIOGRAM_RANGES, VARIOGRAM_SILLS))


class ObservationProvider:
    """
    Observed elevation change for the EnKF, aggregated to elevation bands.

    observations.nc holds the raw data on the model grid: dhdt and dhdt_err
    (m/yr, the rate over the period time[0]..time[1] and its 1-sigma error;
    NaN where there is no data), the surface at the start of the period
    (usurf), topg, icemask and velsurf_mag (for plots).

    The observable is the mean dhdt of each elevation band over the pixels
    with data; bands are 'elevation_step' wide in the start surface, and
    bands without data are left out. Its error covariance follows from the
    pixel errors and their spatial correlation by averaging:
        C = A S A^T + model_error^2 I,  S_pq = dhdt_err_p * dhdt_err_q * rho(d_pq)
    with A the band-averaging matrix. The model error is added per band, so
    averaging over many pixels does not shrink it. The model counterpart is the band mean
    of (usurf_end - usurf_start) / period over the same pixels.

    Authors: Oskar Herrmann

    Args:
        rgi_id_dir (str)          - glacier directory with observations.nc
        elevation_step (int)      - band width (m)
        obs_uncertainty (float)   - synthetic runs: factor on dhdt_err
        synthetic (bool)          - synthetic observations
        model_error (float)       - 1-sigma model error of the band-mean
                                    dh/dt (m/yr)
    """

    def __init__(self, rgi_id_dir, rgi_id, elevation_step, obs_uncertainty,
                 synthetic, model_error=0.0):
        observation_file = os.path.join(rgi_id_dir, 'observations.nc')
        with Dataset(observation_file, 'r') as ds:
            self.usurf = np.array(ds['usurf'][:], dtype=np.float64)
            self.topg = np.array(ds['topg'][:], dtype=np.float64)
            self.icemask = np.array(ds['icemask'][:]) > 0.5
            self.dhdt = np.array(ds['dhdt'][:].filled(np.nan), dtype=np.float64)
            self.dhdt_err = np.array(ds['dhdt_err'][:].filled(np.nan),
                                     dtype=np.float64)
            self.velsurf_mag = np.array(ds['velsurf_mag'][:])
            self.time_period = np.array(ds['time'][:]).astype(int)
            self.x = np.array(ds['x'][:])
            self.y = np.array(ds['y'][:])

        self.resolution = int(self.x[1] - self.x[0])
        self.period = self.time_period[-1] - self.time_period[0]
        self.elevation_step = elevation_step
        self.obs_uncertainty = obs_uncertainty
        self.synthetic = synthetic
        if synthetic and obs_uncertainty is not None:
            self.dhdt_err = self.dhdt_err * obs_uncertainty

        # Elevation bands of the start surface
        usurf_ice = self.usurf[self.icemask]
        min_elev = np.floor(usurf_ice.min() / elevation_step) * elevation_step
        max_elev = np.ceil(usurf_ice.max() / elevation_step) * elevation_step
        self.bin_edges = np.arange(min_elev, max_elev + elevation_step,
                                   elevation_step)
        bin_index = np.digitize(self.usurf, self.bin_edges)

        # Observed pixels: ice with data
        self.valid = (self.icemask & np.isfinite(self.dhdt)
                      & np.isfinite(self.dhdt_err) & (self.dhdt_err > 0))
        # Bands with data, numbered 1..num_bins in bin_map (0: not observed)
        observed_bins = np.unique(bin_index[self.valid])
        self.bin_map = np.zeros(self.usurf.shape, dtype=int)
        for number, bin_id in enumerate(observed_bins, start=1):
            self.bin_map[self.valid & (bin_index == bin_id)] = number
        self.num_bins = len(observed_bins)

        # Band-averaging matrix A over the valid pixels
        rows, cols = np.nonzero(self.valid)
        self.pixel_band = self.bin_map[rows, cols] - 1
        counts = np.bincount(self.pixel_band, minlength=self.num_bins)
        self.averaging = np.zeros((self.num_bins, len(rows)))
        self.averaging[self.pixel_band, np.arange(len(rows))] = \
            1.0 / counts[self.pixel_band]

        self.observation = self.band_mean(self.dhdt)
        self.model_error = model_error
        self.covariance = (self.band_covariance(self.dhdt_err[self.valid])
                           + model_error ** 2 * np.eye(self.num_bins))

    def band_mean(self, field):
        """Mean of a field over the valid pixels of every band."""
        return self.averaging @ field[self.valid]

    def band_covariance(self, sigma):
        """C = A S A^T for pixel errors sigma. The pixels lie on a regular
        grid and rho depends only on the lag, so S w is a convolution of the
        map w with rho: C_bc = <w_b, rho * w_c> with w_c = sigma a_c, by FFT
        in O(bands N log N) instead of O(pixels^2)."""
        rows, cols = np.nonzero(self.valid)
        ny, nx = self.valid.shape
        dx, dy = abs(self.x[1] - self.x[0]), abs(self.y[1] - self.y[0])
        # rho on all lags, wrapped so that lag 0 is at [0, 0]; 2n per axis
        # avoids circular overlap
        lag_y = np.fft.fftfreq(2 * ny, 1 / (2 * ny)) * dy
        lag_x = np.fft.fftfreq(2 * nx, 1 / (2 * nx)) * dx
        kernel = error_correlation(np.hypot(lag_y[:, None], lag_x[None, :]))
        kernel_hat = np.fft.rfft2(kernel)
        weights = np.zeros((self.num_bins, ny, nx))
        weights[self.pixel_band, rows, cols] = (
            sigma * self.averaging[self.pixel_band, np.arange(len(rows))])
        covariance = np.empty((self.num_bins, self.num_bins))
        for c in range(self.num_bins):
            smoothed = np.fft.irfft2(
                np.fft.rfft2(weights[c], s=kernel.shape) * kernel_hat,
                s=kernel.shape)[:ny, :nx]
            covariance[:, c] = weights.reshape(self.num_bins, -1) @ \
                smoothed.ravel()
        # symmetric up to rounding
        return (covariance + covariance.T) / 2

    def get_next_observation(self, current_year, num_samples):
        """Band-mean dh/dt over the period, its covariance, noise samples
        drawn from it, and the dh/dt and velocity maps (for plots)."""
        noise_samples = np.random.multivariate_normal(
            np.zeros(self.num_bins), self.covariance, size=num_samples)
        return (int(self.time_period[-1]), self.observation, self.covariance,
                noise_samples, self.dhdt, self.velsurf_mag)

    def initial_usurf(self, num_samples):
        """Start year and the start surface for every member."""
        ensemble_usurf = np.repeat(self.usurf[None], num_samples, axis=0)
        return int(self.time_period[0]), ensemble_usurf

    def get_ensemble_observables(self, EnKF_object):
        """Band-mean modelled dh/dt of every member over the period."""
        return np.array([self.band_mean((usurf - self.usurf) / self.period)
                         for usurf in EnKF_object.ensemble_usurf])

    def vector_to_map(self, values):
        """Map of per-band values (NaN outside the observed pixels)."""
        mapped = np.full(self.bin_map.shape, np.nan, dtype=np.float32)
        observed = self.bin_map > 0
        mapped[observed] = np.asarray(values)[self.bin_map[observed] - 1]
        return mapped
