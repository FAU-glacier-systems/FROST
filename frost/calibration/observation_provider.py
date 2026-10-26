#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file
import numpy as np
import os
from netCDF4 import Dataset

# Glacier-wide elevation change of every RGI7 glacier (Hugonnet et al.,
# 2021), 5, 10 and 20 year periods, with its 1-sigma error; OGGM's copy
GEODETIC_RATES_URL = ('https://cluster.klima.uni-bremen.de/~oggm/geodetic_ref_mb/'
                      'rgi7_per_glacier_5yr_10yr_20yr_rates.csv')


def glacier_wide_rate(rgi_id, start_year, end_year):
    """Glacier-wide dh/dt (m/yr) and its 1-sigma error of an RGI7 glacier
    from Hugonnet et al. (2021) for start_year-01-01 to end_year-01-01."""
    from oggm import utils

    path = utils.file_downloader(GEODETIC_RATES_URL)
    if path is None:
        raise RuntimeError(f'Download failed: {GEODETIC_RATES_URL}')
    key = f'{rgi_id},{start_year}-01-01_{end_year}-01-01,'
    with open(path) as f:  # 185 MB: scan the lines instead of parsing all
        for line in f:
            if line.startswith(key):
                # rgiid, period, area, dhdt, err_dhdt, dmdt, err_dmdt
                fields = line.strip().split(',')
                return float(fields[3]), float(fields[4])
    raise ValueError(f'No glacier-wide dh/dt for {rgi_id}, '
                     f'{start_year}-{end_year} in {path}')

# Spatial correlation of the elevation errors (Hugonnet et al., 2021,
# Supplementary Eqs. S13-S15): sum of exponential models with these ranges
# (m). The partial sills are absolute variances (m^2) of the elevation at one
# epoch, fitted against ICESat and averaged over the global distribution of
# time lags to the closest ASTER observation (R. Hugonnet, pers. comm.,
# Nov. 2024; normalised they give 0.477, 0.342, 0.067, 0.069, 0.016, 0.029).
# Only the 150 m component scales with the pixel error dhdt_err (S15); the
# longer ranges do not, they are the same for every pixel.
VARIOGRAM_RANGES = [150, 2000, 5000, 20000, 50000, 500000]
ELEVATION_SILLS = [11.1026, 7.9623, 1.5493, 1.6047, 0.3727, 0.6638]


def short_range_correlation(d):
    """Correlation of the pixel errors at distance d (m), 150 m range."""
    return np.exp(-3 * np.asarray(d, dtype=np.float64) / VARIOGRAM_RANGES[0])


def long_range_covariance(d, period):
    """
    Covariance (m^2/yr^2) of the dh/dt errors of two pixels at distance d (m)
    from the ranges of 2 km and more, for a rate over 'period' years: the
    elevation errors of the start and end epoch are independent, so the
    rate variance is 2 * sill / period^2.
    """
    d = np.asarray(d, dtype=np.float64)
    return sum(2 * sill / period ** 2 * np.exp(-3 * d / r)
               for r, sill in zip(VARIOGRAM_RANGES[1:], ELEVATION_SILLS[1:]))


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
    pixel errors and their spatial correlation by averaging, as in Hugonnet
    et al. (2021, Supplementary Eqs. S13-S15):
        C = A S A^T + model_error^2 I
        S_pq = s0_p s0_q rho_150(d_pq) + L(d_pq)
    with A the band-averaging matrix, L the long-range covariance of the
    ranges >= 2 km (absolute sills, the same for every pixel), and
    s0^2 = dhdt_err^2 - L(0) the pixel error left for the 150 m range. Most
    of dhdt_err is short-range noise that averages out over a band; scaling
    the long ranges with dhdt_err as well (as before) made the glacier-mean
    error 4-8 times larger than the published glacier-wide error. The model
    error is added per band, so averaging over many pixels does not shrink
    it. The model counterpart is the band mean of
    (usurf_end - usurf_start) / period over the same pixels.

    With glacier_wide_dhdt, the glacier-wide dh/dt of Hugonnet et al. (2021)
    for the same period is appended as a last observable, with its published
    1-sigma error. It comes from the same DEMs as the bands and its error is
    taken as independent of theirs, so it counts the glacier mean twice; with
    the covariance above the bands already give a glacier-mean error close
    to the published one. The model counterpart is the mean modelled dh/dt
    over the start ice mask.

    Authors: Oskar Herrmann

    Args:
        rgi_id_dir (str)          - glacier directory with observations.nc
        elevation_step (int)      - band width (m)
        obs_uncertainty (float)   - synthetic runs: factor on dhdt_err
        synthetic (bool)          - synthetic observations
        model_error (float)       - 1-sigma model error of the band-mean
                                    dh/dt (m/yr)
        glacier_wide_dhdt (bool)  - append the glacier-wide dh/dt of
                                    Hugonnet et al. (2021) as an observable
    """

    def __init__(self, rgi_id_dir, rgi_id, elevation_step, obs_uncertainty,
                 synthetic, model_error=0.0, glacier_wide_dhdt=False):
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
        self.data_covariance = self.band_covariance(self.dhdt_err[self.valid])
        self.covariance = (self.data_covariance
                           + model_error ** 2 * np.eye(self.num_bins))

        # Glacier-wide dh/dt as a last observable (bands stay 0..num_bins-1)
        self.glacier_wide_dhdt = glacier_wide_dhdt
        if glacier_wide_dhdt:
            if synthetic:
                raise ValueError('glacier_wide_dhdt needs real observations')
            rate, error = glacier_wide_rate(rgi_id, self.time_period[0],
                                            self.time_period[-1])
            if not (np.isfinite(rate) and np.isfinite(error) and error > 0):
                # 482 of 4079 region-11 glaciers have no value (NaN)
                print(f'No glacier-wide dh/dt for {rgi_id} in the Hugonnet '
                      'table; calibrating on the bands only')
                self.glacier_wide_dhdt = glacier_wide_dhdt = False
        if glacier_wide_dhdt:
            self.glacier_wide = (rate, error)
            self.observation = np.append(self.observation, rate)
            covariance = np.zeros((self.num_bins + 1, self.num_bins + 1))
            covariance[:-1, :-1] = self.covariance
            covariance[-1, -1] = error ** 2
            self.covariance = covariance
            print(f'Glacier-wide dh/dt (Hugonnet et al. 2021): '
                  f'{rate:.2f} +- {error:.2f} m/yr; band covariance gives '
                  f'+- {self.glacier_mean_sigma():.2f}')

    def glacier_mean_sigma(self):
        """1-sigma of the area-weighted mean of the band observations
        (data error only, without the model error)."""
        counts = np.bincount(self.pixel_band, minlength=self.num_bins)
        weights = counts / counts.sum()
        return float(np.sqrt(weights @ self.data_covariance @ weights))

    def band_mean(self, field):
        """Mean of a field over the valid pixels of every band."""
        return self.averaging @ field[self.valid]

    def band_covariance(self, sigma):
        """C = A S A^T for pixel errors sigma (see the class docstring): the
        150 m part with the pixel errors left after the long ranges, plus
        the long ranges with their absolute sills."""
        long_range_variance = long_range_covariance(0.0, self.period)
        sigma_short = np.sqrt(np.clip(sigma ** 2 - long_range_variance, 0, None))
        return (self.averaged_covariance(sigma_short, short_range_correlation)
                + self.averaged_covariance(
                    np.ones_like(sigma),
                    lambda d: long_range_covariance(d, self.period)))

    def averaged_covariance(self, sigma, covariance_function):
        """A S A^T with S_pq = sigma_p sigma_q k(d_pq). The pixels lie on a
        regular grid and k depends only on the lag, so S w is a convolution
        of the map w with k: C_bc = <w_b, k * w_c> with w_c = sigma a_c, by
        FFT in O(bands N log N) instead of O(pixels^2)."""
        rows, cols = np.nonzero(self.valid)
        ny, nx = self.valid.shape
        dx, dy = abs(self.x[1] - self.x[0]), abs(self.y[1] - self.y[0])
        # k on all lags, wrapped so that lag 0 is at [0, 0]; 2n per axis
        # avoids circular overlap
        lag_y = np.fft.fftfreq(2 * ny, 1 / (2 * ny)) * dy
        lag_x = np.fft.fftfreq(2 * nx, 1 / (2 * nx)) * dx
        kernel = covariance_function(np.hypot(lag_y[:, None], lag_x[None, :]))
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
            np.zeros(len(self.observation)), self.covariance, size=num_samples)
        return (int(self.time_period[-1]), self.observation, self.covariance,
                noise_samples, self.dhdt, self.velsurf_mag)

    def initial_usurf(self, num_samples):
        """Start year and the start surface for every member."""
        ensemble_usurf = np.repeat(self.usurf[None], num_samples, axis=0)
        return int(self.time_period[0]), ensemble_usurf

    def get_ensemble_observables(self, EnKF_object):
        """Band-mean modelled dh/dt of every member over the period, and
        with glacier_wide_dhdt its mean over the start ice mask."""
        observables = []
        for usurf in EnKF_object.ensemble_usurf:
            dhdt = (usurf - self.usurf) / self.period
            bands = self.band_mean(dhdt)
            if self.glacier_wide_dhdt:
                bands = np.append(bands, np.mean(dhdt[self.icemask]))
            observables.append(bands)
        return np.array(observables)

    def melted_out_fraction(self, EnKF_object, min_thk=1.0):
        """Share of the members in which a band has melted out: most of its
        observed pixels hold less than min_thk (m) of ice at the end of the
        period. There the modelled dh/dt is -thickness / period whatever
        the SMB, so the band carries no information on the parameters."""
        observed = self.bin_map > 0
        melted = []
        for usurf in EnKF_object.ensemble_usurf:
            ice_free = (usurf - self.topg)[observed] < min_thk
            share = np.bincount(self.bin_map[observed] - 1, weights=ice_free,
                                minlength=self.num_bins) / np.bincount(
                self.bin_map[observed] - 1, minlength=self.num_bins)
            melted.append(share > 0.5)
        return np.mean(melted, axis=0)

    def vector_to_map(self, values):
        """Map of per-band values (NaN outside the observed pixels)."""
        mapped = np.full(self.bin_map.shape, np.nan, dtype=np.float32)
        observed = self.bin_map > 0
        mapped[observed] = np.asarray(values)[self.bin_map[observed] - 1]
        return mapped
