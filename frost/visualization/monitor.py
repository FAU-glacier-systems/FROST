#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
import copy
import pyproj
from itertools import product, accumulate
import json
plt.rcParams["font.family"] = "monospace"


class Monitor:
    def __init__(self, EnKF_object, ObsProvider, max_iterations, output_dir,
                 synthetic, dark=True, plots='latest'):

        self.rgi_id = EnKF_object.rgi_id
        self.smb_model = EnKF_object.smb_model
        self.ensemble_size = EnKF_object.ensemble_size
        self.seed = EnKF_object.seed
        self.icemask_init = EnKF_object.icemask_init
        # self.plot_points = ObsProvider.observation_locations
        self.bin_map = ObsProvider.bin_map
        self.obs_provider = ObsProvider
        self.time_period = ObsProvider.time_period
        self.start_year = self.time_period[0]
        self.resolution = ObsProvider.resolution
        self.time_period_repeat = np.repeat(self.time_period, 2)[1:]
        self.reference_smb = EnKF_object.reference_smb
        self.reference_variability = EnKF_object.reference_variability
        self.max_iterations = max_iterations
        self.max_iteration_axis = range(max_iterations + 1)
        self.synthetic = synthetic
        self.colorscale = plt.get_cmap('tab20')

        # plots: 'all' writes one png per iteration, 'latest' overwrites one
        # png, 'final' plots only the last iteration (max_iterations)
        if plots not in ('all', 'latest', 'final'):
            raise ValueError(f'Unknown Monitor plots option {plots}')
        self.plots = plots

        # dark: black background with white text, else white with black
        self.dark = dark
        self.fg = 'white' if dark else 'black'
        self.bg = 'black' if dark else 'white'

        # Maps show the glacier's bounding box plus a margin of ice-free
        # cells, in km from its lower-left corner
        rows, cols = np.nonzero(self.icemask_init)
        margin = 3
        y0 = max(rows.min() - margin, 0)
        y1 = min(rows.max() + margin + 1, self.icemask_init.shape[0])
        x0 = max(cols.min() - margin, 0)
        x1 = min(cols.max() + margin + 1, self.icemask_init.shape[1])
        self.crop = (slice(y0, y1), slice(x0, x1))
        self.extent_km = [0, (x1 - x0) * self.resolution / 1000,
                          0, (y1 - y0) * self.resolution / 1000]

        self.monitor_dir = os.path.join(output_dir, 'Monitor')
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)

        self.keys = ['mean_usurf', 'point1', 'point2']

        self.ensemble_observables_log = {key: [[] for _ in range(self.ensemble_size)]
                                         for key in self.keys}

        self.observation_log = {key: [] for key in self.keys}

        self.observation_std_log = {key: [] for key in self.keys}

        if synthetic:
            if str(self.smb_model) == "ELA":
                self.density_factor = {'ela': 1,
                                       'abl_grad': 1,
                                       'acc_grad': 1,
                                       }
            elif str(self.smb_model) == "TI":
                self.density_factor = {'melt_f': 1,
                                       'prcp_fac': 1,
                                       'temp_bias': 1}

        else:
            if str(self.smb_model) == "ELA":
                self.density_factor = {'ela': 1,
                                       'abl_grad': 0.91,
                                       'acc_grad': 0.55,
                                       }
            elif str(self.smb_model) == "TI":
                self.density_factor = {'melt_f': 1,
                                       'prcp_fac': 1,
                                       'temp_bias': 1}

        # Observables: band-mean elevation change over the period
        period = f'{self.time_period[0]}-{self.time_period[-1]}'
        self.plot_style = dict(
            mean_usurf=dict(y_label=f'Mean elevation change\n{period} '
                                    '(m a$^{-1}$)'),
            point1=dict(y_label='Elevation change of third\nband from '
                                'bottom (m a$^{-1}$)'),
            point2=dict(y_label='Elevation change of third\nband from top '
                                '(m a$^{-1}$)'),
            ela=dict(y_label='Equilibrium Line\nAltitude (m)'),
            abl_grad=dict(y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
            acc_grad=dict(y_label='Accumulation Gradient\n'
                                  '(m a$^{-1}$ km$^{-1}$)'),
            melt_f=dict(y_label='Melt Factor OGGM\n( mm w.e. / (C day) )'),
            prcp_fac=dict(y_label='Precipitation Factor \n( - )'),
            temp_bias=dict(y_label='Temperature Bias ( C )'),
        )

    def plot_now(self, iteration):
        return self.plots != 'final' or iteration == self.max_iterations

    def png_name(self, name, iteration, year):
        if self.plots == 'all':
            return f"{name}_{iteration:03d}_{year}.png"
        return f"{name}.png"

    def summarise_observables(self, ensemble_observables, new_observables,
                              uncertainty_matrix, noise_samples):
        """Log the mean over all bands and two single bands (third from the
        bottom and from the top) of the observation and every member; the
        observation uncertainty (1 sigma) follows from its covariance."""
        size = ensemble_observables.shape[1]
        band1, band2 = (0, -1) if size < 5 else (2, -3)
        band_std = np.sqrt(np.diagonal(uncertainty_matrix))
        mean_std = np.sqrt(np.sum(uncertainty_matrix)) / size

        for key, obs, std, members in [
                ('mean_usurf', np.mean(new_observables), mean_std,
                 np.mean(ensemble_observables, axis=1)),
                ('point1', new_observables[band1], band_std[band1],
                 ensemble_observables[:, band1]),
                ('point2', new_observables[band2], band_std[band2],
                 ensemble_observables[:, band2])]:
            self.observation_log[key].append(obs)
            self.observation_std_log[key].append(std)
            for e, value in enumerate(members):
                self.ensemble_observables_log[key][e].append(value)

    def plot_iteration(self, ensemble_smb_log,
                       new_observation, uncertainty, iteration, year,
                       ensemble_observables, noise_samples):

        # the log is kept in every mode, so the final plot shows all iterations
        self.summarise_observables(ensemble_observables, new_observation,
                                   uncertainty, noise_samples)
        if not self.plot_now(iteration):
            return

        fig, ax = plt.subplots(2, 3, figsize=(12, 6), facecolor=self.bg)

        iteration_axis = range(iteration + 1)
        iteration_axis_repeat = np.repeat(iteration_axis, 2)[1:-1]

        # Plot observables
        from matplotlib.ticker import MaxNLocator

        def set_axis_style(ax, show_x):
            ax.set_ylabel(self.plot_style[key]['y_label'], color=self.fg)
            if show_x:
                ax.set_xlabel("Iteration", color=self.fg)
            ax.set_xlim(-0.2, self.max_iterations + 0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(axis="y", color="lightgray", linestyle="-", zorder=0)
            ax.grid(axis="x", color="lightgray", linestyle="-", zorder=0)
            ax.xaxis.set_tick_params(bottom=False, colors=self.fg)
            ax.yaxis.set_tick_params(left=False, colors=self.fg)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_facecolor(self.bg)

        for i, key in enumerate(self.ensemble_observables_log.keys()):

            for e in range(self.ensemble_size):
                observable_log_values = self.ensemble_observables_log[key][e]
                observable_log_repeat = np.repeat(observable_log_values, 2)
                ax[0, i].plot(iteration_axis_repeat,
                              observable_log_repeat,
                              color=self.colorscale(5), marker='o', markersize=10,
                              markevery=[-1], zorder=2, label='Ensemble Member')

            observable_log_values = self.observation_log[key]
            observation_log_repeat = np.repeat(observable_log_values, 2)
            if self.synthetic:
                label = 'Observation [Synthetic]'
            else:
                label = 'Observation [Hugonnet21]'
            ax[0, i].plot(iteration_axis_repeat,
                          observation_log_repeat,
                          color=self.colorscale(0), marker='o', markersize=10,
                          markevery=[-1], zorder=5, label=label,
                          )

            observable_var_log_values = self.observation_std_log[key]
            observation_var_log_repeat = np.repeat(observable_var_log_values, 2)
            std_plus = observation_log_repeat + observation_var_log_repeat
            std_minus = observation_log_repeat - observation_var_log_repeat

            if self.synthetic:
                label = 'Observation Uncertainty [Synthetic]'
            else:
                label = 'Observation Uncertainty [Hugonnet21]'

            ax[0, i].fill_between(iteration_axis_repeat,
                                  std_minus,
                                  std_plus,
                                  zorder=3,
                                  color=self.colorscale(1), alpha=0.5,
                                  label=label)
            set_axis_style(ax[0, i], show_x=False)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        leg = fig.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper center',
            ncol=4,
            frameon=False  # removes background box
        )

        plt.setp(leg.get_texts(), color=self.fg)
        # Plot surface mass balance parameters
        for i, key in enumerate(ensemble_smb_log.keys()):
            key_smb_log = np.array(ensemble_smb_log[key])
            key_mean_smb = np.mean(key_smb_log, axis=0)
            for e in range(self.ensemble_size):
                smb_log_values = key_smb_log[e]
                # Plot each ensemble member's time series
                ax[1, i].plot(range(len(smb_log_values)),
                              smb_log_values,
                              color='gold', marker='o', markersize=10,
                              markevery=[-1], zorder=2, label='Ensemble Member')
            # ax[1, i].plot(iteration_axis,
            #               key_mean_smb, color='orange', marker='o', markersize=10,
            #               markevery=[-1], zorder=2, label='Ensemble Mean')
            if self.reference_smb is not None:
                referenc_smb_line = np.array([self.reference_smb[key] /
                                              self.density_factor[key] for _ in
                                              range(self.max_iterations + 1)])
                if self.synthetic:
                    label = 'Reference Mean [Synthetic]'
                else:
                    label = 'Reference Mean [GLAMOS]'
                ax[1, i].plot(self.max_iteration_axis,
                              referenc_smb_line,
                              color=self.colorscale(8), zorder=5, label=label
                              )

                std_plus = referenc_smb_line + self.reference_variability[key]
                std_minus = referenc_smb_line - self.reference_variability[key]

                if not self.synthetic:
                    ax[1, i].fill_between(self.max_iteration_axis,
                                          std_minus,
                                          std_plus,
                                          zorder=2,
                                          color=self.colorscale(8), alpha=0.2,
                                          label='Annual Variability [GLAMOS]')

            set_axis_style(ax[1, i], show_x=True)

        handles, labels = ax[1, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        leg = fig.legend(
            by_label.values(),
            by_label.keys(),
            loc='lower center',
            ncol=4,
            frameon=False
        )

        plt.setp(leg.get_texts(), color=self.fg)


        import string
        axes = ax.flatten()  # Flatten for easy iteration

        labels_subplot = [f"{letter})" for letter in
                          string.ascii_lowercase[:len(axes)]]

        for ax, label in zip(axes, labels_subplot):
            # Add label to lower-left corner (relative coordinates)
            ax.text(-0.3, 0.95, label, transform=ax.transAxes,
                    fontsize=12, va='bottom', ha='left', fontweight='bold',
                    color=self.fg)

        # FINISH AND SAVE
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.15)

        fig.savefig(
            os.path.join(self.monitor_dir,
                         self.png_name('status', iteration, year)),
            format='png', facecolor=self.bg, dpi=300)

        plt.close(fig)
        plt.clf()

    def vector_to_map(self, band_values):
        return self.obs_provider.vector_to_map(band_values)

    def plot_glacier_property_map(self, ax, data_map, title, colorlabel,
                                  vmin=-10, vmax=10,
                                  cmap='seismic_r', mask=None):
        """
        Plot a glacier property map (e.g., elevation change, velocity, SMB) on the given Axes.
        """
        if mask is None:
            mask = self.icemask_init == 1

        data_map = data_map.copy()
        data_map[mask == 0] = np.nan

        img = ax.imshow(
            data_map[self.crop],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=self.extent_km,
            zorder=3,
        )

        # Colorbar
        cbar = plt.colorbar(img, ax=ax, orientation="vertical")
        cbar.set_label(colorlabel, color=self.fg)
        cbar.ax.tick_params(colors=self.fg)

        ax.set_facecolor(self.bg)
        ax.tick_params(axis="both", colors=self.fg)
        ax.grid(color=self.fg, linestyle="--", zorder=4, alpha=.3)
        for spine in ax.spines.values():
            spine.set_linewidth(0)

        # Title
        mean_val = np.nanmean(data_map[mask])
        ax.set_title(
            f"{title}\nMean: {mean_val:.2f} m a$^{{-1}}$",
            color=self.fg,
        )

        return mean_val

    def plot_maps_prognostic(self, ensembleKF, obs_dhdt_raster,
                             obs_velsurf_mag_raster, new_observation, noise_samples,
                             ensemble_observables, uncertainty, iteration, year,
                             write_json=True):
        """new_observation, noise_samples and ensemble_observables are
        band-mean elevation changes (m/yr)."""
        if not self.plot_now(iteration):
            return
        # Length of the observation period, for rates in m/yr
        period = year - ensembleKF.start_year


        ###################### MAPS #################################################
        nrows = 2
        ncols = 4
        # panel height from the glacier's aspect ratio (panel width ~2.2 in
        # without colorbar and labels)
        aspect = self.extent_km[3] / self.extent_km[1]
        panel_height = np.clip(2.2 * aspect, 1.5, 5.0)
        fig, ax = plt.subplots(nrows, ncols,
                               figsize=(13, nrows * (panel_height + 1.3)),
                               facecolor=self.bg)

        # bedrock as background
        bedrock = ensembleKF.bedrock[self.crop]
        for row, col in product(range(nrows), range(ncols)):
            ax[row, col].imshow(bedrock, cmap='gray', origin='lower',
                                extent=self.extent_km,
                                vmin=np.percentile(bedrock, 2),
                                vmax=np.percentile(bedrock, 98))

        # mean modelled surface elevation
        surface = np.mean(ensembleKF.ensemble_usurf, axis=0)
        if write_json:
            member_usurf = ensembleKF.ensemble_usurf
            member_thk = member_usurf - ensembleKF.bedrock
            member_mask = member_thk > 0
            member_elevation_change = (member_usurf - ensembleKF.ensemble_init_surf_raster) / period
            member_dhdt_mean = []
            member_dhdt_binned_mean = []
            observation_samples = []

            for elevation_change, mask, member_bands, noise_sample in zip(
                    member_elevation_change, member_mask,
                    ensemble_observables, noise_samples):
                member_dhdt_mean.append(np.nanmean(elevation_change[mask]))

                # binned dhdt
                mapped_difference = self.vector_to_map(member_bands)
                member_dhdt_binned_mean.append(np.nanmean(mapped_difference[mask]))
                noisy_observation = self.vector_to_map(new_observation + noise_sample)
                observation_samples.append(np.nanmean(noisy_observation[mask]))



        # thickness from mean surface and mask.
        # mask should be the same as the maximum extend of glacier in the model
        thk = surface - ensembleKF.bedrock
        new_mask = thk > 0

        # Observed elevation change
        observed_elevation_change = self.plot_glacier_property_map(ax=ax[0, 0],
                                                                   data_map=obs_dhdt_raster,
                                                                   title='Observed\nElevation Change',
                                                                   colorlabel='Elevation Change (m a$^{-1}$)',
                                                                   mask=new_mask)

        # Observed Elevation Changes binned
        new_observation_mapped = self.vector_to_map(new_observation)

        observed_elevation_change_binned = self.plot_glacier_property_map(ax=ax[0, 1],
                                                                          data_map=new_observation_mapped,
                                                                          title='Observed\nElevation Change binned',
                                                                          colorlabel='Elevation Change (m a$^{-1}$)',
                                                                          mask=new_mask)

        # Observed velocities
        obs_velsurf_mag_raster[
            self.icemask_init == 0] = np.nan  # Mask ice-free areas
        observed_velocity = self.plot_glacier_property_map(ax=ax[0, 2],
                                                           data_map=obs_velsurf_mag_raster,
                                                           title='Observed\nSurface Velocity',
                                                           colorlabel='Surface Velocity (m a$^{-1}$)',
                                                           vmin=0,
                                                           vmax=np.nanmax(obs_velsurf_mag_raster),
                                                           cmap='magma')

        # Estimated elevation change (modelled)
        ensemble_usurf = np.mean(ensembleKF.ensemble_usurf, axis=0)
        init_usurf = np.mean(ensembleKF.ensemble_init_surf_raster, axis=0)
        modeled_dhdt = (ensemble_usurf - init_usurf) / period
        modelled_elevation_change = self.plot_glacier_property_map(ax=ax[1, 0],
                                                                   data_map=modeled_dhdt,
                                                                   title='Modelled\nElevation Change',
                                                                   colorlabel='Elevation Change (m a$^{-1}$)',
                                                                   mask=new_mask)

        # Modelled elevation Change (binned)
        ensemble_dhdt_mean_mapped = self.vector_to_map(
            np.mean(ensemble_observables, axis=0))
        modelled_elevation_change_binned = self.plot_glacier_property_map(ax=ax[1, 1],
                                                                          data_map=ensemble_dhdt_mean_mapped,
                                                                          title='Modelled\nElevation Change binned',
                                                                          colorlabel='Elevation Change (m a$^{-1}$)',
                                                                          mask=new_mask)

        # modelled velocity
        modelled_velocity = np.mean(ensembleKF.ensemble_velsurf_mag_raster, axis=0)
        modelled_velocity[self.icemask_init == 0] = np.nan  # Mask ice-free areas
        modelled_velocity_mean = self.plot_glacier_property_map(ax=ax[1, 2],
                                                                data_map=modelled_velocity,
                                                                title='Modelled\nSurface Velocity',
                                                                colorlabel='Surface Velocity (m a$^{-1}$)',
                                                                vmin=0,
                                                                vmax=np.nanmax(obs_velsurf_mag_raster),
                                                                cmap='magma')

        # Surface Mass Balance (SMB) Map
        mean_smb_raster = np.mean(ensembleKF.ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas

        modelled_smb = self.plot_glacier_property_map(ax=ax[1, 3],
                                                      data_map=mean_smb_raster,
                                                      title='Modelled\nSurface Mass Balance',
                                                      colorlabel='Surface Mass Balance (m a$^{'
                                                                 '-1}$)', mask=new_mask)

        # Flux Divergence Map
        mean_smb_raster = np.mean(ensembleKF.ensemble_divflux_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas
        modelled_flux_div = self.plot_glacier_property_map(ax=ax[0, 3],
                                                           data_map=-mean_smb_raster,
                                                           title='Modelled\nFlux Divergence',
                                                           colorlabel='Flux Divergence (m a$^{-1}$)')

        import string
        for axi in ax[1]:
            axi.set_xlabel('km', color=self.fg)
        ax[0, 0].set_ylabel('km', color=self.fg)
        ax[1, 0].set_ylabel('km', color=self.fg)

        axes = ax.flatten()  # Flatten for easy iteration

        labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes)]]

        for a, label in zip(axes, labels_subplot):
            a.text(
                -0.35, 1.01, label,
                transform=a.transAxes,
                fontsize=12,
                va="bottom",
                ha="left",
                fontweight="bold",
                color=self.fg,
            )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                self.monitor_dir,
                self.png_name('maps_prognostic', iteration, year),
            ),
            format="png",
            facecolor=self.bg,
            edgecolor="none", dpi=300
        )

        plt.close(fig)
        plt.clf()

        if write_json:
            def json_safe(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.floating, np.integer)):
                    return obj.item()
                if isinstance(obj, list):
                    return [json_safe(x) for x in obj]
                if isinstance(obj, dict):
                    return {k: json_safe(v) for k, v in obj.items()}
                return obj


            data = {
                "observed_elevation_change": observed_elevation_change,
                "observed_elevation_change_binned": observed_elevation_change_binned,
                "modelled_elevation_change": modelled_elevation_change,
                "modelled_elevation_change_binned": modelled_elevation_change_binned,
                "modelled_flux_divergence": modelled_flux_div,
                "modelled_surface_mass_balance": modelled_smb,
                "modelled_surface_velocity": modelled_velocity_mean,
                "observed_surface_velocity": observed_velocity,
                "ensemble_elevation_change": member_dhdt_mean,
                "ensemble_elevation_change_binned": member_dhdt_binned_mean,
                "observation_samples_binned": observation_samples,
                "ensemble_smb": ensembleKF.ensemble_smb
            }

            # make everything safe before dumping
            data = json_safe(data)

            with open(os.path.join(self.monitor_dir, f"metrics_{iteration:03d}_{year}.json"), "w") as f:
                json.dump(data, f, indent=4)

    def visualise_3d(self, property_map, glacier_surface, bedrock, year, x, y):
        # choose property that is displayed on the glacier surface

        thicknes = glacier_surface - bedrock
        lat_range = x
        lon_range = y
        property_map[thicknes < 0.001] = None

        color_scale = "RdBu"
        max_property_map = np.nanmax(property_map)
        min_property_map = np.nanmin(property_map)

        # make edges equal so that it looks like a volume
        max_bedrock = np.max(bedrock)
        min_bedrock = np.min(bedrock)
        bedrock_border = copy.copy(bedrock)
        bedrock_border[0, :] = min_bedrock
        bedrock_border[-1, :] = min_bedrock
        bedrock_border[:, 0] = min_bedrock
        bedrock_border[:, -1] = min_bedrock

        # create time frames for slider
        glacier_surface[thicknes < 0.001] = None

        glacier_bottom = copy.copy(bedrock)
        glacier_bottom[thicknes < 1] = None

        # create 3D surface plots with property as surface color
        surface_fig = go.Surface(
            z=glacier_surface,
            x=lat_range,
            y=lon_range,
            colorscale=color_scale,
            # cmax=30,
            cmax=5.1,
            # cmin=-30,
            cmin=-5.1,
            surfacecolor=property_map,
            showlegend=False,
            name="glacier surface",
            colorbar=dict(title="Surface Mass Balance (m/a)",
                          titleside="top", thickness=50, orientation="h", y=0.7,
                          len=0.5,
                          titlefont=dict(size=50), tickfont=dict(size=40),
                          tickvals=[-5.1, 5.1], tickformat=".0f"
                          # This limits decimal places to 3
                          ),
            showscale=True,
        )

        # create 3D bedrock plots
        bedrock_fig = go.Surface(
            z=bedrock_border,
            x=lat_range,
            y=lon_range,
            colorscale='gray',
            opacity=1,
            showlegend=False,
            name="bedrock",
            cmax=max_bedrock,
            cmin=0,
            colorbar=dict(title="Bedrock Elevation (m)", titleside="top",
                          thickness=50, orientation="h", y=0.7, len=0.5,
                          titlefont=dict(size=50), tickfont=dict(size=40),
                          tickvals=[int(0), int(max_bedrock)]),
            showscale=False,
        )

        # compute aspect ratio of the base
        resolution = int(lat_range[1] - lat_range[0])
        ratio_y = bedrock.shape[0] / bedrock.shape[1]
        ratio_z = (max_bedrock - min_bedrock) / (bedrock.shape[0] * resolution)
        ratio_z *= 2  # emphasize z-axis to make mountians look twice as steep

        # # transform angle[0-180] into values between [0, 1] for camera postion
        # radians = math.radians(camera_angle - 180)
        # camera_x = math.sin(-radians) - 1
        # camera_y = math.cos(-radians) - 1

        # transform angle[0-180] into values between [0, 1] for camera postion
        # theta = 2 * math.pi * camera_angle / 100
        camera_x = 0
        camera_y = -2

        print(camera_x, camera_y)
        # Define the UTM projection (UTM zone 32N)
        utm_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')

        # Define the WGS84 projection
        wgs84_proj = pyproj.Proj(proj='latlong', datum='WGS84')

        # Example coordinate in UTM zone 32N (replace these values with your coordinates)
        utm_easting = lat_range  # example easting value
        utm_northing = lon_range  # example northing value

        # Reproject the coordinate
        lon_x, lat_x = pyproj.transform(utm_proj, wgs84_proj, utm_easting,
                                        np.ones_like(utm_easting) * utm_northing[0])
        lon_y, lat_y = pyproj.transform(utm_proj, wgs84_proj,
                                        np.ones_like(utm_northing) * utm_easting[0],
                                        utm_northing)

        # Output the WGS84 coordinate

        fig_dict = dict(
            data=[surface_fig, bedrock_fig],

            layout=dict(  # width=1800,
                height=800,
                margin=dict(l=0, r=0, t=30, b=0),
                title="title",
                font=dict(family="monospace", size=20),
                legend={"orientation": "h", "yanchor": "bottom", "xanchor": "left"},
                scene=dict(
                    zaxis=dict(showbackground=False, showticklabels=False, title="",
                               showgrid=False,  # Remove grid lines
                               zeroline=False,  # Remove axis zero line
                               showline=False,  # Remove axis line
                               ),
                    xaxis=dict(
                        showbackground=False,
                        showticklabels=True,
                        showgrid=False,  # Remove grid lines
                        zeroline=False,  # Remove axis zero line
                        showline=False,  # Remove axis line
                        visible=False,
                        range=[lat_range[0], lat_range[-1]],
                        tickvals=[ticks for ticks in lat_range[::42]],
                        ticktext=["%.2fE" % ticks for ticks in lon_x[::42]],

                        title="Longitude",

                    ),
                    yaxis=dict(
                        showbackground=False,
                        showticklabels=True,
                        showgrid=False,  # Remove grid lines
                        zeroline=False,  # Remove axis zero line
                        showline=False,  # Remove axis line
                        visible=False,
                        range=[lon_range[0], lon_range[-1]],
                        title="Latitude",
                        tickvals=[ticks for ticks in lon_range[::42]],
                        ticktext=["%.2fN" % ticks for ticks in lat_y[::42]],

                    ),
                ),
                scene_aspectratio=dict(x=1, y=ratio_y, z=ratio_z),
                scene_camera_eye=dict(x=camera_x, y=camera_y, z=1),
                scene_camera_center=dict(x=0, y=0, z=0),

            ),
        )
        # create figure
        fig = go.Figure(fig_dict)
        fig.update_layout(
            title={'text': str(year), 'font': {'size': 50}, 'x': 0.5,
                   'y': 0.1},
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',  # Make outer background transparent
            plot_bgcolor='rgba(0,0,0,0)'  # Make inner plot background transparent
        )

        os.makedirs("Plots/aletsch", exist_ok=True)
        fig.write_image(f"Plots/aletsch/glacier_surface_{year}.png", width=1500,
                        height=1200, scale=0.75)
