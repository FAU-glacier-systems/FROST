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


class Monitor:
    def __init__(self, EnKF_object, ObsProvider, max_iterations, output_dir,
                 synthetic, binned_usurf_init, plot_dhdt):

        self.rgi_id = EnKF_object.rgi_id
        self.smb_model = EnKF_object.smb_model
        self.ensemble_size = EnKF_object.ensemble_size
        self.seed = EnKF_object.seed
        self.icemask_init = EnKF_object.icemask_init
        # self.plot_points = ObsProvider.observation_locations
        self.bin_map = ObsProvider.bin_map
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
        self.binned_usurf_init = binned_usurf_init
        self.plot_dhdt = plot_dhdt

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

        if self.plot_dhdt:
            if str(self.smb_model) == "ELA":
                self.plot_style = dict(
                    mean_usurf=dict(y_label='Mean surface elevation \n change 2000- '
                                            '2019 ('
                                            'm)'),
                    point1=dict(
                        y_label='Mean surface elevation change\nof fifth bin from front ('
                                'm)'),
                    point2=dict(
                        y_label=f'Mean surface elevation change\nof fifth bin from '
                                f'top ('
                                f'm)'),
                    ela=dict(y_label='Equilibrium Line\nAltitude (m)'),
                    abl_grad=dict(
                        y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                    acc_grad=dict(
                        y_label='Accumulation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                )
            elif str(self.smb_model) == "TI":
                self.plot_style = dict(
                    mean_usurf=dict(y_label='Mean surface elevation \n change 2000- '
                                            '2019 ('
                                            'm)'),
                    point1=dict(
                        y_label='Mean surface elevation change\nof fifth bin from front ('
                                'm)'),
                    point2=dict(
                        y_label=f'Mean surface elevation change\nof fifth bin from '
                                f'top ('
                                f'm)'),
                    melt_f=dict(y_label='Melt Factor OGGM\n( mm w.e. / (C day) )'),
                    prcp_fac=dict(y_label='Precipitation Factor \n( - )'),
                    temp_bias=dict(y_label='Temperature Bias ( C )'),
                )

        else:
            self.plot_style = dict(
                mean_usurf=dict(y_label='Mean surface elevation \nin 2019 (m)'),
                point1=dict(
                    y_label='Mean surface elevation\nof fifth bin from front ('
                            'm)'),
                point2=dict(y_label=f'Mean surface elevation\nof fifth bin from '
                                    f'top ('
                                    f'm)'),
                # start JJF
                melt_f=dict(y_label='Melt Factor OGGM\n( mm w.e. / (C day) )'),
                prcp_fac=dict(y_label='Precipitation Factor \n( - )'),
                temp_bias=dict(y_label='Temperature Bias ( C )'),
                ela=dict(y_label='Equilibrium Line\nAltitude (m)'),
                abl_grad=dict(y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                acc_grad=dict(
                    y_label='Accumulation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                # end JJF
            )
            if str(self.smb_model) == "ELA":
                self.plot_style = dict(
                    mean_usurf=dict(y_label='Mean surface elevation \n change 2000- '
                                            '2019 ('
                                            'm)'),
                    point1=dict(
                        y_label='Mean surface elevation\nof fifth bin from front ('
                                'm)'),
                    point2=dict(y_label=f'Mean surface elevation\nof fifth bin from '
                                        f'top ('
                                        f'm)'),
                    ela=dict(y_label='Equilibrium Line\nAltitude (m)'),
                    abl_grad=dict(
                        y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                    acc_grad=dict(
                        y_label='Accumulation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
                )
            elif str(self.smb_model) == "TI":
                self.plot_style = dict(
                    mean_usurf=dict(y_label='Mean surface elevation \n 2000- '
                                            '2019 ('
                                            'm)'),
                    point1=dict(
                        y_label='Mean surface elevation\nof fifth bin from front ('
                                'm)'),
                    point2=dict(y_label=f'Mean surface elevation\nof fifth bin from '
                                        f'top ('
                                        f'm)'),
                    melt_f=dict(y_label='Melt Factor OGGM\n( mm w.e. / (C day) )'),
                    prcp_fac=dict(y_label='Precipitation Factor \n( - )'),
                    temp_bias=dict(y_label='Temperature Bias ( C )'),
                )

    def summarise_observables(self, ensemble_observables, new_observables,
                              uncertainty_matrix, noise_samples):

        # uncertainty = np.sqrt(np.diagonal(uncertainty_matrix))
        if self.plot_dhdt:
            time = self.time_period[-1] - self.time_period[0]
            ensemble_observables = ((ensemble_observables - self.binned_usurf_init) /
                                    time)
            noise_samples = (noise_samples -
                             self.binned_usurf_init) / time
            new_observables = ((new_observables - np.mean(self.binned_usurf_init,
                                                          axis=0)) / time)

        uncertainty = np.std(noise_samples, axis=0)
        ensemble_mean = np.mean(ensemble_observables, axis=1)
        ensemble_std = np.std(ensemble_observables, axis=1)

        size = ensemble_observables.shape[1]
        if size < 5:
            ensemble_point1 = ensemble_observables[:, 0]
            ensemble_point2 = ensemble_observables[:, -1]
            obs_point1 = new_observables[0]
            obs_point2 = new_observables[-1]
            var_point1 = uncertainty[0]
            var_point2 = uncertainty[-1]
        else:
            ensemble_point1 = ensemble_observables[:, 2]
            ensemble_point2 = ensemble_observables[:, -3]
            obs_point1 = new_observables[2]
            obs_point2 = new_observables[-3]
            var_point1 = uncertainty[2]
            var_point2 = uncertainty[-3]

        obs_mean = np.mean(new_observables)
        var_mean = np.mean(uncertainty)

        self.observation_log['mean_usurf'].append(obs_mean)
        self.observation_log['point1'].append(obs_point1)
        self.observation_log['point2'].append(obs_point2)

        self.observation_std_log['mean_usurf'].append(var_mean)
        self.observation_std_log['point1'].append(var_point1)
        self.observation_std_log['point2'].append(var_point2)

        for e in range(len(ensemble_mean)):
            self.ensemble_observables_log['mean_usurf'][e].append(
                ensemble_mean[e])
            self.ensemble_observables_log['point1'][e].append(ensemble_point1[e])
            self.ensemble_observables_log['point2'][e].append(ensemble_point2[e])

        return var_mean, ensemble_std

    def plot_iteration(self, ensemble_smb_log,
                       new_observation, uncertainty, iteration, year,
                       ensemble_observables, noise_samples):

        fig, ax = plt.subplots(2, 3, figsize=(10, 6))

        self.summarise_observables(ensemble_observables, new_observation,
                                   uncertainty, noise_samples)

        iteration_axis = range(iteration + 1)
        iteration_axis_repeat = np.repeat(iteration_axis, 2)[1:-1]

        # Plot observables
        from matplotlib.ticker import MaxNLocator

        def set_axis_style(ax, show_x):
            ax.set_ylabel(self.plot_style[key]['y_label'])
            if show_x:
                ax.set_xlabel("Iteration")
            ax.set_xlim(-0.2, self.max_iterations + 0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(axis="y", color="lightgray", linestyle="-", zorder=0)
            ax.grid(axis="x", color="lightgray", linestyle="-", zorder=0)
            ax.xaxis.set_tick_params(bottom=False)
            ax.yaxis.set_tick_params(left=False)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

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
        fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)

        # Plot surface mass balance parameters
        for i, key in enumerate(ensemble_smb_log.keys()):
            key_smb_log = np.array(ensemble_smb_log[key])
            key_mean_smb = np.mean(key_smb_log, axis=0)
            for e in range(self.ensemble_size):
                smb_log_values = key_smb_log[e]
                # Plot each ensemble member's time series
                ax[1, i].plot(iteration_axis,
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
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4)

        import string
        axes = ax.flatten()  # Flatten for easy iteration

        labels_subplot = [f"{letter})" for letter in
                          string.ascii_lowercase[:len(axes)]]

        for ax, label in zip(axes, labels_subplot):
            # Add label to lower-left corner (relative coordinates)
            ax.text(-0.3, 0.95, label, transform=ax.transAxes,
                    fontsize=12, va='bottom', ha='left', fontweight='bold')

        # FINISH AND SAVE
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.15)

        fig.savefig(
            os.path.join(self.monitor_dir, f"status_{iteration:03d}_{year}.png"),
            format='png')

        plt.close(fig)
        plt.clf()

    def set_axis_labels(self, ax, x_ticks, y_ticks, show_x=True, show_y=True):
        if show_x:
            ax.set_xlabel('km')
        if show_y:
            ax.set_ylabel('km')
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        def formatter(x, _):
            return str(int(x * self.resolution / 1000))

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis="y", color="black", linestyle="--", zorder=-1, alpha=.2)
        ax.grid(axis="x", color="black", linestyle="--", zorder=-1, alpha=.2)
        ax.xaxis.set_tick_params(bottom=False)
        ax.yaxis.set_tick_params(left=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def vector_to_map(self, new_observation):
        obs_mapped = np.full_like(self.bin_map, np.nan, dtype=np.float32)
        for bin_id, value in enumerate(new_observation, start=1):
            obs_mapped[self.bin_map == bin_id] = value
        obs_mapped[self.bin_map == 0] = np.nan
        return obs_mapped

    def plot_glacier_property_map(self, ax, data_map, x_ticks, y_ticks,
                                  crop_padding, title, colorlabel, vmin=-10, vmax=10,
                                  cmap='seismic_r', mask=None):
        """
        Plot a glacier property map (e.g., elevation change, velocity, SMB) on the given Axes.

        Parameters:
            cmap:
            colorlabel:
            title:
            ax (matplotlib.axes.Axes): Target axes for the plot.
            data_map (np.ndarray): 2D array of the glacier property.
            x_ticks, y_ticks (list): Tick positions in grid units.
            crop_padding (int): Number of pixels to crop on all sides.
        """
        if mask is None:
            mask = self.icemask_init == 1
        data_map[mask == 0] = np.nan  # Mask non-glacier areas
        cropped = data_map[crop_padding:-crop_padding, crop_padding:-crop_padding]

        img = ax.imshow(cropped, cmap=cmap, vmin=vmin, vmax=vmax,
                        origin='lower',
                        zorder=3)
        plt.colorbar(img, ax=ax, orientation='vertical').set_label(colorlabel)

        self.set_axis_labels(ax, x_ticks, y_ticks, show_x=True, show_y=False)

        mean_val = np.nanmean(data_map[mask])
        ax.set_title(f"{title}\nMean: {mean_val:.2f} m a$^{{-1}}$")

    def plot_maps_prognostic(self, ensembleKF, obs_dhdt_raster,
                             obs_velsurf_mag_raster, init_surf_bin, new_observation,
                             modeled_surface, uncertainty, iteration, year):

        ###################### MAPS #################################################
        nrows = 2
        ncols = 4
        fig, ax = plt.subplots(nrows, ncols, figsize=(15, 10))

        # Define x and y ticks for both plots

        crop_padding = 25
        x_ticks = np.arange(crop_padding, self.bin_map.shape[1] - crop_padding * 2,
                            step=self.resolution)
        y_ticks = np.arange(crop_padding, self.bin_map.shape[0] - crop_padding * 2,
                            step=self.resolution)

        cropped_bedrock = ensembleKF.bedrock[crop_padding:-crop_padding,
                          crop_padding:-crop_padding]

        for row, col in product(range(nrows), range(ncols)):
            ax[row, col].imshow(cropped_bedrock, cmap='gray',
                                vmin=1450, vmax=3600, origin='lower')

        surface = np.mean(ensembleKF.ensemble_usurf, axis=0)
        thk = surface - ensembleKF.bedrock
        new_mask = thk > 0
        # Observed elevation change
        self.plot_glacier_property_map(ax=ax[0, 0],
                                       data_map=obs_dhdt_raster,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Observed\nElevation Change',
                                       colorlabel='Elevation Change (m a$^{-1}$)',
                                       mask=new_mask)

        # Observed Elevation Changes binned
        binned_difference = (new_observation - init_surf_bin) / 19
        new_observation_mapped = self.vector_to_map(binned_difference)

        self.plot_glacier_property_map(ax=ax[0, 1],
                                       data_map=new_observation_mapped,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Observed\nElevation Change binned',
                                       colorlabel='Elevation Change (m a$^{-1}$)',
                                       mask=new_mask)

        # Observed velocities
        obs_velsurf_mag_raster[
            self.icemask_init == 0] = np.nan  # Mask ice-free areas
        self.plot_glacier_property_map(ax=ax[0, 2],
                                       data_map=obs_velsurf_mag_raster,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Observed\nSurface Velocity',
                                       colorlabel='Surface Velocity (m a$^{-1}$)',
                                       vmin=0,
                                       vmax=np.nanmax(obs_velsurf_mag_raster),
                                       cmap='magma')

        # Estimated elevation change (modelled)
        ensemble_usurf = np.mean(ensembleKF.ensemble_usurf, axis=0)
        init_usurf = np.mean(ensembleKF.ensemble_init_surf_raster, axis=0)
        modeled_dhdt = (ensemble_usurf - init_usurf) / 19
        self.plot_glacier_property_map(ax=ax[1, 0],
                                       data_map=modeled_dhdt,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Modelled\nElevation Change',
                                       colorlabel='Elevation Change (m a$^{-1}$)',
                                       mask=new_mask)

        # Modelled elevation Change (binned)
        ensemble_surface_mean = np.mean(modeled_surface, axis=0)
        ensemble_dhdt_mean = (ensemble_surface_mean - init_surf_bin) / 19
        ensemble_dhdt_mean_mapped = self.vector_to_map(ensemble_dhdt_mean)
        self.plot_glacier_property_map(ax=ax[1, 1],
                                       data_map=ensemble_dhdt_mean_mapped,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Modelled\nElevation Change binned',
                                       colorlabel='Elevation Change (m a$^{-1}$)',
                                       mask=new_mask)

        # modelled velocity
        modelled_velocity = np.mean(ensembleKF.ensemble_velsurf_mag_raster, axis=0)
        modelled_velocity[self.icemask_init == 0] = np.nan  # Mask ice-free areas
        self.plot_glacier_property_map(ax=ax[1, 2],
                                       data_map=modelled_velocity,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Modelled\nSurface Velocity',
                                       colorlabel='Surface Velocity (m a$^{-1}$)',
                                       vmin=0,
                                       vmax=np.nanmax(obs_velsurf_mag_raster),
                                       cmap='magma')

        # Surface Mass Balance (SMB) Map
        mean_smb_raster = np.mean(ensembleKF.ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas

        self.plot_glacier_property_map(ax=ax[1, 3],
                                       data_map=mean_smb_raster,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Modelled\nSurface Mass Balance',
                                       colorlabel='Surface Mass Balance (m a$^{'
                                                  '-1}$)', mask=new_mask)

        # Flux Divergence Map
        mean_smb_raster = np.mean(ensembleKF.ensemble_divflux_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas
        self.plot_glacier_property_map(ax=ax[0, 3],
                                       data_map=mean_smb_raster,
                                       x_ticks=x_ticks,
                                       y_ticks=y_ticks,
                                       crop_padding=crop_padding,
                                       title='Modelled\nFlux Divergence',
                                       colorlabel='Flux Divergence (m a$^{-1}$)')

        import string
        axes = ax.flatten()  # Flatten for easy iteration

        labels_subplot = [f"{letter})" for letter in
                          string.ascii_lowercase[:len(axes)]]

        for ax, label in zip(axes, labels_subplot):
            # Add label to lower-left corner (relative coordinates)
            ax.text(-0.35, 1.01, label, transform=ax.transAxes,
                    fontsize=12, va='bottom', ha='left', fontweight='bold')

        fig.tight_layout()
        plt.subplots_adjust(wspace=0.5, left=0.05)

        fig.savefig(
            os.path.join(self.monitor_dir, f"maps_prognostic_{iteration:03d}_"
                                           f"{year}.png"),
            format='png')

        plt.close(fig)

        plt.clf()

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

        fig.write_image(f"Plots/glacier_surface_{year}.png", width=1500,
                        height=1200, scale=0.75)
