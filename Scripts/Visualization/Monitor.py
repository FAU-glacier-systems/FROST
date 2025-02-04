import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.graph_objects as go
import copy
import pyproj


class Monitor:
    def __init__(self, EnKF_object, ObsProvider, max_iterations, output_dir):

        self.rgi_id = EnKF_object.rgi_id
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
        self.max_iterations = max_iterations
        self.max_iteration_axis = range(max_iterations + 1)

        self.colorscale = plt.get_cmap('tab20')

        self.monitor_dir = output_dir
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)

        self.keys = ['mean_usurf', 'point1', 'point2']

        self.ensemble_observables_log = {key: [[] for _ in range(self.ensemble_size)]
                                         for key in self.keys}

        self.observation_log = {key: [] for key in self.keys}

        self.observation_std_log = {key: [] for key in self.keys}

        self.density_factor = {'ela': 1,
                               'gradabl': 0.91,
                               'gradacc': 0.55}

        self.plot_style = dict(
            mean_usurf=dict(y_label='Mean surface elevation in 2019 (m)'),
            point1=dict(y_label='Mean surface elevation\nof third bin from top('
                                'm)'),
            point2=dict(y_label=f'Mean surface elevation\nof third bin from '
                                f'bottom ('
                                f'm)'),
            ela=dict(y_label='Equilibrium Line  Altitude (m)'),
            gradabl=dict(y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
            gradacc=dict(y_label='Accumulated Gradient\n(m a$^{-1}$ km$^{-1}$)'),
        )

    def summarise_observables(self, ensemble_observables, new_observables,
                              uncertainty_matrix):
        uncertainty = np.sqrt(np.diagonal(uncertainty_matrix))
        ensemble_mean_dhdt = np.mean(ensemble_observables, axis=1)
        ensemble_point1 = ensemble_observables[:, 3]
        ensemble_point2 = ensemble_observables[:, -3]

        obs_mean_dhdt = np.mean(new_observables)
        obs_point1 = new_observables[3]
        obs_point2 = new_observables[-3]

        var_mean_dhdt = np.mean(uncertainty)
        var_point1 = uncertainty[3]
        var_point2 = uncertainty[-3]

        self.observation_log['mean_usurf'].append(obs_mean_dhdt)
        self.observation_log['point1'].append(obs_point1)
        self.observation_log['point2'].append(obs_point2)

        self.observation_std_log['mean_usurf'].append(var_mean_dhdt)
        self.observation_std_log['point1'].append(var_point1)
        self.observation_std_log['point2'].append(var_point2)

        for e in range(len(ensemble_mean_dhdt)):
            self.ensemble_observables_log['mean_usurf'][e].append(
                ensemble_mean_dhdt[e])
            self.ensemble_observables_log['point1'][e].append(ensemble_point1[e])
            self.ensemble_observables_log['point2'][e].append(ensemble_point2[e])

    def plot_iteration(self, ensemble_smb_log, ensemble_smb_raster,
                       new_observation, uncertainty, iteration, year,
                       ensemble_observables):

        fig, ax = plt.subplots(2, 5, figsize=(15, 6))

        self.summarise_observables(ensemble_observables, new_observation,
                                   uncertainty)

        iteration_axis = range(iteration + 1)
        iteration_axis_repeat = np.repeat(iteration_axis, 2)[1:-1]

        # Plot observables

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
            ax[0, i].plot(iteration_axis_repeat,
                          observation_log_repeat,
                          color=self.colorscale(0), marker='o', markersize=10,
                          markevery=[-1], zorder=2, label='Observation',
                          )

            observable_var_log_values = self.observation_std_log[key]
            observation_var_log_repeat = np.repeat(observable_var_log_values, 2)
            std_plus = observation_log_repeat + observation_var_log_repeat
            std_minus = observation_log_repeat - observation_var_log_repeat

            ax[0, i].fill_between(iteration_axis_repeat,
                                  std_minus,
                                  std_plus,
                                  color=self.colorscale(1), alpha=0.5,
                                  label='Observation Uncertainty')
            set_axis_style(ax[0, i], show_x=False)

        handles, labels = ax[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)

        # Plot surface mass balance parameters
        for i, key in enumerate(ensemble_smb_log.keys()):
            for e in range(self.ensemble_size):
                smb_log_values = np.array(ensemble_smb_log[key][e])
                # Plot each ensemble member's time series
                ax[1, i].plot(iteration_axis,
                              smb_log_values,
                              color='gold', marker='o', markersize=10,
                              markevery=[-1], zorder=2, label='Ensemble Member')

            ax[1, i].plot(self.max_iteration_axis,
                          [self.reference_smb[key] / self.density_factor[key] for
                           _ in
                           range(self.max_iterations + 1)],
                          color=self.colorscale(8), zorder=2, label='Reference SMB'
                          )
            set_axis_style(ax[1, i], show_x=True)

        handles, labels = ax[1, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4)

        # Define x and y ticks for both plots
        x_ticks = np.arange(0, self.bin_map.shape[1],
                            step=20)  # Adjust step as needed
        y_ticks = np.arange(0, self.bin_map.shape[0], step=20)
        x_labels = (x_ticks * self.resolution) / 1000  # Convert to km
        y_labels = (y_ticks * self.resolution) / 1000

        # Common function for setting axis properties
        def set_axis_labels(ax, show_x=True, show_y=True):
            if show_x:
                ax.set_xlabel('(km)')
            if show_y:
                ax.set_ylabel('(km)')
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels([f"{label:.0f}" for label in x_labels])  # No decimals
            ax.set_yticklabels([f"{label:.0f}" for label in y_labels])
            ax.grid(axis="y", color="black", linestyle="--", zorder=-1, alpha=.2)
            ax.grid(axis="x", color="black", linestyle="--", zorder=-1, alpha=.2)
            ax.xaxis.set_tick_params(bottom=False)
            ax.yaxis.set_tick_params(left=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

        # Observation Map
        def map_observation(new_observation):
            obs_mapped = np.full_like(self.bin_map, np.nan, dtype=np.float32)
            for bin_id, value in enumerate(new_observation, start=1):
                obs_mapped[self.bin_map == bin_id] = value
            obs_mapped[self.bin_map == 0] = np.nan
            return obs_mapped

        obs_mapped = map_observation(new_observation)

        ax_obs_map = ax[0, 3]
        img_obs = ax_obs_map.imshow(obs_mapped, origin='lower', cmap='Blues_r',
                                    vmin=new_observation[0],
                                    vmax=new_observation[-1], zorder=3)
        plt.colorbar(img_obs, ax=ax_obs_map, orientation='vertical').set_label(
            'Surface Elevation\nin 2019 (m)')
        set_axis_labels(ax_obs_map, show_x=False)

        obs_std_mapped = map_observation(np.sqrt(np.diagonal(uncertainty)))
        ax_obs_std_map = ax[0, 4]
        img_obs_std = ax_obs_std_map.imshow(obs_std_mapped, origin='lower',
                                            cmap='Blues', zorder=3)
        plt.colorbar(img_obs_std, ax=ax_obs_std_map,
                     orientation='vertical').set_label(
            'Elevation Uncertainty\nin 2019 (m)')
        set_axis_labels(ax_obs_std_map, show_x=False, show_y=False)

        # Surface Mass Balance (SMB) Map
        mean_smb_raster = np.mean(ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas

        ax_smb_map = ax[1, 3]
        smb_img = ax_smb_map.imshow(mean_smb_raster, cmap='RdBu', vmin=-10, vmax=10,
                                    origin='lower', zorder=3)
        plt.colorbar(smb_img, ax=ax_smb_map, orientation='vertical').set_label(
            'Estimated Surface Mass Balance\n(m a$^{-1}$)')
        set_axis_labels(ax_smb_map, show_x=True)

        # Surface Mass Balance (SMB) Map
        std_smb_raster = np.std(ensemble_smb_raster, axis=0)
        std_smb_raster[self.icemask_init == 0] = np.nan  # Mask ice-free areas

        ax_smb_std_map = ax[1, 4]

        std_smb_img = ax_smb_std_map.imshow(std_smb_raster, cmap='YlOrBr',
                                            origin='lower', zorder=3,
                                            vmin=0,
                                            vmax=1)
        plt.colorbar(std_smb_img, ax=ax_smb_std_map,
                     orientation='vertical').set_label(
            'Surface Mass Balance\nUncertainty (m a$^{-1}$)')
        set_axis_labels(ax_smb_std_map, show_x=True, show_y=False)

        # FINISH AND SAVE
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.15)

        fig.savefig(os.path.join(self.monitor_dir, f"status_{iteration}_"
                                                   f"{year}.png"), dpi=200)
        plt.close(fig)

    def visualise_3d(self, property_map, glacier_surface, bedrock, year, x, y):
        # choose property that is displayed on the glacier surface

        thicknes = glacier_surface - bedrock
        lat_range = x
        lon_range = y
        property_map[thicknes <= 1] = None

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
        glacier_surface[thicknes < 1] = None

        glacier_bottom = copy.copy(bedrock)
        glacier_bottom[thicknes < 1] = None

        # create 3D surface plots with property as surface color
        surface_fig = go.Surface(
            z=glacier_surface,
            x=lat_range,
            y=lon_range,
            colorscale=color_scale,
            cmax=30,
            # cmax=5,
            cmin=-30,
            # cmin=-5,
            surfacecolor=property_map,
            showlegend=False,
            name="glacier surface",
            colorbar=dict(title="Surface Elevation Deviation (m)",
                          titleside="top", thickness=50, orientation="h", y=0.7,
                          len=0.5,
                          titlefont=dict(size=50), tickfont=dict(size=40),
                          tickvals=[-30, 30], tickformat=".0f"
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
            data=[surface_fig],

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
