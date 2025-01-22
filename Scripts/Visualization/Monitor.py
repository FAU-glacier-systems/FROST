import matplotlib.pyplot as plt
import os
import numpy as np


class Monitor:
    def __init__(self, EnKF_object, ObsProvider, output_dir):

        self.rgi_id = EnKF_object.rgi_id
        self.ensemble_size = EnKF_object.ensemble_size
        self.seed = EnKF_object.seed
        self.icemask_init = EnKF_object.icemask_init
        # self.plot_points = ObsProvider.observation_locations
        self.bin_map = ObsProvider.bin_map
        self.time_period = ObsProvider.time_period
        self.start_year = self.time_period[0]
        self.time_period_repeat = np.repeat(self.time_period, 2)[1:]
        self.reference_smb = EnKF_object.reference_smb

        self.colorscale = plt.get_cmap('tab20')

        self.monitor_dir = output_dir
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)

        self.keys = ['mean_usurf', 'point1', 'point2']

        self.ensemble_observables_log = {key: [[] for _ in range(self.ensemble_size)]
                                         for key in self.keys}

        self.observation_log = {key: [] for key in self.keys}

        self.observation_std_log = {key: [] for key in self.keys}

        self.plot_style = dict(
            mean_usurf=dict(y_label='Mean surface elevation in 2020 (m)'),
            point1=dict(y_label='Mean surface elevation of bin 3 (m)'),
            point2=dict(y_label='Mean surface elevation of bin -3 (m)'),
            ela=dict(y_label='Equilibrium Line  Altitude (m)'),
            gradabl=dict(y_label='Ablation Gradient\n(m a$^{-1}$ km$^{-1}$)'),
            gradacc=dict(y_label='Accumulated Gradient\n(m a$^{-1}$ km$^{-1}$)'),
        )

    def summariese_observables(self, ensemble_observables, new_observables,
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

        fig, ax = plt.subplots(2, 4, figsize=(12, 6))

        self.summariese_observables(ensemble_observables, new_observation,
                                    uncertainty)

        iteration_axis = range(iteration + 1)
        iteration_axis_repeat = np.repeat(iteration_axis, 2)[1:-1]
        # Plot observables
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
                          markevery=[-1], zorder=2, label='Observation'
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

            ax[0, i].set_ylabel(self.plot_style[key]['y_label'])
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

            ax[1, i].plot(iteration_axis,
                          [self.reference_smb[key] for _ in range(len(
                              smb_log_values))],
                          color=self.colorscale(8), marker='o', markersize=10,
                          markevery=[-1], zorder=2, label='Reference SMB'
                          )
            ax[1, i].set_ylabel(self.plot_style[key]['y_label'])
            ax[1, i].set_xlabel('Iterations')

        handles, labels = ax[1,0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=4)

        ax_obs_map = ax[0, 3]

        obs_mapped = np.zeros_like(self.bin_map, dtype=np.float32)

        for bin_id, value in enumerate(new_observation, start=1):
            obs_mapped[self.bin_map == bin_id] = value

        obs_mapped[self.bin_map == 0] = None

        img_obs = ax_obs_map.imshow(obs_mapped, origin='lower', cmap='Blues_r',
                                    vmin=new_observation[0],
                                    vmax=new_observation[-1])

        cbar = plt.colorbar(img_obs, ax=ax_obs_map, orientation='vertical',
                            label='Binned Surface Elevation (m)')

        ax_smb_map = ax[1, 3]
        mean_smb_raster = np.mean(ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = 0
        smb_img = ax_smb_map.imshow(mean_smb_raster, cmap='RdBu', vmin=-10, vmax=10,
                                    origin='lower')
        cbar = plt.colorbar(smb_img, ax=ax_smb_map, orientation='vertical',
                            label='Estimated Surface Mass Balance (m a$^{-1}$)')
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.15)

        fig.savefig(os.path.join(self.monitor_dir, f"status_{iteration}_{year}.png"))
        plt.close(fig)
