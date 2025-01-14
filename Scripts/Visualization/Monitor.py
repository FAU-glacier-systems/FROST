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
        self.time_period = ObsProvider.time_period
        self.start_year = self.time_period[0]
        self.time_period_repeat = np.repeat(self.time_period, 2)[1:]
        self.reference_smb = EnKF_object.reference_smb

        self.colorscale = plt.get_cmap('tab20')

        self.monitor_dir = output_dir
        if not os.path.exists(self.monitor_dir):
            os.makedirs(self.monitor_dir)

        self.ensemble_observables_log = {key: [[] for _ in range(self.ensemble_size)]
                                         for key in
                                         ['mean_usurf', 'point1', 'point2']}

        self.observation_log = {key: []
                                for key in
                                ['mean_usurf', 'point1', 'point2']}

    def summariese_observables(self, ensemble_observables, new_observables):
        ensemble_mean_dhdt = np.mean(ensemble_observables, axis=1)
        ensemble_point1 = ensemble_observables[:, 3]
        ensemble_point2 = ensemble_observables[:, -3]

        obs_mean_dhdt = np.mean(new_observables)
        obs_point1 = new_observables[3]
        obs_point2 = new_observables[-3]

        self.observation_log['mean_usurf'].append(obs_mean_dhdt)
        self.observation_log['point1'].append(obs_point1)
        self.observation_log['point2'].append(obs_point2)

        for e in range(len(ensemble_mean_dhdt)):
            self.ensemble_observables_log['mean_usurf'][e].append(
                ensemble_mean_dhdt[e])
            self.ensemble_observables_log['point1'][e].append(ensemble_point1[e])
            self.ensemble_observables_log['point2'][e].append(ensemble_point2[e])

    def plot_status(self, ensemble_smb_log, ensemble_smb_raster,
                    new_observation, usurf_raster, iteration, year,
                    ensemble_observables):

        fig, ax = plt.subplots(2, 4, figsize=(16, 9), layout="tight")

        self.summariese_observables(ensemble_observables, new_observation)

        # Plot observables
        for i, key in enumerate(self.ensemble_observables_log.keys()):
            for e in range(self.ensemble_size):
                observable_log_values = self.ensemble_observables_log[key][e]
                observable_log_repeat = np.repeat(observable_log_values, 2)
                ax[0, i].plot(self.time_period_repeat[:len(observable_log_repeat)],
                              observable_log_repeat,
                              color=self.colorscale(5), marker='o', markersize=10,
                              markevery=[-1], zorder=2)

            observable_log_values = self.observation_log[key]
            observation_log_repeat = np.repeat(observable_log_values, 2)
            ax[0, i].plot(self.time_period_repeat[:len(observation_log_repeat)],
                          observation_log_repeat,
                          color=self.colorscale(0), marker='o', markersize=10,
                          markevery=[-1], zorder=2
                          )

        # Plot surface mass balance parameters
        for i, key in enumerate(ensemble_smb_log.keys()):
            for e in range(self.ensemble_size):
                smb_log_values = np.array(ensemble_smb_log[key][e])
                # Plot each ensemble member's time series
                ax[1, i].plot(self.time_period[:len(smb_log_values)],
                              smb_log_values,
                              color='gold', marker='o', markersize=10,
                              markevery=[-1], zorder=2)

            ax[1, i].plot(self.time_period[:len(smb_log_values)],
                          [self.reference_smb[key] for _ in range(len(
                              smb_log_values))],
                          color=self.colorscale(8), marker='o', markersize=10,
                          markevery=[-1], zorder=2
                          )

        ax_dhdt_map = ax[0, 3]
        obs_img = ax_dhdt_map.imshow(usurf_raster, cmap='Blues',
                                     origin='lower')
        cbar = plt.colorbar(obs_img, ax=ax_dhdt_map,
                            orientation='vertical')

        # ax_dhdt_map.scatter(self.plot_points[20, 1], self.plot_points[20, 0])
        # ax_dhdt_map.scatter(self.plot_points[-20, 1], self.plot_points[-20, 0])

        ax_smb_map = ax[1, 3]
        mean_smb_raster = np.mean(ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = 0
        smb_img = ax_smb_map.imshow(mean_smb_raster, cmap='RdBu', vmin=-10, vmax=10,
                                    origin='lower')
        cbar = plt.colorbar(smb_img, ax=ax_smb_map,
                            orientation='vertical')

        fig.savefig(os.path.join(self.monitor_dir, f"status_{iteration}_{year}.png"))
        plt.close(fig)

    def plot_iteration(self, ensemble_smb_log, ensemble_smb_raster,
                    new_observation, usurf_raster, iteration, year,
                    ensemble_observables):

        fig, ax = plt.subplots(2, 4, figsize=(16, 9), layout="tight")

        self.summariese_observables(ensemble_observables, new_observation)
        iteration_axis = range(iteration+1)
        iteration_axis_repeat = np.repeat(iteration_axis, 2)[1:-1]
        # Plot observables
        for i, key in enumerate(self.ensemble_observables_log.keys()):
            for e in range(self.ensemble_size):
                observable_log_values = self.ensemble_observables_log[key][e]
                observable_log_repeat = np.repeat(observable_log_values, 2)
                ax[0, i].plot(iteration_axis_repeat,
                              observable_log_repeat,
                              color=self.colorscale(5), marker='o', markersize=10,
                              markevery=[-1], zorder=2)

            observable_log_values = self.observation_log[key]
            observation_log_repeat = np.repeat(observable_log_values, 2)
            ax[0, i].plot(iteration_axis_repeat,
                          observation_log_repeat,
                          color=self.colorscale(0), marker='o', markersize=10,
                          markevery=[-1], zorder=2
                          )

        # Plot surface mass balance parameters
        for i, key in enumerate(ensemble_smb_log.keys()):
            for e in range(self.ensemble_size):
                smb_log_values = np.array(ensemble_smb_log[key][e])
                # Plot each ensemble member's time series
                ax[1, i].plot(iteration_axis,
                              smb_log_values,
                              color='gold', marker='o', markersize=10,
                              markevery=[-1], zorder=2)

            ax[1, i].plot(iteration_axis,
                          [self.reference_smb[key] for _ in range(len(
                              smb_log_values))],
                          color=self.colorscale(8), marker='o', markersize=10,
                          markevery=[-1], zorder=2
                          )

        ax_dhdt_map = ax[0, 3]
        obs_img = ax_dhdt_map.imshow(usurf_raster, cmap='Blues_r',
                                     origin='lower')
        cbar = plt.colorbar(obs_img, ax=ax_dhdt_map,
                            orientation='vertical')

        # ax_dhdt_map.scatter(self.plot_points[20, 1], self.plot_points[20, 0])
        # ax_dhdt_map.scatter(self.plot_points[-20, 1], self.plot_points[-20, 0])

        ax_smb_map = ax[1, 3]
        mean_smb_raster = np.mean(ensemble_smb_raster, axis=0)
        mean_smb_raster[self.icemask_init == 0] = 0
        smb_img = ax_smb_map.imshow(mean_smb_raster, cmap='RdBu', vmin=-10, vmax=10,
                                    origin='lower')
        cbar = plt.colorbar(smb_img, ax=ax_smb_map,
                            orientation='vertical')

        fig.savefig(os.path.join(self.monitor_dir, f"status_{iteration}_{year}.png"))
        plt.close(fig)
