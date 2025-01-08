import matplotlib.pyplot as plt
import os
import numpy as np


class Monitor:
    def __init__(self, EnKF_object, ObsProvider):

        self.rgi_id = EnKF_object.rgi_id
        self.plot_points = ObsProvider.observation_locations
        self.time_period = ObsProvider.time_period
        self.start_year = self.time_period[0]
        self.time_period_repeat = np.repeat(self.time_period, 2)[1:]

        self.colorscale = plt.get_cmap('tab20')

        monitor_dir = os.path.join('Experiments', self.rgi_id, 'Monitor')
        if not os.path.exists(monitor_dir):
            os.makedirs(monitor_dir)

    def plot_status(self, EnKF_object, observation, iteration, year):

        fig, ax = plt.subplots(2, 4, figsize=(16, 9), layout="tight")

        modeled_observables_log = [] #TODO

        # Plot observables
        for i, key in enumerate(modeled_observables_log.keys()):
            for e in range(EnKF_object.ensemble_size):
                observable_log_values = modeled_observables_log[key][e]
                observable_log_repeat = np.repeat(observable_log_values, 2)
                ax[0, i].plot(self.year_range_repeat[:len(observable_log_repeat)],
                              observable_log_repeat,
                              color=self.colorscale(5), marker='o', markersize=10,
                              markevery=[-1], zorder=2)

        # Plot surface mass balance parameters
        for i, key in enumerate(EnKF_object.ensemble_smb_log.keys()):
            for e in range(EnKF_object.ensemble_size):
                smb_log_values = np.array(EnKF_object.ensemble_smb_log[key][e])
                # Plot each ensemble member's time series
                ax[1, i].plot(self.year_range[:len(smb_log_values)],
                              smb_log_values,
                              color='gold', marker='o', markersize=10,
                              markevery=[-1], zorder=2)

        ax_dhdt_map = ax[0, 3]
        ax_dhdt_map.imshow(observation, cmap='RdBu', vmin=-10, vmax=10,
                           origin='lower')
        # ax_dhdt_map.scatter(self.plot_points[:, 1], self.plot_points[:, 0],)

        ax_smb_map = ax[1, 3]
        mean_smb_raster = np.mean(EnKF_object.ensemble_smb_raster, axis=0)
        mean_smb_raster[EnKF_object.icemask_init == 0] = 0
        ax_smb_map.imshow(mean_smb_raster, cmap='RdBu', vmin=-10, vmax=10,
                          origin='lower')

        fig.savefig(os.path.join(self.monitor_dir, f"status_{iteration}_{year}.png"))
