import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_results():
    # global figure
    fig_para, ax_para = plt.subplots(nrows=5, ncols=2, figsize=(10, 12))

    # run for every hyperparameters
    for fig_num, hyperparameter in enumerate(['elevation_step',
                                              'ensemble_size',
                                              'iterations',
                                              'obs_uncertainty',
                                              'init_offset'
                                              ]):

        if hyperparameter == 'elevation_step':
            experiment_folder = os.path.join(
                '../../Experiments/RGI2000-v7.0-G-11-01706/', hyperparameter)
        elif hyperparameter == 'ensemble_size':
            experiment_folder = os.path.join(
                '../../Experiments/RGI2000-v7.0-G-11-01706_v1/', hyperparameter)
        elif hyperparameter == 'init_offset':
            experiment_folder = os.path.join(
                '../../Experiments/RGI2000-v7.0-G-11-01706_v2/', hyperparameter)
        elif hyperparameter == 'iterations':
            experiment_folder = os.path.join(
                '../../Experiments/RGI2000-v7.0-G-11-01706_v3/', hyperparameter)
        elif hyperparameter == 'obs_uncertainty':
            experiment_folder = os.path.join(
                '../../Experiments/RGI2000-v7.0-G-11-01706_v4/', hyperparameter)

        # load result json files
        results = []
        bin_centers = []
        for elevation_step in os.listdir(experiment_folder):
            path_elevation_step = os.path.join(experiment_folder, elevation_step)
            bin_centers.append(int(elevation_step))
            for seed_folder in os.listdir(path_elevation_step):
                if seed_folder.startswith('Seed'):
                    seed_path = os.path.join(path_elevation_step, seed_folder)
                    files = os.listdir(seed_path)
                    for file in files:
                        if file.startswith('result'):
                            file_path = os.path.join(seed_path, file)
                            with open(file_path, 'r') as f:
                                content = json.load(f)
                                if content != None:
                                    results.append(content)

        # get the hyperparameter values
        # if hyperparameter == 'initial_uncertainty':
        #    hyper_results = [exp['initial_uncertainity'] for exp in results]
        # else:

        # Compute the MAE and track the maximum MAE for normalisation
        mean_estimate = []
        var_estimate = []
        hyper_results = []
        for i, run in enumerate(results):
            print(i)
            try:
                hyper_results.append(run[hyperparameter])
            except KeyError:
                continue  #
            mean_estimate.append(np.array(run['final_mean']))

            var_estimate.append(np.array(run['final_std']))
            # TODO: hotfix

        MAE = np.empty((0, len(hyper_results)))
        MAX_para_total = np.array([])
        spread = np.empty((0, len(hyper_results)))
        MAX_spread_total = np.array([])
        # TODO
        reference_smb = results[0]['reference_smb']
        true_x = [reference_smb['ela'], reference_smb['gradabl'], reference_smb[
            'gradacc']]
        for x in range(3):
            MAE_para = np.array([abs(true_x[x] - exp[x]) for exp in mean_estimate])

            MAX_para = np.max(MAE_para)
            MAE = np.append(MAE, [MAE_para], axis=0)
            MAX_para_total = np.append(MAX_para_total, MAX_para)

            spread_para = np.sqrt(np.array([exp[x] for exp in var_estimate]))
            MAX_spread = np.max(spread_para)
            spread = np.append(spread, [spread_para], axis=0)
            MAX_spread_total = np.append(MAX_spread_total, MAX_spread)

        # Normalise
        #MAX_para_total = [50, 5, 5]
        #MAX_spread_total = [50, 5, 5]
        print(int(MAX_para_total[0]))

        max_gradient = np.max(MAX_para_total[1:])
        MAE[0] = MAE[0] / MAX_para_total[0]
        MAE[1] = MAE[1] / max_gradient
        MAE[2] = MAE[2] / max_gradient
        # MAX = [5000, 0.05, 0.05]
        max_gradient_spread = np.max(MAX_spread_total[1:])
        spread[0] = spread[0] / MAX_spread_total[0]
        spread[1:] = spread[1:] / max_gradient_spread
        # MAE = MAE.flatten()

        # create pandas data frame
        df = pd.DataFrame({'MAE0': MAE[2],
                           'MAE1': MAE[0],
                           'MAE2': MAE[1],
                           'spread0': spread[2],
                           'spread1': spread[0],
                           'spread2': spread[1],
                           hyperparameter: hyper_results
                           # + hyper_results + hyper_results,
                           })
        # define colors
        # print(len(df_glamos_bin))
        # df = df[df['MAE1'] < 0.2]
        # df = df[df['MAE0'] < 0.2]
        # df = df[df['MAE2'] < 0.2]
        # df = df[df['spread0'] < 0.2]
        # df = df[df['spread1'] < 0.2]
        # df = df[df['spread2'] < 0.2]
        colorscale = plt.get_cmap('tab20c')
        colormap = [colorscale(0), colorscale(2), colorscale(3),
                    'black', colorscale(18), colorscale(19),

                    colorscale(4), colorscale(6), colorscale(7)]
        # csfont = {'fontname': 'Comic Sans'}

        # define bin centers
        if hyperparameter == 'dt':
            bin_centers = [1, 2, 4, 5, 10, 20]

        elif hyperparameter == 'covered_area':
            bin_centers = [0.1, 0.2, 0.5, 1, 2, 10]

        elif hyperparameter == 'ensemble_size':
            bin_centers = [8, 16, 32, 64, 128]

        elif hyperparameter == 'process_noise':
            bin_centers = [0, 0.5, 1, 2, 4]

        elif hyperparameter == 'init_offset':
            bin_centers = [20, 40, 60, 80, 100]

        elif hyperparameter == 'obs_uncertainty':
            bin_centers = [5, 10, 20, 40, 80]

        elif hyperparameter == 'specal_noise':
            bin_centers = [1, 2, 3]

        elif hyperparameter == 'bias':
            bin_centers = [0, 2, 4, 6, 8, 10]

        elif hyperparameter == 'elevation_step':
            bin_centers = [5, 10, 25, 50, 100]

        elif hyperparameter == 'iterations':
            bin_centers = [1, 2, 5, 8, 10]

        # group the MAE by bin_centers\\
        marker = ["^", "o", "v", ]
        mean_max = 0
        for x, label in enumerate(
                ["Accumulation Gradient", "Equilibrium Line Altitude (ELA)",
                 "Ablation Gradient"]):
            bin_list_para = [df['MAE' + str(x)][(df[hyperparameter] == center)] for
                             center in bin_centers]
            bin_means_para = np.array([np.median(bins) for bins in bin_list_para])
            if mean_max < max(bin_means_para):
                mean_max = max(bin_means_para)
            bin_list_spread = [df['spread' + str(x)][(df[hyperparameter] == center)]
                               for center in bin_centers]
            bin_means_spread = np.array(
                [np.median(bins) for bins in bin_list_spread])
            if mean_max < max(bin_means_para):
                mean_max = max(bin_means_para)

            print(hyperparameter)
            print([len(bin) for bin in bin_list_para])

            bplot_para = ax_para[fig_num, 0].boxplot(bin_list_para,
                                                     positions=np.arange(x,
                                                                         len(bin_list_para) * 3,
                                                                         3) - ((
                                                                                       x - 1) * 0.44),
                                                     showmeans=True,
                                                     showfliers=False,
                                                     patch_artist=True,
                                                     boxprops=dict(
                                                         facecolor=colormap[
                                                             x * 3 + 2],
                                                         color=colormap[x * 3 + 1]),
                                                     capprops=dict(
                                                         color=colormap[x * 3 + 1]),
                                                     whiskerprops=dict(
                                                         color=colormap[x * 3 + 1]),
                                                     flierprops=dict(
                                                         markeredgecolor=colormap[
                                                             x * 3 + 2],
                                                         marker=marker[x]),
                                                     meanprops=dict(marker='o',
                                                                    markeredgecolor='none',
                                                                    markersize=8,
                                                                    markerfacecolor="none"),
                                                     medianprops=dict(linestyle='-',
                                                                      linewidth=4,
                                                                      color="none"))

            bplot_spreadd = ax_para[fig_num, 1].boxplot(bin_list_spread,
                                                        positions=np.arange(x,
                                                                            len(bin_list_spread) * 3,
                                                                            3) - (
                                                                          (
                                                                                  x - 1) * 0.44),
                                                        showmeans=True,
                                                        showfliers=False,
                                                        patch_artist=True,
                                                        boxprops=dict(
                                                            facecolor=colormap[
                                                                x * 3 + 2],
                                                            color=colormap[
                                                                x * 3 + 1]),
                                                        capprops=dict(color=colormap[
                                                            x * 3 + 1]),
                                                        whiskerprops=dict(
                                                            color=colormap[
                                                                x * 3 + 1]),
                                                        flierprops=dict(
                                                            markeredgecolor=colormap[
                                                                x * 3 + 2],
                                                            marker=marker[x]),
                                                        meanprops=dict(marker='o',
                                                                       markeredgecolor='none',
                                                                       markersize=8,
                                                                       markerfacecolor="none"),
                                                        medianprops=dict(
                                                            linestyle='-',
                                                            linewidth=4,
                                                            color="none"))
            # fill with colors

            # for patch in bplot['boxes']:
            #    patch.set_facecolor(colorscale(x*2+1))
            # ax[i].plot(np.arange(1, len(bin_centers) + 1), bin_var_mean, color=var_color, marker='v')
            ax_para[fig_num, 0].plot(
                np.arange(x, len(bin_list_para) * 3, 3) - ((x - 1) * 0.44),
                bin_means_para,
                color=colormap[x * 3], marker=marker[x], label=label,
                zorder=10 + (-abs(x - 1)))
            ax_para[fig_num, 1].plot(
                np.arange(x, len(bin_list_spread) * 3, 3) - ((x - 1) * 0.44),
                bin_means_spread,
                color=colormap[x * 3], marker=marker[x], label=label,
                zorder=10 + (-abs(x - 1)))
        # ax[i,j].plot(np.arange(1, len(bin_centers) + 1), bin_medians, color=median_color)
        #ax_para[fig_num, 0].set_yscale('log')
        grad_axis_para = ax_para[fig_num, 0].secondary_yaxis('right')
        #grad_axis_para.set_yscale('log')
        grad_axis_para.set_ylabel('Gradient Error (m a$^{-1}$ km$^{-1}$)')
        ax_para[fig_num, 0].set_ylabel('ELA Error (m)')

        #ax_para[fig_num, 1].set_yscale('log')
        grad_axis_spread = ax_para[fig_num, 1].secondary_yaxis('right')
        #grad_axis_spread.set_yscale('log')
        grad_axis_spread.set_ylabel('Gradient Spread (m a$^{-1}$ km$^{-1}$)')
        ax_para[fig_num, 1].set_ylabel('ELA Spread (m)')

        yticks_positions_log = np.logspace(-3, 0, 4)
        yticks_positions_lin = np.arange(-0.5, 1.1, 0.25)

        ax_para[fig_num, 0].set_yticks(yticks_positions_lin,
                                       ['%.1f' % (MAX_para_total[0] * pos) for pos in
                                        yticks_positions_lin])
        grad_axis_para.set_yticks(yticks_positions_lin,
                                  ['%.3f' % (e * max_gradient) for e in
                                   yticks_positions_lin])
        ax_para[fig_num, 1].set_yticks(yticks_positions_lin,
                                       ['%.1f' % (MAX_spread_total[0] * pos) for pos
                                        in yticks_positions_lin])
        grad_axis_spread.set_yticks(yticks_positions_lin,
                                    ['%.3f' % (e * max_gradient_spread) for e in
                                     yticks_positions_lin])

        for num, grad_axis in [(0, grad_axis_para), (1, grad_axis_spread)]:
            ax_para[fig_num, num].get_yaxis().set_tick_params(which='minor', size=0)
            ax_para[fig_num, num].get_yaxis().set_tick_params(which='minor', width=0)
            grad_axis.get_yaxis().set_tick_params(which='minor', size=0)
            grad_axis.get_yaxis().set_tick_params(which='minor', width=0)
            ax_para[fig_num, num].spines['top'].set_visible(False)
            ax_para[fig_num, num].spines['right'].set_visible(False)
            ax_para[fig_num, num].spines['bottom'].set_visible(False)
            ax_para[fig_num, num].spines['left'].set_visible(False)
            grad_axis.spines['right'].set_visible(False)

            ax_para[fig_num, num].grid(axis="y", color="lightgray", linestyle="-")
            ax_para[fig_num, num].grid(axis="x", color="lightgray", linestyle="-",
                                       which='minor')
            ax_para[fig_num, num].set_ylim(10 ** -3.1, 1.2)
            ax_para[fig_num, num].set_xlim(-0.75, len(bin_list_para) * 3 - 0.25)
            ax_para[fig_num, num].set_xticks(
                np.arange(-0.5, len(bin_list_para) * 3, 3), minor=True)
            ax_para[fig_num, num].set_xticks(np.arange(1, len(bin_list_para) * 3, 3),
                                             bin_centers)
            ax_para[fig_num, num].yaxis.set_tick_params(left=False)
            # ax[i,j].xaxis.set_tick_params(bottom=True, which='minor',color="lightgray")
            ax_para[fig_num, num].xaxis.set_tick_params(bottom=False, which='both', )

            grad_axis.yaxis.set_tick_params(right=False)
            handles, labels = ax_para[fig_num, num].get_legend_handles_labels()

            if hyperparameter == 'covered_area':
                ax_para[fig_num, num].set_xlabel("Covered Area Fraction (%)")
            elif hyperparameter == 'dt':
                ax_para[fig_num, num].set_xlabel(
                    "Observation Interval ($dt$) [years]")
            elif hyperparameter == 'ensemble_size':
                ax_para[fig_num, num].set_xlabel('Ensemble Size')
            elif hyperparameter == 'process_noise':
                ax_para[fig_num, num].set_xlabel('Process Noise ($Q$)')
            elif hyperparameter == 'initial_offset':
                ax_para[fig_num, num].set_xlabel('Initial Offset')
            elif hyperparameter == 'initial_uncertainty':
                ax_para[fig_num, num].set_xlabel('Initial Uncertainty')
            elif hyperparameter == 'specal_noise':
                ax_para[fig_num, num].set_xlabel('Specal Noise')
            elif hyperparameter == 'bias':
                ax_para[fig_num, num].set_xlabel('Elevation Bias')
            elif hyperparameter == 'elevation_step':
                ax_para[fig_num, num].set_xlabel('Elevation Bin Step (m)')
            elif hyperparameter == 'observation_uncertainty':
                ax_para[fig_num, num].set_xlabel(
                    'Elevation Change Uncertainty (m a$^{-1}$)')
            else:
                ax_para[fig_num, num].set_xlabel(hyperparameter)

    fig_para.legend(handles, labels, loc='upper center', ncol=3)
    fig_para.tight_layout()
    fig_para.subplots_adjust(top=0.97, bottom=0.04)
    # fig_para.savefig(f'MAE_ext.pdf', format="pdf")
    fig_para.savefig(f'../../Plots/MAE_ext.png', format="png", dpi=300)


if __name__ == '__main__':
    plot_results()
