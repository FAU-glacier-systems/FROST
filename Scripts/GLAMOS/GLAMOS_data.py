import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import json
import matplotlib.gridspec as gridspec


def extract_gradients(ela, mb, elevation):
    # Find the index of the row with the smallest absolute mass balance
    # compute gradients
    ablation = np.array(mb[mb < 0])
    accumulation = np.array(mb[mb > 0])

    # Concatenate the new value
    elevation = np.array(elevation)
    elevation_abl = elevation[mb < 0]
    elevation_acc = elevation[mb > 0]

    adjusted_elevation_acc = elevation_acc - ela
    adjusted_elevation_abl = elevation_abl - ela

    gradient_accumulation = np.sum(adjusted_elevation_acc * accumulation) / np.sum(adjusted_elevation_acc ** 2)
    gradient_ablation = np.sum(adjusted_elevation_abl * ablation) / np.sum(adjusted_elevation_abl ** 2)

    return gradient_ablation, gradient_accumulation


def compute_specific_mass_balance_from_ela(ela, gradabl, gradacc, usurf, thkse):
    maxacc = 100
    mb = []
    for surf,thk in zip(usurf,thkse):
        smb = surf - ela
        smb *= np.where(np.less(smb, 0), gradabl, gradacc)
        smb = np.clip(smb, -100, maxacc)
        smb = np.where((smb < 0) | (thk > 1), smb, -10)
        mb.append(np.sum(smb[thk > 1]) / np.sum(thk > 1))

    return np.mean(mb)

def get_ela_and_specific_mb(glacier_2000_total, time):
    specific_mass_balances = []
    ELA = []
    for i, year in enumerate(time):
        keys_for_specific_year = [key for key in list(glacier_2000_total['end date of observation']) if
                                  key.year == year]
        if len(keys_for_specific_year) == 1:
            entry = glacier_2000_total[glacier_2000_total['end date of observation'] == keys_for_specific_year[0]]
            mass_balance = float(entry['annual mass balance'].iloc[0])
            specific_mass_balances.append(mass_balance / 1000)

            ela = int(entry['equilibrium line altitude'])
            ELA.append(ela)
        else:
            specific_mass_balances.append(np.nan)
            ELA.append(np.nan)

    return ELA, specific_mass_balances

def create_elevation_bins(ela, gradabl, gradacc, elevation):
    elevation_bins = np.arange(elevation.min() - 50, elevation.max() + 50, 100)
    elevation_mb = []
    for elevation in elevation_bins:
        if elevation < ela:
            elevation_mb.append(-abs(elevation - ela) * gradabl)
        else:
            elevation_mb.append(abs(elevation - ela) * gradacc)
    return elevation_bins, elevation_mb

def main(params):
    ### GET GEODETIC SPECIFIC MASS BALANCE ####
    time = np.arange(2000, 2020)
    hugonnet_nc = xr.open_dataset(params['file_path_hugonnet'])
    dhdt = np.array(hugonnet_nc['dhdt'])[0]
    dhdt_error = np.array(hugonnet_nc['obs_error'])[1]
    usurf = np.array(hugonnet_nc['usurf'])
    thk = np.array(hugonnet_nc['thk'])
    icemask = np.array(hugonnet_nc['icemask'])[0]

    # compute specific mass balance
    dhdt[icemask == 0] = 0
    dhdt_error[icemask == 0] = 0
    hugonnet_mass_balance = np.sum(dhdt) / np.sum(icemask)
    hugonnet_mass_error = np.sum(dhdt_error) / np.sum(icemask)
    hugonnet_mass_balance *= 0.91 # conversion to water equivalent

    ### COMPUTE SPECIFIC MASS BALANCE OF ENSEMBLE
    with open(params['results_file'], 'r') as f:
        results = json.load(f)

    ensemble = np.array(results['final_ensemble'])
    ensemble_mean_ela, ensemble_mean_abl, ensemble_mean_acc = np.array(results['final_mean_estimate'])
    ensemble_mean_abl *= params['ablation_density']
    ensemble_mean_acc *= params['accumulation_density']

    mbs = []

    for ensemble_member in ensemble:
        member_ela = ensemble_member[0]
        grad_abl = ensemble_member[1] * 0.91 # hugonnet uses 0.91 so we are going to use
        grad_acc = ensemble_member[2] * 0.91 # it here too
        mbs.append(compute_specific_mass_balance_from_ela(member_ela, grad_abl, grad_acc, usurf, thk))

    mean_mb = np.mean(mbs)
    std_mb = np.std(mbs)
    print("Ensemble mean specific mass balance and std: ", mean_mb, std_mb)
    #######################################################################


    ### GET GLACIOLOGICAL DATA ###
    file_path_glamos_bin = 'massbalance_observation_elevationbins.csv'
    file_path_glamos = 'massbalance_observation.csv'

    # Read the CSV file into a pandas DataFrame, skipping the first 6 lines
    df_glamos_bin = pd.read_csv(file_path_glamos_bin, delimiter=';', skiprows=6)
    df_glamos = pd.read_csv(file_path_glamos, delimiter=';', skiprows=6)

    # Filter rows where the 'glacier name'
    glacier_name = params['glacier_name']
    glamos_df_bin = df_glamos_bin[df_glamos_bin['glacier name'] == glacier_name]
    glamos_df_total = df_glamos[df_glamos['glacier name'] == glacier_name]

    # Convert the 'start date of observation' column to datetime
    glamos_df_bin['start date of observation'] = pd.to_datetime(glamos_df_bin['start date of observation'])
    glamos_df_bin['end date of observation'] = pd.to_datetime(glamos_df_bin['end date of observation'])
    glamos_df_bin['upper elevation of bin'] = pd.to_numeric(glamos_df_bin['upper elevation of bin'])
    glamos_df_bin['annual mass balance'] = pd.to_numeric(glamos_df_bin['annual mass balance'])

    glamos_df_total['end date of observation'] = pd.to_datetime(glamos_df_total['end date of observation'])
    glamos_df_total['annual mass balance'] = pd.to_numeric(glamos_df_total['annual mass balance'])

    # Filter rows after the year 2000
    glacier_2000_binned = glamos_df_bin[np.logical_and(glamos_df_bin['end date of observation'].dt.year >= 2000,
                                                   glamos_df_bin['end date of observation'].dt.year <= 2019)]
    glacier_2000_aggregated = glamos_df_total[np.logical_and(glamos_df_total['end date of observation'].dt.year >= 2000,
                                                        glamos_df_total['end date of observation'].dt.year <= 2019)]

    # extract ela and specific massbalance from
    ELA, specific_mass_balances = get_ela_and_specific_mb(glacier_2000_aggregated, time)
    avg_ela = np.nanmean(np.array(ELA))

    # extract gradients
    elevation_group_df = glacier_2000_binned.groupby('upper elevation of bin')
    elevation_series = elevation_group_df['upper elevation of bin'].mean()
    mean_mb_elevation_bin = elevation_group_df['annual mass balance'].mean()/1000
    avg_grad_abl, avg_grad_acc = extract_gradients(avg_ela, mean_mb_elevation_bin, elevation_series - 50)

    # Extract data for plotting left plot
    date = glacier_2000_binned['end date of observation'].dt.year
    elevation = glacier_2000_binned['upper elevation of bin']
    binned_mass_balance = glacier_2000_binned['annual mass balance']
    binned_mass_balance /= 1000
    ######################################################

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 4])  # Adjust the width ratios as needed
    a0 = fig.add_subplot(gs[0])
    a1 = fig.add_subplot(gs[1])

    ### plot left figure ###
    a0.set_title('Elevation dependent Annual Mass Balance of '+ glacier_name)
    # plot background elevation change bin
    scatter_bins = a0.scatter(date, elevation - 50, c=binned_mass_balance, cmap='seismic_r',vmin=-10, vmax=10,
                              marker='s', s=300, zorder=2)
    fig.colorbar(scatter_bins, ax=a0, label='Mass Balance (m w.e. a$^{-1}$)')
    a0.scatter(np.array(time) - 0.03, ELA, alpha=0.3, c='black', label="Equilibrium Line Altitude", marker='_', s=300,
               zorder=3)

    # create mean elevation_bins of GLAMOS
    elevation_bins, elevation_mb = create_elevation_bins(avg_ela, avg_grad_abl, avg_grad_acc, elevation)
    a0.scatter([2023] * len(elevation_mb), elevation_bins, c=elevation_mb, cmap='seismic_r',
               vmin=-10, vmax=10,
               marker='s', s=300, zorder=2)
    a0.scatter([2023], avg_ela, alpha=0.3, c='black', marker='_', s=300, zorder=3)

    y_range = max(elevation.values)-min(elevation.values)

    a0.text(2024, avg_ela - y_range/10, f'$s_{{ELA}}$: {int(avg_ela)}', rotation=90, color='black')
    a0.text(2024, min(elevation), f'$\gamma_{{abl}}$: {avg_grad_abl:.4f}', rotation=90, color='red')
    a0.text(2024, max(elevation)-y_range/4, f'$\gamma_{{acc}}$: {avg_grad_acc:.4f}', rotation=90,
            label='Mean Accumulation Gradient', color='blue')

    # create mean elevation_bins of ENSEMBLE
    elevation_bins, elevation_mb = create_elevation_bins(ensemble_mean_ela, ensemble_mean_abl, ensemble_mean_acc, elevation)
    a0.scatter([2026] * len(elevation_mb), elevation_bins, c=elevation_mb, cmap='seismic_r',
               vmin=-10, vmax=10,
               marker='s', s=300, zorder=2)
    a0.scatter([2026], ensemble_mean_ela, alpha=0.3, c='black', marker='_', s=300, zorder=3)
    a0.text(2027, ensemble_mean_ela - y_range/10, f'$s_{{ELA}}$: {int(ensemble_mean_ela)}', rotation=90, color='black')
    a0.text(2027, min(elevation), f'$\gamma_{{abl}}$: {ensemble_mean_abl:.4f}', rotation=90,
            label='Mean Ablation Gradient', color='red')
    a0.text(2027, max(elevation)-y_range/4, f'$\gamma_{{acc}}$: {ensemble_mean_acc:.4f}', rotation=90,
            label='Mean Accumulation Gradient', color='blue')

    a0.set_xticks(list(np.array(time)[::4]) + [2023, 2026],
                  list(np.array(time)[::4]) + ['GLAMOS\n Mean', 'EnKF\n Mean'])

    a0.set_xlabel('Year of Measurement')
    a0.set_ylabel('Elevation (m)')
    a0.legend(loc='upper left')

    ### plot right figure ###
    a1.set_title('Specific Mass Balance of ' + params['glacier_name'])

    mean_specific_mass_balance = np.nanmean(specific_mass_balances)
    mean_specific_mass_balance_line = [mean_specific_mass_balance] * len(time)

    a1.plot(time, specific_mass_balances, label='Glaciological Annually [GLAMOS]', color='black', alpha=0.3)
    a1.plot(time, mean_specific_mass_balance_line, label='Glaciological Mean [GLAMOS]', color='black')
    a1.text(time[0], mean_specific_mass_balance_line[-1] - 0.2, f'{mean_specific_mass_balance:.4f} ', color='black')

    geodetic_table_dmdtda = [hugonnet_mass_balance] * 2
    a1.plot([2032, 2038], geodetic_table_dmdtda, label='Geodetic Mean [Hugonnet21]', color='C0', zorder=10)
    a1.fill_between([2032, 2038], geodetic_table_dmdtda - hugonnet_mass_error,
                    geodetic_table_dmdtda + hugonnet_mass_error, color='C0',
                    alpha=0.1, zorder=0,
                    label='Geodetic Uncertainty [Hugonnet21]')

    a1.text(2032, geodetic_table_dmdtda[-1] + 0.03, f'{hugonnet_mass_balance:.4f}', color='C0', zorder=10)

    ensemble_var0_list = [mean_mb - std_mb] * 2
    ensemble_var1_list = [mean_mb + std_mb] * 2

    a1.plot([2025, 2031], [mean_mb, mean_mb], color='orange')
    a1.fill_between([2025, 2031], ensemble_var0_list, ensemble_var1_list, color='C1', alpha=0.1, zorder=0, )
    a1.text(2025, mean_mb + 0.03, f'{mean_mb:.4f}', color='C1', zorder=10)

    a1.set_xticks(list(np.array(time)[::4]) + [2028, 2035],
                  list(np.array(time)[::4]) + ['EnKF\n Mean', 'Geodetic\n Mean'])

    a1.set_ylabel('Mass Balance (m w.e. a$^{-1}$)')
    a1.set_ylim(-2.1, 1.1)
    a1.set_xlabel('Year')
    a1.legend(loc='upper left')

    for axi in [a0, a1]:
        axi.grid(axis="y", color="lightgray", linestyle="-", zorder=-1)
        axi.grid(axis="x", color="lightgray", linestyle="-", zorder=-1)
        axi.spines['top'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['bottom'].set_visible(False)
        axi.spines['left'].set_visible(False)
        axi.xaxis.set_tick_params(bottom=False)
        axi.yaxis.set_tick_params(left=False)

    plt.tight_layout()
    plt.savefig(params['output_dir']+'specific_mass_balance.pdf', format='pdf')
    plt.savefig(params['output_dir']+'specific_mass_balance.png', format='png', dpi=300)

if __name__ == '__main__':
    # load parameter file
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params",
                        type=str,
                        help="Path pointing to the parameter file",
                        required=True)
    arguments, _ = parser.parse_known_args()

    # Load the JSON file with parameters
    with open(arguments.params, 'r') as f:
        params = json.load(f)

    main(params)