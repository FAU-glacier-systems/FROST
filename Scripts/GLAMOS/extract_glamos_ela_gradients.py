#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GlacierAnalysis:
    def __init__(self, file_path_glamos, file_path_glamos_bin):
        self.time = np.arange(2000, 2020, dtype=int)
        self.df_glamos = self.load_data(file_path_glamos)
        self.df_glamos_bin = self.load_data(file_path_glamos_bin)
        self.results = self.initialize_results()
        self.process_glaciers()
        self.results_df = pd.DataFrame(self.results)
        self.save_results("../../Data/GLAMOS/glacier_analysis_results.csv")
        self.plot_results()

    def load_data(self, file_path):
        df = pd.read_csv(file_path, delimiter=';', skiprows=6)[2:]
        df['end date of observation'] = pd.to_datetime(df['end date of observation'])
        if 'annual mass balance' in df.columns:
            df['annual mass balance'] = pd.to_numeric(df['annual mass balance'])
        if 'upper elevation of bin' in df.columns:
            df['upper elevation of bin'] = pd.to_numeric(
                df['upper elevation of bin'])
        return df[(df['end date of observation'].dt.year >= 2000) & (
                df['end date of observation'].dt.year <= 2019)]

    def initialize_results(self):
        return {"Glacier Name": [], "Mean ELA": [], "Mean Ablation Gradient": [],
                "Mean Accumulation Gradient": [],
                "ELAS": [], "ablation gradients": [], "accumulation gradients": [],
                "Years with ELA": [], "Annual Variability ELA": [], "Annual Variability "
                                                                 "Ablation Gradient": [],
                "Annual Variability Accumulation Gradient": []}

    def get_ela_and_specific_mb(self, glacier_df):
        elas, specific_mass_balances = [], []
        for year in self.time:
            entry = glacier_df[glacier_df['end date of observation'].dt.year == year]
            if not entry.empty:
                specific_mass_balances.append(
                    float(entry['annual mass balance'].iloc[0]) / 1000)
                elas.append(int(entry['equilibrium line altitude'].iloc[0]))
            else:
                specific_mass_balances.append(np.nan)
                elas.append(np.nan)
        return elas, specific_mass_balances

    def extract_gradients(self, glacier_df_bin, elas):
        abl_gradients, acc_gradients = [], []
        for i, year in enumerate(self.time):
            elevation_bin_df = glacier_df_bin[
                glacier_df_bin['end date of observation'].dt.year == year]
            mb, elevation = np.array(
                elevation_bin_df['annual mass balance']), np.array(
                elevation_bin_df['upper elevation of bin']) - 50
            abl_gradients.append(
                self.compute_gradient(mb, elevation, elas[i], negative=True))
            acc_gradients.append(
                self.compute_gradient(mb, elevation, elas[i], negative=False))
        return abl_gradients, acc_gradients

    def compute_gradient(self, mb, elevation, ela, negative=True):
        mask = mb < 0 if negative else mb > 0
        filtered_mb, filtered_elev = mb[mask], elevation[mask]
        if filtered_mb.size > 0:
            adjusted_elevation = filtered_elev - ela
            gradient = np.sum(adjusted_elevation * filtered_mb) / np.sum(
                adjusted_elevation ** 2)
            return gradient if 0 < gradient < 20 else np.nan
        return np.nan

    def process_glaciers(self):
        for glacier_name in self.df_glamos['glacier name'].unique():
            print(glacier_name)
            if pd.isna(glacier_name): continue
            glacier_df = self.df_glamos[
                self.df_glamos['glacier name'] == glacier_name]
            elas, _ = self.get_ela_and_specific_mb(glacier_df)
            glacier_df_bin = self.df_glamos_bin[
                self.df_glamos_bin['glacier name'] == glacier_name]
            abl_gradients, acc_gradients = self.extract_gradients(glacier_df_bin,
                                                                  elas)

            self.store_results(glacier_name, elas, abl_gradients, acc_gradients)

    def store_results(self, glacier_name, elas, abl_gradients, acc_gradients):
        self.results["Glacier Name"].append(glacier_name)
        self.results["Mean ELA"].append(np.nanmean(elas))
        self.results["Annual Variability ELA"].append(np.nanstd(elas))
        self.results["ELAS"].append(elas)
        self.results["Mean Ablation Gradient"].append(np.nanmean(abl_gradients))
        self.results["Annual Variability Ablation Gradient"].append(
            np.nanstd(abl_gradients))
        self.results["Years with ELA"].append(np.sum(~np.isnan(elas)))
        self.results["ablation gradients"].append(abl_gradients)
        self.results["Mean Accumulation Gradient"].append(np.nanmean(acc_gradients))
        self.results["Annual Variability Accumulation Gradient"].append(
            np.nanstd(acc_gradients))
        self.results["accumulation gradients"].append(acc_gradients)

    def save_results(self, filename):
        self.results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

    def plot_results(self):
        fig, (ax_ela, ax_abl_grad, ax_acc_grad) = plt.subplots(1, 3,
                                                               figsize=(15, 5))
        self.plot_subplot(ax_ela, "ELAS", "ELA", "Mean ELA",
                          "Mean ELA: {:.0f} ± {:.0f} m")
        self.plot_subplot(ax_abl_grad, "ablation gradients", "Ablation Gradient",
                          "Mean Ablation Gradient",
                          "Mean Ablation Gradient: \n{:.4f} ± {:.4f} m/a/km")
        self.plot_subplot(ax_acc_grad, "accumulation gradients",
                          "Accumulation Gradient", "Mean Accumulation Gradient",
                          "Mean Accumulation Gradient: \n{:.4f} ± {:.4f} m/a/km",
                          show_legend=True)
        plt.tight_layout()
        plt.savefig('../../Plots/all_gradients.png', dpi=300)

    def plot_subplot(self, ax, key, ylabel, mean_key, title_fmt, show_legend=False):
        filtered_df = self.results_df[self.results_df["Years with ELA"] > 15]

        data_array = np.array(filtered_df[key].tolist())
        yearly_mean = np.nanmean(data_array, axis=0)
        for index, row in filtered_df.iterrows():
            ax.plot(self.time, row[key], linestyle='-', alpha=0.3,
                    label=row['Glacier Name'])

        ax.boxplot(filtered_df[mean_key], positions=[2020], widths=0.8)
        ax.plot(self.time, yearly_mean, linestyle='-', color='black')
        ax.set_xticks(range(2000, 2021, 5))
        ax.set_xticklabels(range(2000, 2021, 5))
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(title_fmt.format(np.nanmean(filtered_df[mean_key]),
                                      np.nanstd(filtered_df[mean_key])))
        if show_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))


if __name__ == "__main__":
    GlacierAnalysis('../../Data/GLAMOS/massbalance_observation.csv',
                    '../../Data/GLAMOS/massbalance_observation_elevationbins.csv')
