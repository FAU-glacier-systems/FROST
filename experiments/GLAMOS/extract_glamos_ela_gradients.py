import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GlacierAnalysis:
    def __init__(self, file_path_glamos, file_path_glamos_bin, file_path_rgi, file_path_glamos_rgi, output_file):
        self.time = np.arange(2000, 2020, dtype=int)
        self.df_glamos = self.load_data(file_path_glamos)
        self.df_glamos_bin = self.load_data(file_path_glamos_bin)
        self.rgi_df = pd.read_csv(file_path_rgi)
        self.glamos_rgi_df = pd.read_csv(file_path_glamos_rgi)
        self.output_file = output_file

        self.results_df = self.process_glaciers()
        self.add_rgi_id()
        self.merge_and_filter_results()
        self.plot_results()

    def load_data(self, file_path):
        df = pd.read_csv(file_path, delimiter=';', skiprows=6)[2:]
        df['end date of observation'] = pd.to_datetime(df['end date of observation'])
        for col in ['annual mass balance', 'upper elevation of bin', 'equilibrium line altitude']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[(df['end date of observation'].dt.year >= 2000) &
                  (df['end date of observation'].dt.year <= 2019)]

    def get_ela_and_specific_mb(self, glacier_df):
        elas = [glacier_df[glacier_df['end date of observation'].dt.year == year]['equilibrium line altitude'].dropna().iloc[0]
                if not glacier_df[glacier_df['end date of observation'].dt.year == year].empty else np.nan
                for year in self.time]
        smb = [glacier_df[glacier_df['end date of observation'].dt.year == year]['annual mass balance'].dropna().iloc[0]
                if not glacier_df[glacier_df['end date of observation'].dt.year == year].empty else np.nan
                for year in self.time]

        return elas, smb

    def extract_gradients(self, glacier_df_bin, elas):
        abl_gradients, acc_gradients = [], []
        for i, year in enumerate(self.time):
            elevation_bin_df = glacier_df_bin[glacier_df_bin['end date of observation'].dt.year == year]
            mb = np.array(elevation_bin_df['annual mass balance'])

            elevation = np.array(elevation_bin_df['upper elevation of bin']) - 50
            abl_gradients.append(self.compute_gradient(mb, elevation, elas[i], 30,
                                                       negative=True))
            acc_gradients.append(self.compute_gradient(mb, elevation, elas[i], 10, negative=False))
        return [abl  if abl is not np.nan else np.nan for abl in abl_gradients], \
               [acc  if acc is not np.nan else np.nan for acc in \
                acc_gradients]

    def compute_gradient(self, mb, elevation, ela, threshold, negative=True):
        if np.isnan(ela): return np.nan
        mask = mb < 0 if negative else mb > 0
        filtered_mb, filtered_elev = mb[mask], elevation[mask]
        if filtered_mb.size > 0:
            adjusted_elevation = filtered_elev - ela
            gradient = np.sum(adjusted_elevation * filtered_mb) / np.sum(adjusted_elevation ** 2)
            return gradient if 0 < gradient < threshold else np.nan
        return np.nan

    def process_glaciers(self):
        records = []
        for glacier_name in self.df_glamos['glacier name'].dropna().unique():
            glacier_df = self.df_glamos[self.df_glamos['glacier name'] == glacier_name]
            elas, smb = self.get_ela_and_specific_mb(glacier_df)
            glacier_df_bin = self.df_glamos_bin[
                (self.df_glamos_bin['glacier name'] == glacier_name)
                & (self.df_glamos_bin['annual mass balance'] > -20000)
                ]

            abl_gradients, acc_gradients = self.extract_gradients(glacier_df_bin, elas)

            record = {
                "Glacier_Name": glacier_name,
                "Mean_ELA": np.nanmean(elas),
                "Annual_Variability_ELA": np.nanstd(elas),
                "ELAS": elas,
                "Years_with_ELA": np.sum(~np.isnan(elas)),
                "annual_mass_balance": np.nanmean(smb),
                "annual_mass_balance_std": np.nanstd(smb),
                "Mean_Ablation_Gradient": np.nanmean(abl_gradients),
                "Annual_Variability_Ablation_Gradient": np.nanstd(abl_gradients),
                "ablation_gradients": abl_gradients,
                "Mean_Accumulation_Gradient": np.nanmean(acc_gradients),
                "Annual_Variability_Accumulation_Gradient": np.nanstd(acc_gradients),
                "accumulation_gradients": acc_gradients
            }
            records.append(record)
        return pd.DataFrame(records)

    def add_rgi_id(self):
        self.results_df = self.results_df.merge(self.glamos_rgi_df,
                                                left_on="Glacier_Name",
                                                right_on="glamos_name", how="left")

    def merge_and_filter_results(self):
        merged_df = pd.merge(self.rgi_df, self.glamos_rgi_df, left_on="rgi_id",
                             right_on="rgi_id", how="inner")
        merged2_df = pd.merge(merged_df, self.results_df, left_on="rgi_id",
                              right_on="rgi_id", how="inner")


        merged2_df = merged2_df.drop('glamos_name_y', axis=1)
        merged2_df = merged2_df.rename(columns={'glamos_name_x': 'glamos_name'})
        filtered_df = merged2_df[merged2_df["area_km2"] >= 1]
        
        filtered_df = filtered_df[filtered_df["Years_with_ELA"] >= 5]
        filtered_df.to_csv(self.output_file, index=False, sep=",")
        print(f"Merged CSV saved to: {self.output_file}")

    def plot_results(self):
        fig, (ax_ela, ax_abl_grad, ax_acc_grad) = plt.subplots(1, 3, figsize=(12, 4))
        self.plot_subplot(ax_ela, "ELAS", "Equilibrium Line Altitude (m)", "Mean_ELA",
                          "Equilibrium Line Altitude:\n{:.0f} ± {:.0f} m", "a")
        self.plot_subplot(ax_abl_grad, "ablation_gradients", "Ablation Gradient (m a$^{-1}$ km$^{-1}$)",
                          "Mean_Ablation_Gradient", "Ablation Gradient:\n{:.2f} ± {:.2f} m a$^{{-1}}$ km$^{{-1}}$", "b")
        self.plot_subplot(ax_acc_grad, "accumulation_gradients", "Accumulation Gradient (m a$^{-1}$ km$^{-1}$)",
                          "Mean_Accumulation_Gradient",
                          "Accumulation Gradient:\n{:.2f} ± {:.2f} m a$^{{-1}}$ km$^{{-1}}$", "c")
        plt.tight_layout()
        plt.savefig('all_gradients.pdf')

    def plot_subplot(self, ax, key, ylabel, mean_key, title_fmt, label):
        std = np.nanstd(self.results_df[mean_key])
        mean = np.nanmean(self.results_df[mean_key])

        ax.add_patch(plt.Rectangle((2010 - 10, mean - std), 20, 2 * std,
                                   facecolor='black', alpha=0.3, edgecolor='black', zorder=5))
        ax.hlines(mean, 2000, 2019, colors='black', linestyles='-', linewidth=1.5,
                  label="Distribution\nof 20 year mean", zorder=10)

        for _, row in self.results_df.iterrows():
            ax.plot(self.time, row[key], linestyle='-', alpha=1.0)

        ax.set_xticks(range(2000, 2021, 10))
        ax.set_xticklabels(range(2000, 2021, 10))
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(title_fmt.format(mean, std))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-5)
        ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-5)
        ax.xaxis.set_tick_params(bottom=False)
        ax.yaxis.set_tick_params(left=False)
        ax.text(-0.3, 1.1, label + ")", transform=ax.transAxes, fontsize=12,
                va='bottom', ha='left', fontweight='bold')


if __name__ == "__main__":
    GlacierAnalysis(
        '../../data/raw/glamos/massbalance_observation.csv',
        '../../data/raw/glamos/massbalance_observation_elevationbins.csv',
        '../../data/raw/RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G'
        '-11_central_europe-attributes.csv',
        '../../data/raw/glamos/GLAMOS_RGI.csv',

        '../../data/raw/glamos/GLAMOS_analysis_results.csv'
    )