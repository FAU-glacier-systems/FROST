from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GlacierAnalysis:
    def __init__(
        self,
        file_path_glamos,
        file_path_glamos_bin,
        file_path_rgi,
        file_path_glamos_rgi,
        output_file,
    ):
        self.time = np.arange(2000, 2020, dtype=int)

        self.file_path_glamos = Path(file_path_glamos)
        self.file_path_glamos_bin = Path(file_path_glamos_bin)
        self.file_path_rgi = Path(file_path_rgi)
        self.file_path_glamos_rgi = Path(file_path_glamos_rgi)
        self.output_file = Path(output_file)

        self.df_glamos = None
        self.df_glamos_bin = None
        self.rgi_df = None
        self.glamos_rgi_df = None
        self.results_df = None
        self.final_df = None

    # ============================================================
    # Public workflow
    # ============================================================

    def run(self):
        self.load_all_data()
        self.results_df = self.process_glaciers()
        self.results_df = self.add_rgi_id(self.results_df)
        self.final_df = self.merge_and_filter_results(self.results_df)
        self.save_results(self.final_df)
        self.plot_results(self.results_df)

    # ============================================================
    # Loading
    # ============================================================

    def load_csv_with_header_offset(self, file_path):
        df = pd.read_csv(file_path, delimiter=";", skiprows=6)[2:].copy()
        df["end date of observation"] = pd.to_datetime(
            df["end date of observation"], errors="coerce"
        )

        numeric_cols = [
            "annual mass balance",
            "upper elevation of bin",
            "equilibrium line altitude",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        year = df["end date of observation"].dt.year
        df = df[(year >= 2000) & (year <= 2019)].copy()
        return df

    def load_all_data(self):
        self.df_glamos = self.load_csv_with_header_offset(self.file_path_glamos)
        self.df_glamos_bin = self.load_csv_with_header_offset(self.file_path_glamos_bin)
        self.rgi_df = pd.read_csv(self.file_path_rgi)
        self.glamos_rgi_df = pd.read_csv(self.file_path_glamos_rgi)

    # ============================================================
    # Glacier processing
    # ============================================================

    def get_yearly_series(self, glacier_df):
        elas = []
        smb = []

        for year in self.time:
            yearly = glacier_df[glacier_df["end date of observation"].dt.year == year]

            ela_values = yearly["equilibrium line altitude"].dropna()
            smb_values = yearly["annual mass balance"].dropna()

            elas.append(ela_values.iloc[0] if not ela_values.empty else np.nan)
            smb.append(smb_values.iloc[0] if not smb_values.empty else np.nan)

        return elas, smb

    def compute_gradient(self, mb, elevation, ela, threshold, negative=True):
        if np.isnan(ela):
            return np.nan

        mask = mb < 0 if negative else mb > 0
        filtered_mb = mb[mask]
        filtered_elev = elevation[mask]

        # require at least 3 unique elevation bins
        if np.unique(filtered_elev).size < 3:
            return np.nan

        if filtered_mb.size == 0:
            return np.nan

        adjusted_elevation = filtered_elev - ela
        denom = np.sum(adjusted_elevation ** 2)

        if denom == 0:
            return np.nan

        gradient = np.sum(adjusted_elevation * filtered_mb) / denom
        return gradient if 0 < gradient < threshold else np.nan

    def extract_gradients(self, glacier_df_bin, elas):
        abl_gradients = []
        acc_gradients = []

        for i, year in enumerate(self.time):
            yearly = glacier_df_bin[
                glacier_df_bin["end date of observation"].dt.year == year
            ]

            mb = yearly["annual mass balance"].to_numpy()
            elevation = yearly["upper elevation of bin"].to_numpy() - 50

            abl = self.compute_gradient(mb, elevation, elas[i], threshold=30, negative=True)
            acc = self.compute_gradient(mb, elevation, elas[i], threshold=10, negative=False)

            abl_gradients.append(abl)
            acc_gradients.append(acc)

        return abl_gradients, acc_gradients

    def process_glaciers(self):
        records = []

        glacier_names = self.df_glamos["glacier name"].dropna().unique()

        for glacier_name in glacier_names:
            glacier_df = self.df_glamos[self.df_glamos["glacier name"] == glacier_name]
            glacier_df_bin = self.df_glamos_bin[
                (self.df_glamos_bin["glacier name"] == glacier_name)
                & (self.df_glamos_bin["annual mass balance"] > -20000)
            ]

            elas, smb = self.get_yearly_series(glacier_df)
            abl_gradients, acc_gradients = self.extract_gradients(glacier_df_bin, elas)

            records.append(
                {
                    "glacier_name": glacier_name,
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
                    "accumulation_gradients": acc_gradients,
                }
            )

        return pd.DataFrame(records)

    # ============================================================
    # Merging
    # ============================================================

    def add_rgi_id(self, results_df):
        return results_df.merge(
            self.glamos_rgi_df,
            left_on="glacier_name",
            right_on="glamos_name",
            how="left",
        )

    def merge_and_filter_results(self, results_df):
        merged = self.rgi_df.merge(
            self.glamos_rgi_df,
            on="rgi_id",
            how="inner",
        )

        merged = merged.merge(
            results_df,
            on="rgi_id",
            how="inner",
            suffixes=("", "_results"),
        )

        # final clean glacier name column
        merged["glacier_name"] = merged["glamos_name"].combine_first(merged["glacier_name"])

        # drop columns you do not want
        drop_cols = [
            "glamos_name_results",
            "rgi_name_results",
            "glac_name",
            "rgi_name_x",
            "rgi_name_y",
            "rgi_name",
            "Glacier_Name",
            "glamos_name",
        ]
        existing_drop_cols = [col for col in drop_cols if col in merged.columns]
        merged = merged.drop(columns=existing_drop_cols)

        filtered = merged[merged["area_km2"] >= 1].copy()
        filtered = filtered[filtered["Years_with_ELA"] >= 10].copy()


        return filtered

    def save_results(self, df):
        # round all numeric columns to 5 decimals
        numeric_cols = df.select_dtypes(include=["float", "int"]).columns
        df[numeric_cols] = df[numeric_cols].round(4)

        # round ELA to integer values
        df["Mean_ELA"] = df["Mean_ELA"].round(0).astype("Int64")

        cols = df.columns.tolist()

        # remove glacier_name and insert at position 1 (second column)
        cols.insert(1, cols.pop(cols.index("glacier_name")))

        df = df[cols]

        df.to_csv(self.output_file, index=False, float_format="%.4f")
        print(f"Merged CSV saved to: {self.output_file}")

    # ============================================================
    # Plotting
    # ============================================================

    def plot_results(self, results_df):
        fig, (ax_ela, ax_abl, ax_acc) = plt.subplots(1, 3, figsize=(12, 4))

        self.plot_subplot(
            ax=ax_ela,
            df=results_df,
            key="ELAS",
            ylabel="Equilibrium Line Altitude (m)",
            mean_key="Mean_ELA",
            title_fmt="Equilibrium Line Altitude:\n{:.0f} ± {:.0f} m",
            panel_label="a",
        )

        self.plot_subplot(
            ax=ax_abl,
            df=results_df,
            key="ablation_gradients",
            ylabel="Ablation Gradient (m a$^{-1}$ km$^{-1}$)",
            mean_key="Mean_Ablation_Gradient",
            title_fmt="Ablation Gradient:\n{:.2f} ± {:.2f} m a$^{{-1}}$ km$^{{-1}}$",
            panel_label="b",
        )

        self.plot_subplot(
            ax=ax_acc,
            df=results_df,
            key="accumulation_gradients",
            ylabel="Accumulation Gradient (m a$^{-1}$ km$^{-1}$)",
            mean_key="Mean_Accumulation_Gradient",
            title_fmt="Accumulation Gradient:\n{:.2f} ± {:.2f} m a$^{{-1}}$ km$^{{-1}}$",
            panel_label="c",
        )

        plt.tight_layout()
        plt.savefig("all_gradients.pdf", dpi=300)
        plt.close()

    def plot_subplot(self, ax, df, key, ylabel, mean_key, title_fmt, panel_label):
        std = np.nanstd(df[mean_key])
        mean = np.nanmean(df[mean_key])

        ax.add_patch(
            plt.Rectangle(
                (2010 - 10, mean - std),
                20,
                2 * std,
                facecolor="black",
                alpha=0.3,
                edgecolor="black",
                zorder=5,
            )
        )
        ax.hlines(
            mean,
            2000,
            2019,
            colors="black",
            linestyles="-",
            linewidth=1.5,
            label="Distribution\nof 20 year mean",
            zorder=10,
        )

        for _, row in df.iterrows():
            ax.plot(self.time, row[key], linestyle="-", alpha=1.0)

        ax.set_xticks([2000, 2010, 2020])
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_title(title_fmt.format(mean, std))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-5)
        ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-5)
        ax.xaxis.set_tick_params(bottom=False)
        ax.yaxis.set_tick_params(left=False)

        ax.text(
            -0.3,
            1.1,
            f"{panel_label})",
            transform=ax.transAxes,
            fontsize=12,
            va="bottom",
            ha="left",
            fontweight="bold",
        )


if __name__ == "__main__":
    analysis = GlacierAnalysis(
        "../../data/raw/glamos/massbalance_observation.csv",
        "../../data/raw/glamos/massbalance_observation_elevationbins.csv",
        "../../data/raw/RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe-attributes.csv",
        "../../data/raw/glamos/GLAMOS_RGI.csv",
        "../../data/raw/glamos/GLAMOS_analysis_results.csv",
    )
    analysis.run()