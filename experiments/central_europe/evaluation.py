#!/usr/bin/env python3
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frost.visualization.utils import scatter_plot


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_glacier_labels(names: pd.Series, rgi_ids: pd.Series) -> list[str]:
    # "Name (last-digits)" e.g., "...-01706"
    labels = [f"{n} ({r.split('-')[-2]}-{r.split('-')[-1]})" for n, r in zip(names, rgi_ids)]
    #labels = [f"{n}" for n, r in zip(names, rgi_ids)]

    return labels


def plot_glamos_vs_predictions(merged_df_glamos: pd.DataFrame, out_path: Path) -> None:
    # Extract arrays
    Modeled_ela = merged_df_glamos['ela'].to_numpy(dtype=float)
    Modeled_grad_abl = merged_df_glamos['gradabl'].to_numpy(dtype=float)
    Modeled_grad_acc = merged_df_glamos['gradacc'].to_numpy(dtype=float)

    Modeled_ela_std = merged_df_glamos['ela_std'].to_numpy(dtype=float)
    Modeled_grad_abl_std = merged_df_glamos['gradabl_std'].to_numpy(dtype=float)
    Modeled_grad_acc_std = merged_df_glamos['gradacc_std'].to_numpy(dtype=float)

    reference_ela = merged_df_glamos['Mean_ELA'].to_numpy(dtype=float)
    reference_grad_abl = merged_df_glamos['Mean_Ablation_Gradient'].to_numpy(dtype=float)
    reference_grad_acc = merged_df_glamos['Mean_Accumulation_Gradient'].to_numpy(dtype=float)

    reference_ela_std = merged_df_glamos['Annual_Variability_ELA'].to_numpy(dtype=float)
    reference_grad_abl_std = merged_df_glamos['Annual_Variability_Ablation_Gradient'].to_numpy(dtype=float)
    reference_grad_acc_std = merged_df_glamos['Annual_Variability_Accumulation_Gradient'].to_numpy(dtype=float)

    glacier_names = build_glacier_labels(
        merged_df_glamos.get('glamos_name', pd.Series([""] * len(merged_df_glamos))),
        merged_df_glamos.get('rgi_id', pd.Series([""] * len(merged_df_glamos))),
    )

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    # ELA
    scatter_handles_ela = scatter_plot(
        ax=axes[0],
        x=reference_ela,
        y=Modeled_ela,
        x_std=reference_ela_std,
        y_std=Modeled_ela_std,
        xlabel="GLAMOS ELA (m)",
        ylabel="Modeled ELA (m)",
        title="Equilibrium line altitude",
        glacier_names=glacier_names,
        ticks=np.arange(2750, 4000, 250),
    )



    # Ablation gradient
    scatter_handles_abl = scatter_plot(
        ax=axes[1],
        x=reference_grad_abl,
        y=Modeled_grad_abl,
        x_std=reference_grad_abl_std,
        y_std=Modeled_grad_abl_std,
        xlabel="GLAMOS gradient (m w.e. $\,$yr$^{-1}\,$km$^{-1}$)",
        ylabel="Modeled gradient (m$\,$yr$^{-1}\,$km$^{-1}$)",
        title="Ablation",
        glacier_names=glacier_names,
        ticks=np.arange(0, 30, 6),
    )

    # Accumulation gradient
    scatter_handles_acc = scatter_plot(
        ax=axes[2],
        x=reference_grad_acc,
        y=Modeled_grad_acc,
        x_std=reference_grad_acc_std,
        y_std=Modeled_grad_acc_std,
        xlabel="GLAMOS gradient (m w.e. $\,$yr$^{-1}\,$km$^{-1}$)",
        ylabel="Modeled gradient (m$\,$yr$^{-1}\,$km$^{-1}$)",
        title="Accumulation",
        glacier_names=glacier_names,
        ticks=np.arange(-2, 7, 2),
    )

    scatter_handle_vel = scatter_plot(
        ax=axes[7],
        x=merged_df_glamos['Mean_velsurfobs_mag'].to_numpy(dtype=float),
        y=merged_df_glamos['Mean_velsurf_mag'].to_numpy(dtype=float),
        x_std=merged_df_glamos['Std_velsurfobs_mag'].to_numpy(dtype=float) / 2,
        y_std=merged_df_glamos['Std_velsurf_mag'].to_numpy(dtype=float) / 2,
        xlabel="Millan velocity (m$\,$yr$^{-1}$)",
        ylabel="Modeled velocity of inversion (m$\,$yr$^{-1}$)",
        title="Velocity 2000",
        glacier_names=glacier_names,
        ticks=np.arange(0, 54, 12),
    )

    scatter_handle_vel = scatter_plot(
        ax=axes[5],
        x=merged_df_glamos['dhdt_mean'].to_numpy(dtype=float),
        y=merged_df_glamos['annual_mass_balance'].to_numpy(dtype=float)/1000,
        x_std=merged_df_glamos['dhdt_std']/4,
        y_std=merged_df_glamos['annual_mass_balance_std']/1000,
        xlabel="Hugonnet dhdt (m$\,$yr$^{-1}$)",
        ylabel="GLAMOS mass balance (m w.e. $\,$yr$^{-1}$)",
        title="Hugonnet vs GLAMOS",
        glacier_names=glacier_names,
        ticks=np.arange(-2, .1, .5),
    )
    scatter_handle_vel = scatter_plot(
        ax=axes[8],
        x=merged_df_glamos['Mean_velsurfobs_mag'].to_numpy(dtype=float),
        y=merged_df_glamos['vel_ensemble_year20'].to_numpy(dtype=float),
        x_std=merged_df_glamos['Std_velsurfobs_mag'].to_numpy(dtype=float) / 2,
        y_std=merged_df_glamos['Std_velsurf_mag'].to_numpy(dtype=float) / 2,
        xlabel="Millan velocity (m$\,$yr$^{-1}$)",
        ylabel="Modeled velocity after 20 years(m$\,$yr$^{-1}$)",
        title="Velocity 2020",
        glacier_names=glacier_names,
        ticks=np.arange(0, 54, 12),
    )

    scatter_handle_thk = scatter_plot(
        ax=axes[6],
        x=merged_df_glamos['Mean_thk_obs'].to_numpy(dtype=float),
        y=merged_df_glamos['Mean_thk_model_at_obs'].to_numpy(dtype=float),
        x_std=merged_df_glamos['Std_thk_obs'].to_numpy(dtype=float) / 2,
        y_std=merged_df_glamos['Std_thk_model_at_obs'].to_numpy(dtype=float) / 2,
        xlabel="GlaThiDa thickness(m)",
        ylabel="Modeled thickness (m)",
        title="Thickness",
        glacier_names=glacier_names,
        ticks=np.arange(0, 281, 65),
    )

    scatter_handles_acc = scatter_plot(
        ax=axes[3],
        x=merged_df_glamos['dhdt_mean'],
        y=merged_df_glamos['dhdt_ensemble_mean'],
        x_std=merged_df_glamos['dhdt_std']/4,
        y_std=merged_df_glamos['dhdt_ensemble_std'],
        xlabel="Hugonnet dhdt (m yr$^{-1}$)",
        ylabel="Modeled dhdt (m yr$^{-1}$)",
        title="Mean elevation change",
        glacier_names=glacier_names,
        ticks=np.arange(-3, 1.1, 1),
    )

    handles = scatter_handles_ela
    leg = fig.legend(
        handles, glacier_names,
        loc="upper left", bbox_to_anchor=(0.36, 0.67),
        fontsize=10,
    )
    fig.text(
        0.36, 0.5, "Glaciers (RGI7-ID)",  # x, y in figure coordinates
        rotation=90, va="center", ha="center", fontsize=11
    )
    # Optionally adjust alignment
    leg.get_title().set_verticalalignment("bottom")
    leg.get_title().set_horizontalalignment("center")
    leg.get_frame().set_edgecolor("none")
    # Subplot labels

    # Keep the top-right empty for spacing
    axes[4].axis('off')
    import string
    axes_with_label = [axes[0], axes[1], axes[2],  axes[3],axes[5], axes[6], axes[7], axes[8]]
    labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes_with_label)]]
    for ax, label in zip(axes_with_label, labels_subplot):
        ax.text(0, 1.02, label, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left', fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sla_vs_predictions(merged_df_sla: pd.DataFrame, out_path: Path) -> None:
    # ELA
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes = axes.flatten()
    ax = axes[1]
    min = 2300
    max = 3601
    ax.plot([min - 100, max + 100], [min - 100, max + 100], "--", color="black", alpha=0.3,
            zorder=-4,
            label="1:1 Correlation")

    x = merged_df_sla['sla_mean']
    y = merged_df_sla['ela']
    print('ELA mean: ' , np.mean(y))
    print('ELA std: ' , np.std(y))
    hb = ax.hexbin(
        x, y,
        gridsize=30,
        bins=None,
        cmap='viridis_r',
        extent=(min, max, min, max),
        linewidths=0,
        zorder=10,
        mincnt=1,
    )

    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    print(bias)
    y_corr = y - bias
    bc_didf = y_corr - x
    bcmae = np.mean(np.abs(bc_didf))  # Bias-corrected MAE
    correlation = np.corrcoef(x, y)[0, 1]
    txt = (
        f"Correlation: {correlation:.2f}\n"
        f"Mean error: {mae:.0f}\n"
        f"Bias: {bias:.0f}\n"
        f"Bias-corrected MAE: {bcmae:.0f}"
    )
    ax.text(0.95, 0.05, txt,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='bottom',
            horizontalalignment='right')

    ticks = np.arange(min, max, 400)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(min - 100, max + 40)
    ax.set_ylim(min - 100, max + 40)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.tick_params(axis="y", labelrotation=90)

    ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-10)
    ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-10)
    ax.xaxis.set_tick_params(bottom=False)
    ax.yaxis.set_tick_params(left=False)
    ax.set_xlabel('Snow line altitude (m)')
    ax.set_ylabel('Equilibrium line altitude (m)')
    ax.set_title('Equilibrium line altitude')
    ax.set_aspect('equal', adjustable='box')

    cb = fig.colorbar(hb, ax=ax, shrink=0.8)  # shrink to 80%
    cb.set_label("Number of glaciers")  # add label

    ax = axes[0]
    min = -2.
    max = 1.1
    ax.plot([min, max ], [min, max ], "--", color="black", alpha=0.3,
            zorder=-4,
            label="1:1 Correlation")

    merged_df_sla = merged_df_sla.dropna(subset=['dhdt_mean', 'smb_ensemble_mean'])
    x = merged_df_sla['dhdt_mean']
    y = merged_df_sla['dhdt_ensemble_mean']
    
    hb = ax.hexbin(
        x, y,
        gridsize=30,
        bins=None,
        cmap='viridis_r',
        extent=(min, max, min, max),
        linewidths=0,
        zorder=10,
        mincnt=1,
    )

    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    y_corr = y - bias
    bc_didf = y_corr - x
    bcmae = np.mean(np.abs(bc_didf))  # Bias-corrected MAE
    correlation = np.corrcoef(x, y)[0, 1]
    txt = (
        f"Correlation: {correlation:.2f}\n"
        f"Mean error: {mae:.2f}"
        # f"Bias-corrected MAE: {bcmae:.2f}"
    )
    ax.text(0.95, 0.05, txt,
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='bottom', zorder=10,
            horizontalalignment='right')

    ticks = np.arange(min, max, )
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(min - 0.2 , max +0.2)
    ax.set_ylim(min - 0.2, max +0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.tick_params(axis="y", labelrotation=90)

    ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-10)
    ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-10)
    ax.xaxis.set_tick_params(bottom=False)
    ax.yaxis.set_tick_params(left=False)
    ax.set_xlabel('Hugonnet (m yr$^{-1}$)')
    ax.set_ylabel('Modeled elevation change (m yr$^{-1}$)')
    ax.set_title('Mean elevation change')
    ax.set_aspect('equal', adjustable='box')

    cb = fig.colorbar(hb, ax=ax, shrink=0.8)  # shrink to 80%
    cb.set_label("Number of glaciers")  # add label

    import string
    axes_with_label = [axes[0], axes[1]]
    labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes_with_label)]]
    for ax, label in zip(axes_with_label, labels_subplot):
        ax.text(-0.05, 1.02, label, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left', fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def compute_sla_ela_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an absolute SLA-ELA difference column named 'difference'.
    Requires columns 'sla_mean' and 'ela'.
    """
    out = df.copy()
    if 'sla_mean' in out.columns and 'ela' in out.columns:
        out['difference'] = (out['ela'] - out['sla_mean']).abs()
    else:
        out['difference'] = np.nan
    return out


def _find_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def print_top_differences(df_with_diff: pd.DataFrame, top_n: int = 10) -> None:
    cols_needed = ['rgi_id', 'Glacier_Name', 'sla_mean', 'ela', 'difference']
    df = df_with_diff.loc[:, [c for c in cols_needed if c in df_with_diff.columns]].copy()
    top_ = df.nlargest(top_n, 'difference')
    print("\nTop glaciers with highest SLA-ELA difference:")
    print("=" * 60)
    for _, row in top_.iterrows():
        rid = row.get('rgi_id', '')
        name = row.get('glamos_name', '')  # may be empty for non-GLAMOS entries
        sla = row.get('sla_mean', np.nan)
        ela = row.get('ela', np.nan)
        diff = row.get('difference', np.nan)
        try:
            print(f"RGI ID: {rid}")
            print(f"Glacier Name: {name}")
            print(f"SLA: {float(sla):.0f} m")
            print(f"ELA: {float(ela):.0f} m")
            print(f"Difference: {float(diff):.0f} m")
        except Exception:
            print(f"RGI ID: {rid} | Name: {name} | SLA: {sla} | ELA: {ela} | Diff: {diff}")
        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Modeled ELA and gradients using already-merged aggregated results.")
    parser.add_argument("--Modeled_results", type=str,
                        default="../../experiments/central_europe_1000/tables/aggregated_results"
                                ".csv")
    parser.add_argument("--output_dir", type=str,
                        default="../central_europe_1000/plots")
    parser.add_argument("--top_n", type=int, default=10, help="How many top SLA-ELA differences to print")
    args = parser.parse_args()

    Modeled_path = Path(args.Modeled_results).resolve()
    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    # Load the already-merged aggregated results
    df = pd.read_csv(Modeled_path)

    # Coerce numeric on all columns we use in plotting/analysis
    numeric_cols = [
        # predictions
        "ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std",
        # geometry/context
        "area_km2",
        "glamos_name"
        # SLA
        "sla_mean",
        "sla_n"
        # GLAMOS references
        "Mean_ELA", "Mean_Ablation_Gradient", "Mean_Accumulation_Gradient",
        "Annual_Variability_ELA", "Annual_Variability_Ablation_Gradient",
        "Annual_Variability_Accumulation_Gradient",
        # Potential MAE column names (coerce if present)
        "velocity_mae", "mae_velocity", "vel_mae", "mae_vel", "mae_v", "mae",
    ]
    df = coerce_numeric(df, numeric_cols)

    # Subsets for plotting
    merged_df_glamos = df.dropna(subset=["ela", "Mean_ELA", "glamos_name"]).copy()
    if "area_km2" in merged_df_glamos.columns:
        merged_df_glamos = merged_df_glamos.sort_values(by="area_km2",
                                                        ascending=False).reset_index(
            drop=True)

    merged_df_sla = df.dropna(subset=["ela", "sla_mean"]).copy()
    merged_df_sla = merged_df_sla[merged_df_sla["sla_n"] >= 5].copy()

    if "area_km2" in merged_df_sla.columns:
        merged_df_sla = merged_df_sla.sort_values(by="area_km2", ascending=True).reset_index(drop=True)

    merged_df_glamos_sla = df.dropna(subset=["ela", "Mean_ELA", "sla_mean"]).copy()

    # Labels for selection plot
    glacier_names = build_glacier_labels(
        merged_df_glamos_sla.get('glamos_name', pd.Series([""] * len(
            merged_df_glamos_sla))),
        merged_df_glamos_sla.get('rgi_id', pd.Series([""] * len(merged_df_glamos_sla))),
    )

    plot_glamos_vs_predictions(merged_df_glamos, out_dir / "GLAMOS_regional_run.pdf", merged_df_sla)

    plot_sla_vs_predictions(merged_df_sla, out_dir / "SLA_regional_run.pdf")

    # Compute and print Top-N across ALL glaciers with SLA + predictions
    merged_df_sla_with_diff = compute_sla_ela_difference(merged_df_sla)
    print_top_differences(merged_df_sla_with_diff, top_n=args.top_n)


if __name__ == "__main__":
    main()
