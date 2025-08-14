#!/usr/bin/env python3
# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from frost.visualization.utils import scatter_plot


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def merge_and_cleanup(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str) -> pd.DataFrame:
    merged = pd.merge(left, right, on=on, how=how, suffixes=('', '_drop'))
    # Remove "_drop" duplicate columns from merges
    merged = merged.loc[:, ~merged.columns.str.endswith('_drop')].copy()
    return merged


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_glacier_labels(names: pd.Series, rgi_ids: pd.Series) -> list[str]:
    # "Name (last-digits)" e.g., "...-01706"
    labels = [f"{n} ({r.split('-')[-1]})" for n, r in zip(names, rgi_ids)]
    return labels


def plot_glamos_vs_predictions(merged_df_glamos: pd.DataFrame, out_path: Path) -> None:
    # Extract arrays
    predicted_ela = merged_df_glamos['ela'].to_numpy(dtype=float)
    predicted_grad_abl = merged_df_glamos['gradabl'].to_numpy(dtype=float)
    predicted_grad_acc = merged_df_glamos['gradacc'].to_numpy(dtype=float)

    predicted_ela_std = merged_df_glamos['ela_std'].to_numpy(dtype=float)
    predicted_grad_abl_std = merged_df_glamos['gradabl_std'].to_numpy(dtype=float)
    predicted_grad_acc_std = merged_df_glamos['gradacc_std'].to_numpy(dtype=float)

    reference_ela = merged_df_glamos['Mean_ELA'].to_numpy(dtype=float)
    reference_grad_abl = merged_df_glamos['Mean_Ablation_Gradient'].to_numpy(dtype=float)
    reference_grad_acc = merged_df_glamos['Mean_Accumulation_Gradient'].to_numpy(dtype=float)

    reference_ela_std = merged_df_glamos['Annual_Variability_ELA'].to_numpy(dtype=float)
    reference_grad_abl_std = merged_df_glamos['Annual_Variability_Ablation_Gradient'].to_numpy(dtype=float)
    reference_grad_acc_std = merged_df_glamos['Annual_Variability_Accumulation_Gradient'].to_numpy(dtype=float)

    glacier_names = build_glacier_labels(
        merged_df_glamos['Glacier_Name'], merged_df_glamos['rgi_id']
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    # ELA
    scatter_handles_ela = scatter_plot(
        ax=axes[0],
        x=reference_ela,
        y=predicted_ela,
        x_std=reference_ela_std,
        y_std=predicted_ela_std,
        xlabel="GLAMOS ELA (m)",
        ylabel="Predicted ELA (m)",
        title="Equilibrium Line Altitude",
        glacier_names=glacier_names,
        ticks=np.arange(2500, 3501, 250),
    )

    # Keep the top-right empty for spacing
    axes[1].axis('off')

    # Ablation gradient
    scatter_handles_abl = scatter_plot(
        ax=axes[2],
        x=reference_grad_abl,
        y=predicted_grad_abl,
        x_std=reference_grad_abl_std,
        y_std=predicted_grad_abl_std,
        xlabel="GLAMOS Ablation Gradient (m/yr/km)",
        ylabel="Predicted Ablation Gradient (m/yr/km)",
        title="Ablation Gradient",
        glacier_names=glacier_names,
        ticks=np.arange(0, 26, 5),
    )

    # Accumulation gradient
    scatter_handles_acc = scatter_plot(
        ax=axes[3],
        x=reference_grad_acc,
        y=predicted_grad_acc,
        x_std=reference_grad_acc_std,
        y_std=predicted_grad_acc_std,
        xlabel="GLAMOS Accumulation Gradient (m/yr/km)",
        ylabel="Predicted Accumulation Gradient (m/yr/km)",
        title="Accumulation Gradient",
        glacier_names=glacier_names,
        ticks=np.arange(0, 16, 3),
    )

    # Single legend outside the plotting grid
    handles = scatter_handles_ela
    fig.legend(handles, glacier_names, loc="upper left",
               bbox_to_anchor=(0.6, 0.95), fontsize=10)

    # Subplot labels
    import string
    axes_with_label = [axes[0], axes[2], axes[3]]
    labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes_with_label)]]
    for ax, label in zip(axes_with_label, labels_subplot):
        ax.text(0, 1.02, label, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left', fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sla_vs_ela(merged_df_sla: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    sla_vals = merged_df_sla['sla'].to_numpy(dtype=float)
    ela_vals = merged_df_sla['ela'].to_numpy(dtype=float)
    ae_ela = np.abs(sla_vals - ela_vals)

    scatter_plot(
        ax=ax,
        x=sla_vals,
        y=ela_vals,
        xlabel="End of summer snow line altitude (m)",
        ylabel="Predicted ELA (m)",
        title="End of summer snow line altitude vs. Predicted ELA",
        ticks=np.arange(2200, 4000, 500),
        color=ae_ela,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_glamos_sla_selection(merged_df_glamos_sla: pd.DataFrame, glacier_names: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    axes = axes.flatten()

    # GLAMOS ELA vs SLA
    scatter_plot(
        ax=axes[0],
        y=merged_df_glamos_sla['sla'].to_numpy(dtype=float),
        x=merged_df_glamos_sla['Mean_ELA'].to_numpy(dtype=float),
        ylabel="End of summer snow line altitude (m)",
        xlabel="GLAMOS ELA (m)",
        title="",
        glacier_names=glacier_names,
        ticks=np.arange(2200, 4000, 500),
    )

    # Predicted ELA vs SLA
    scatter_plot(
        ax=axes[1],
        x=merged_df_glamos_sla['sla'].to_numpy(dtype=float),
        y=merged_df_glamos_sla['ela'].to_numpy(dtype=float),
        xlabel="End of summer snow line altitude (m)",
        ylabel="Predicted ELA (m)",
        title="SLA vs Predicted ELA on Selected Glaciers",
        glacier_names=glacier_names,
        ticks=np.arange(2200, 4000, 500),
    )

    # Compute differences for reporting
    merged_df_glamos_sla = merged_df_glamos_sla.copy()
    merged_df_glamos_sla['difference'] = (
        merged_df_glamos_sla['ela'] - merged_df_glamos_sla['sla']
    ).abs()

    # Legend anchored outside
    handles_for_legend = []  # scatter_plot returns handles; we only need labels
    fig.legend(handles_for_legend, glacier_names, loc="upper left",
               bbox_to_anchor=(1, 0.9), fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return merged_df_glamos_sla


def compute_sla_ela_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an absolute SLA-ELA difference column named 'difference'.
    Requires columns 'sla' and 'ela'.
    """
    out = df.copy()
    if 'sla' in out.columns and 'ela' in out.columns:
        out['difference'] = (out['ela'] - out['sla']).abs()
    else:
        out['difference'] = np.nan
    return out


def print_top_differences(df_with_diff: pd.DataFrame, top_n: int = 10) -> None:
    cols_needed = ['rgi_id', 'Glacier_Name', 'sla', 'ela', 'difference']
    df = df_with_diff.loc[:, [c for c in cols_needed if c in df_with_diff.columns]].copy()
    top_ = df.nlargest(top_n, 'difference')
    print("\nTop glaciers with highest SLA-ELA difference:")
    print("=" * 60)
    for _, row in top_.iterrows():
        rid = row.get('rgi_id', '')
        name = row.get('Glacier_Name', '')  # may be empty for non-GLAMOS entries
        sla = row.get('sla', np.nan)
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
    parser = argparse.ArgumentParser(description="Evaluate predicted ELA and gradients against GLAMOS and SLA.")
    parser.add_argument("--glamos_results", type=str,
                        default="../../data/raw/glamos/GLAMOS_analysis_results.csv")
    parser.add_argument("--predicted_results", type=str,
                        default="../../experiments/central_europe_sliding"
                                "/aggregated_results.csv")
    parser.add_argument("--sla_path", type=str,
                        default="../../data/raw/central_europe"
                                "/Alps_EOS_SLA_2000-2019_mean.csv")
    parser.add_argument("--output_dir", type=str,
                        default="../central_europe_sliding/plots")
    parser.add_argument("--top_n", type=int, default=10, help="How many top SLA-ELA differences to print")
    args = parser.parse_args()

    glamos_path = Path(args.glamos_results).resolve()
    predicted_path = Path(args.predicted_results).resolve()
    sla_path = Path(args.sla_path).resolve()
    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    # Load
    glamos_df = load_csv(glamos_path)
    predicted_df = load_csv(predicted_path)
    sla_df = load_csv(sla_path)

    # Coerce numeric where needed before merges
    numeric_cols_pred = ["ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std", "area_km2"]
    numeric_cols_glamos = [
        "Mean_ELA", "Mean_Ablation_Gradient", "Mean_Accumulation_Gradient",
        "Annual_Variability_ELA", "Annual_Variability_Ablation_Gradient",
        "Annual_Variability_Accumulation_Gradient", "area_km2"
    ]
    numeric_cols_sla = ["sla", "area_km2"]

    predicted_df = coerce_numeric(predicted_df, numeric_cols_pred)
    glamos_df = coerce_numeric(glamos_df, numeric_cols_glamos)
    sla_df = coerce_numeric(sla_df, numeric_cols_sla)

    # Merge GLAMOS with predictions (keep only rows where predicted ELA exists)
    merged_df_glamos = merge_and_cleanup(glamos_df, predicted_df, on="rgi_id", how="left")
    merged_df_glamos = merged_df_glamos.dropna(subset=["ela"]).copy()
    if "area_km2" in merged_df_glamos.columns:
        merged_df_glamos = merged_df_glamos.sort_values(by="area_km2", ascending=True).reset_index(drop=True)

    # Merge predictions with SLA (this covers ALL glaciers having both SLA and predictions)
    merged_df_sla = merge_and_cleanup(predicted_df, sla_df, on="rgi_id", how="inner")
    if "area_km2" in merged_df_sla.columns:
        merged_df_sla = merged_df_sla.sort_values(by="area_km2", ascending=True).reset_index(drop=True)

    # GLAMOS subset additionally merged with SLA for the selection plot
    merged_df_glamos_sla = merge_and_cleanup(merged_df_glamos, sla_df, on="rgi_id", how="inner")

    # Labels for selection plot
    glacier_names = build_glacier_labels(
        merged_df_glamos_sla.get('Glacier_Name', pd.Series([""] * len(merged_df_glamos_sla))),
        merged_df_glamos_sla.get('rgi_id', pd.Series([""] * len(merged_df_glamos_sla))),
    )

    # Plots (unchanged)
    plot_glamos_vs_predictions(merged_df_glamos, out_dir / "GLAMOS_regional_run.pdf")
    plot_sla_vs_ela(merged_df_sla, out_dir / "SLA_comparison.pdf")
    _ = plot_glamos_sla_selection(
        merged_df_glamos_sla,
        glacier_names,
        out_dir / "SLA_comparison_selection.pdf",
    )

    # NEW: compute and print Top-N across ALL glaciers with SLA + predictions
    merged_df_sla_with_diff = compute_sla_ela_difference(merged_df_sla)
    print_top_differences(merged_df_sla_with_diff, top_n=args.top_n)


if __name__ == "__main__":
    main()
