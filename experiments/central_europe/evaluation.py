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


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


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
        merged_df_glamos.get('glamos_name', pd.Series([""] * len(merged_df_glamos))),
        merged_df_glamos.get('rgi_id', pd.Series([""] * len(merged_df_glamos))),
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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
        ticks=np.arange(2500, 4001, 500),
    )

    # Keep the top-right empty for spacing
    axes[5].axis('off')

    # Ablation gradient
    scatter_handles_abl = scatter_plot(
        ax=axes[3],
        x=reference_grad_abl,
        y=predicted_grad_abl,
        x_std=reference_grad_abl_std,
        y_std=predicted_grad_abl_std,
        xlabel="GLAMOS Ablation Gradient (m/yr/km)",
        ylabel="Predicted Ablation Gradient (m/yr/km)",
        title="Ablation Gradient",
        glacier_names=glacier_names,
        ticks=np.arange(0, 31, 5),
    )

    # Accumulation gradient
    scatter_handles_acc = scatter_plot(
        ax=axes[4],
        x=reference_grad_acc,
        y=predicted_grad_acc,
        x_std=reference_grad_acc_std,
        y_std=predicted_grad_acc_std,
        xlabel="GLAMOS Accumulation Gradient (m/yr/km)",
        ylabel="Predicted Accumulation Gradient (m/yr/km)",
        title="Accumulation Gradient",
        glacier_names=glacier_names,
        ticks=np.arange(-1, 10, 3),
    )

    scatter_plot(
        ax=axes[1],
        y=merged_df_glamos['sla'].to_numpy(dtype=float),
        x=merged_df_glamos['Mean_ELA'].to_numpy(dtype=float),
        ylabel="End of summer snow line altitude (m)",
        xlabel="GLAMOS ELA (m)",
        title="Equilibrium Line Altitude",
        glacier_names=glacier_names,
        ticks=np.arange(2500, 4001, 500),
    )

    # Predicted ELA vs SLA
    scatter_plot(
        ax=axes[2],
        x=merged_df_glamos['sla'].to_numpy(dtype=float),
        y=merged_df_glamos['ela'].to_numpy(dtype=float),
        xlabel="End of summer snow line altitude (m)",
        ylabel="Predicted ELA (m)",
        title="Equilibrium Line Altitude",
        glacier_names=glacier_names,
        ticks=np.arange(2500, 4001, 500),
    )

    # Single legend outside the plotting grid
    handles = scatter_handles_ela
    fig.legend(handles, glacier_names, loc="upper left",
               bbox_to_anchor=(0.7, 0.5), fontsize=10)

    # Subplot labels
    import string
    axes_with_label = [axes[0], axes[1], axes[2], axes[3], axes[4]]
    labels_subplot = [f"{letter})" for letter in string.ascii_lowercase[:len(axes_with_label)]]
    for ax, label in zip(axes_with_label, labels_subplot):
        ax.text(0, 1.02, label, transform=ax.transAxes,
                fontsize=12, va='bottom', ha='left', fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


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


def _find_first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_sla_vs_ela(merged_df_sla: pd.DataFrame, out_path: Path) -> None:
    """
    Left: SLA vs Predicted ELA (existing).
    Right: Correlation between |ELA - SLA| and velocity MAE (if available in dataframe).
    """
    # Compute absolute SLA-ELA difference
    merged_df_sla = compute_sla_ela_difference(merged_df_sla)
    sla_vals = merged_df_sla['sla'].to_numpy(dtype=float)
    ela_vals = merged_df_sla['ela'].to_numpy(dtype=float)
    area = merged_df_sla['area_km2'].to_numpy(dtype=float)
    ae_ela = np.abs(sla_vals - ela_vals)

    # Try to find a velocity MAE column in the already-merged CSV
    mae_candidates = [
        "MAE_velsurf_mag"
    ]
    mae_col = _find_first_present(merged_df_sla, mae_candidates)

    # Create figure with two subplots if MAE is available, otherwise keep single
    # if mae_col is not None:
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     ax_left, ax_right = axes
    # else:
    fig, ax_left = plt.subplots(1, 1, figsize=(5, 5))
    ax_right = None

    # Left subplot: SLA vs Predicted ELA
    scatter_plot(
        ax=ax_left,
        x=sla_vals,
        y=ela_vals,
        xlabel="End of summer snow line altitude (m)",
        ylabel="Predicted ELA (m)",
        title="End of summer snow line altitude vs. Predicted ELA",
        ticks=np.arange(2200, 4000, 500),
        #color=area,
    )

    # Right subplot: |ELA - SLA| vs Velocity MAE
    if ax_right is not None:
        xdiff = merged_df_sla['difference'].to_numpy(dtype=float)
        ymae = pd.to_numeric(merged_df_sla[mae_col], errors="coerce").to_numpy(dtype=float)

        # Drop NaNs pairwise
        mask = np.isfinite(xdiff) & np.isfinite(ymae)
        x = xdiff[mask]
        y = ymae[mask]

        ax_right.scatter(x, y, s=15, alpha=0.8, edgecolor="k", linewidths=0.2)
        ax_right.set_xlabel("|ELA - SLA| (m)")
        ax_right.set_ylabel("Velocity MAE (m/yr)")
        ax_right.set_title("Velocity MAE vs |ELA - SLA|")
        ax_right.grid(True, linestyle="--", alpha=0.5)

        if len(x) >= 2:
            # Pearson r
            r = np.corrcoef(x, y)[0, 1]
            # Simple linear fit
            coeffs = np.polyfit(x, y, 1)
            xx = np.linspace(0, np.nanmax(x) * 1.05 if np.nanmax(x) > 0 else 1.0, 100)
            yy = np.polyval(coeffs, xx)
            ax_right.plot(xx, yy, color="tab:red", lw=1.5, label=f"Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.1f}")
            ax_right.legend(loc="upper left", fontsize=9)
            ax_right.text(
                0.98, 0.02, f"Pearson r = {r:.2f}",
                transform=ax_right.transAxes,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", alpha=0.8)
            )

        # Nicely padded axes limits
        if len(x) > 0:
            ax_right.set_xlim(0, max(1.0, np.nanmax(x) * 1.05))
        if len(y) > 0:
            ax_right.set_ylim(0, max(1.0, np.nanmax(y) * 1.05))

        # Tight layout for both subplots
        fig.tight_layout()
    else:
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


def print_top_differences(df_with_diff: pd.DataFrame, top_n: int = 10) -> None:
    cols_needed = ['rgi_id', 'Glacier_Name', 'sla', 'ela', 'difference']
    df = df_with_diff.loc[:, [c for c in cols_needed if c in df_with_diff.columns]].copy()
    top_ = df.nlargest(top_n, 'difference')
    print("\nTop glaciers with highest SLA-ELA difference:")
    print("=" * 60)
    for _, row in top_.iterrows():
        rid = row.get('rgi_id', '')
        name = row.get('glamos_name', '')  # may be empty for non-GLAMOS entries
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
    parser = argparse.ArgumentParser(description="Evaluate predicted ELA and gradients using already-merged aggregated results.")
    parser.add_argument("--predicted_results", type=str,
                        default="../../experiments/central_europe/aggregated_results"
                                ".csv")
    parser.add_argument("--output_dir", type=str,
                        default="../central_europe/plots")
    parser.add_argument("--top_n", type=int, default=10, help="How many top SLA-ELA differences to print")
    args = parser.parse_args()

    predicted_path = Path(args.predicted_results).resolve()
    out_dir = Path(args.output_dir).resolve()
    ensure_dir(out_dir)

    # Load the already-merged aggregated results
    df = load_csv(predicted_path)

    # Coerce numeric on all columns we use in plotting/analysis
    numeric_cols = [
        # predictions
        "ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std",
        # geometry/context
        "area_km2",
        "glamos_name"
        # SLA
        "sla",
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

    merged_df_sla = df.dropna(subset=["ela", "sla"]).copy()
    if "area_km2" in merged_df_sla.columns:
        merged_df_sla = merged_df_sla.sort_values(by="area_km2", ascending=True).reset_index(drop=True)

    merged_df_glamos_sla = df.dropna(subset=["ela", "Mean_ELA", "sla"]).copy()

    # Labels for selection plot
    glacier_names = build_glacier_labels(
        merged_df_glamos_sla.get('glamos_name', pd.Series([""] * len(
            merged_df_glamos_sla))),
        merged_df_glamos_sla.get('rgi_id', pd.Series([""] * len(merged_df_glamos_sla))),
    )

    # Plots
    if not merged_df_glamos.empty:
        plot_glamos_vs_predictions(merged_df_glamos, out_dir / "GLAMOS_regional_run.pdf")
    if not merged_df_sla.empty:
        plot_sla_vs_ela(merged_df_sla, out_dir / "SLA_comparison.pdf")
    if not merged_df_glamos_sla.empty:
        _ = plot_glamos_sla_selection(
            merged_df_glamos_sla,
            glacier_names,
            out_dir / "SLA_comparison_selection.pdf",
        )

    # Compute and print Top-N across ALL glaciers with SLA + predictions
    merged_df_sla_with_diff = compute_sla_ela_difference(merged_df_sla)
    print_top_differences(merged_df_sla_with_diff, top_n=args.top_n)


if __name__ == "__main__":
    main()
