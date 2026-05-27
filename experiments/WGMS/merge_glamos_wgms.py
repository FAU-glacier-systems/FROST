from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================

WGMS_FILE = Path("tables/wgms_ELA_gradients.csv")
GLAMOS_FILE = Path("../../data/raw/glamos/GLAMOS_analysis_results.csv")
KEY = "rgi_id"

OUTPUT_ELA_COMPARISON = Path("tables/ELA_comparison_wgms_vs_glamos.csv")
OUTPUT_MB_COMPARISON = Path("tables/MB_comparison_wgms_vs_glamos.csv")
OUTPUT_COMBINED = Path("tables/combined_ela_gradients.csv")


# ============================================================
# Data loading and preparation
# ============================================================

def load_data(wgms_file: Path, glamos_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load WGMS and GLAMOS tables."""
    wgms = pd.read_csv(wgms_file)
    glamos = pd.read_csv(glamos_file)
    return wgms, glamos


def prepare_wgms(wgms: pd.DataFrame) -> pd.DataFrame:
    """Keep and rename relevant WGMS columns."""
    cols = [
        KEY,
        "country",
        "glacier_name",
        "ela_mean",
        "lower_gradient",
        "upper_gradient",
        "annual_mass_balance_mean",
        "annual_mass_balance_std",
        "n_years_mb"
    ]
    wgms = wgms[cols].copy()

    wgms = wgms.rename(
        columns={
            "ela_mean": "ELA_wgms",
            "lower_gradient": "abl_wgms",
            "upper_gradient": "acc_wgms",
            "annual_mass_balance_mean": "mb_wgms",
            "annual_mass_balance_std": "mb_wgms_std"
        }
    )
    return wgms


def prepare_glamos(glamos: pd.DataFrame) -> pd.DataFrame:
    """Keep and rename relevant GLAMOS columns."""
    cols = [
        KEY,
        "glacier_name",
        "Mean_ELA",
        "Mean_Ablation_Gradient",
        "Mean_Accumulation_Gradient",
        "annual_mass_balance",
        "annual_mass_balance_std",
        "Years_with_ELA"
    ]
    glamos = glamos[cols].copy()

    glamos = glamos.rename(
        columns={
            "Mean_ELA": "ELA_glamos",
            "Mean_Ablation_Gradient": "abl_glamos",
            "Mean_Accumulation_Gradient": "acc_glamos",
            "annual_mass_balance": "mb_glamos_mm",
            "annual_mass_balance_std" : "mb_glamos_mm_std"
        }
    )

    # GLAMOS annual_mass_balance is in mm w.e. -> convert to m w.e.
    glamos["mb_glamos"] = glamos["mb_glamos_mm"] / 1000.0
    glamos["mb_glamos_std"] = glamos["mb_glamos_mm_std"] / 1000.0
    glamos = glamos.drop(columns=["mb_glamos_mm", "mb_glamos_mm_std"])
    return glamos


# ============================================================
# Comparison utilities
# ============================================================

def compute_stats(df: pd.DataFrame, col_a: str, col_b: str, diff_name: str) -> pd.DataFrame:
    """Return dataframe with difference column added."""
    out = df.dropna(subset=[col_a, col_b]).copy()
    out[diff_name] = out[col_a] - out[col_b]
    return out


def print_basic_stats(df: pd.DataFrame, diff_col: str, col_a: str, col_b: str, label: str) -> None:
    """Print basic comparison statistics."""
    rmse = np.sqrt((df[diff_col] ** 2).mean())
    corr = df[col_a].corr(df[col_b])

    print(f"{label}")
    print(f"Common glaciers: {len(df)}")
    print(f"Mean diff ({col_a} minus {col_b}): {df[diff_col].mean():.3f}")
    print(f"Median diff: {df[diff_col].median():.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Correlation: {corr:.3f}")
    print()


def find_nonmatching_glaciers(wgms: pd.DataFrame, glamos: pd.DataFrame) -> None:
    """Print glaciers that only exist in one of the two datasets."""
    merged = wgms.merge(glamos, on=KEY, how="outer", indicator=True)
    merged["glacier_name"] = merged["glacier_name_x"].combine_first(merged["glacier_name_y"])

    only_wgms = merged[
        (merged["_merge"] == "left_only") & (merged["country"] == "CH")
    ][["glacier_name"]]

    only_glamos = merged[merged["_merge"] == "right_only"][["glacier_name"]]

    print("Only in WGMS (Switzerland):")
    if len(only_wgms) > 0:
        print(only_wgms.to_string(index=False))
    else:
        print("None")
    print()

    print("Only in GLAMOS:")
    if len(only_glamos) > 0:
        print(only_glamos.to_string(index=False))
    else:
        print("None")
    print()


# ============================================================
# Final combined table
# ============================================================

def build_combined_table(wgms: pd.DataFrame, glamos: pd.DataFrame) -> pd.DataFrame:
    """Build final table with WGMS priority over GLAMOS."""
    merged = wgms.merge(glamos, on=KEY, how="outer", suffixes=("_wgms", "_glamos"))
    merged["glacier_name"] = merged["glacier_name_wgms"].combine_first(merged["glacier_name_glamos"])

    # ELA
    merged["ela"] = merged["ELA_glamos"].combine_first(merged["ELA_wgms"])
    merged["ela_source"] = np.where(
        merged["ELA_glamos"].notna(),
        "GLAMOS",
        np.where(merged["ELA_wgms"].notna(), "WGMS", pd.NA),
    )

    # Ablation gradient
    merged["gradabl"] = merged["abl_glamos"].combine_first(merged["abl_wgms"])
    merged["gradabl_source"] = np.where(
        merged["abl_glamos"].notna(),
        "GLAMOS",
        np.where(merged["abl_wgms"].notna(), "WGMS", pd.NA),
    )

    # Accumulation gradient
    merged["gradacc"] = merged["acc_glamos"].combine_first(merged["acc_wgms"])
    merged["gradacc_source"] = np.where(
        merged["acc_glamos"].notna(),
        "GLAMOS",
        np.where(merged["acc_wgms"].notna(), "WGMS", pd.NA),
    )

    # annual_mass_balance
    merged["annual_mass_balance"] = merged["mb_glamos"].combine_first(merged["mb_wgms"])
    merged["annual_mass_balance_source"] = np.where(
        merged["mb_glamos"].notna(),
        "GLAMOS",
        np.where(merged["mb_wgms"].notna(), "WGMS", pd.NA),
    )

    # std
    merged["annual_mass_balance_std"] = merged["mb_glamos_std"].combine_first(merged["mb_wgms_std"])
    merged["annual_mass_balance_std_source"] = np.where(
        merged["mb_glamos_std"].notna(),
        "GLAMOS",
        np.where(merged["mb_wgms_std"].notna(), "WGMS", pd.NA),
    )

    # number of years (MB / ELA)
    merged["n_years"] = merged["n_years_mb"].combine_first(merged["Years_with_ELA"])
    merged["n_years_source"] = np.where(
        merged["n_years_mb"].notna(),
        "WGMS",
        np.where(merged["Years_with_ELA"].notna(), "GLAMOS", pd.NA),
    )
    final = merged[
        [
            KEY,
            "glacier_name",
            "country",
            "ela",
            "ela_source",
            "gradabl",
            "gradabl_source",
            "gradacc",
            "gradacc_source",
            "annual_mass_balance",
            "annual_mass_balance_std",
            "annual_mass_balance_source",
            "n_years",
            "n_years_source",

        ]
    ].copy()
    # remove glaciers without rgi_id
    final = final[final[KEY].notna()].copy()
    mask = final["country"].isna() & (final["annual_mass_balance_source"] == "GLAMOS")
    final.loc[mask, "country"] = "CH"

    # keep only rows where at least one parameter exists
    final = final[
        final[["ela", "gradabl", "gradacc", "annual_mass_balance"]].notna().any(axis=1)
    ].copy()

    return final


# ============================================================
# Plotting
# ============================================================

def make_color_dict(names: np.ndarray) -> dict:
    """Assign a distinct color to each glacier."""
    cmap = plt.cm.tab20
    n = max(len(names), 1)
    return {name: cmap(i / n) for i, name in enumerate(names)}


def scatter_comparison(
    ax: plt.Axes,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    name_col: str,
    colors: dict,
    xlabel: str,
    ylabel: str,
    title: str,
    pad: float,
) -> None:
    """Create one scatter comparison subplot."""
    names = df[name_col].dropna().unique()

    for name in names:
        subset = df[df[name_col] == name]
        ax.scatter(
            subset[xcol],
            subset[ycol],
            color=colors.get(name, "grey"),
            label=name,
        )

    mn = min(df[xcol].min(), df[ycol].min()) - pad
    mx = max(df[xcol].max(), df[ycol].max()) + pad

    ax.plot([mn, mx], [mn, mx])
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_comparisons(
    ela_df: pd.DataFrame,
    grad_df: pd.DataFrame,
    mb_df: pd.DataFrame,
) -> None:
    """Plot ELA, ablation gradient, and annual mass balance comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    all_names = pd.concat(
        [ela_df["glacier_name"], grad_df["glacier_name"], mb_df["glacier_name"]],
        ignore_index=True,
    ).dropna().unique()
    colors = make_color_dict(all_names)
    axes = axes.ravel()
    scatter_comparison(
        ax=axes[0],
        df=ela_df,
        xcol="ELA_glamos",
        ycol="ELA_wgms",
        name_col="glacier_name",
        colors=colors,
        xlabel="ELA GLAMOS [m]",
        ylabel="ELA WGMS [m]",
        title="ELA comparison",
        pad=50,
    )

    scatter_comparison(
        ax=axes[1],
        df=grad_df,
        xcol="abl_glamos",
        ycol="abl_wgms",
        name_col="glacier_name",
        colors=colors,
        xlabel="Ablation gradient GLAMOS",
        ylabel="Ablation gradient WGMS",
        title="Ablation gradient comparison",
        pad=1,
    )

    scatter_comparison(
        ax=axes[2],
        df=mb_df,
        xcol="mb_glamos",
        ycol="mb_wgms",
        name_col="glacier_name",
        colors=colors,
        xlabel="Annual MB GLAMOS [m w.e.]",
        ylabel="Annual MB WGMS [m w.e.]",
        title="Annual mass balance comparison",
        pad=0.2,
    )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=colors[name], label=name)
        for name in all_names
    ]

    axes[3].set_axis_off()
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig("glamos_wgms_comparison.pdf", bbox_inches="tight")


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    wgms_raw, glamos_raw = load_data(WGMS_FILE, GLAMOS_FILE)

    wgms = prepare_wgms(wgms_raw)
    glamos = prepare_glamos(glamos_raw)

    # ----------------------
    # ELA comparison
    # ----------------------
    ela_merged = wgms.merge(glamos, on=KEY, how="inner", suffixes=("_wgms", "_glamos"))
    ela_merged["glacier_name"] = ela_merged["glacier_name_wgms"].combine_first(ela_merged["glacier_name_glamos"])

    ela_intersection = compute_stats(
        ela_merged, col_a="ELA_wgms", col_b="ELA_glamos", diff_name="ELA_diff"
    )

    print_basic_stats(
        ela_intersection,
        diff_col="ELA_diff",
        col_a="ELA_wgms",
        col_b="ELA_glamos",
        label="ELA comparison",
    )

    ela_intersection.to_csv(OUTPUT_ELA_COMPARISON, index=False)

    # ----------------------
    # Gradient comparison
    # ----------------------
    grad_merged = wgms.merge(glamos, on=KEY, how="inner", suffixes=("_wgms", "_glamos"))
    grad_merged["glacier_name"] = grad_merged["glacier_name_wgms"].combine_first(grad_merged["glacier_name_glamos"])

    grad_intersection = compute_stats(
        grad_merged, col_a="abl_wgms", col_b="abl_glamos", diff_name="grad_diff"
    )

    print_basic_stats(
        grad_intersection,
        diff_col="grad_diff",
        col_a="abl_wgms",
        col_b="abl_glamos",
        label="Ablation gradient comparison",
    )

    # ----------------------
    # Annual mass balance comparison
    # ----------------------
    mb_merged = wgms.merge(glamos, on=KEY, how="inner", suffixes=("_wgms", "_glamos"))
    mb_merged["glacier_name"] = mb_merged["glacier_name_wgms"].combine_first(mb_merged["glacier_name_glamos"])

    mb_intersection = compute_stats(
        mb_merged, col_a="mb_wgms", col_b="mb_glamos", diff_name="mb_diff"
    )

    print_basic_stats(
        mb_intersection,
        diff_col="mb_diff",
        col_a="mb_wgms",
        col_b="mb_glamos",
        label="Annual mass balance comparison",
    )

    mb_intersection.to_csv(OUTPUT_MB_COMPARISON, index=False)

    # ----------------------
    # Non-matching glaciers
    # ----------------------
    find_nonmatching_glaciers(wgms, glamos)

    # ----------------------
    # Final combined table
    # ----------------------
    final = build_combined_table(wgms, glamos)
    final.to_csv(OUTPUT_COMBINED, index=False, float_format="%.3f")

    print(final.head())
    print()
    print(f"Number of glaciers in final table: {len(final)}")

    # ----------------------
    # Plot
    # ----------------------
    plot_comparisons(ela_intersection, grad_intersection, mb_intersection)


if __name__ == "__main__":
    main()