from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================

INPUT_FILE_MB_Bands = Path("../../data/raw/DOI-WGMS-FoG-2025-02b/data/mass_balance_band.csv")
INPUT_FILE_MB = Path("../../data/raw/DOI-WGMS-FoG-2025-02b/data/mass_balance.csv")
LOOKUP_FILE = Path("tables/RGI6-7.csv")
OUTPUT_CSV = Path("tables/wgms_ELA_gradients.csv")
PLOT_DIR = Path("Plots")

COUNTRIES = ["AT", "IT", "FR", "CH", "DE"]
YEAR_MIN = 2000
YEAR_MAX = 2019
MIN_UNIQUE_YEARS = 10
MIN_PROFILE_POINTS = 3
ANNUAL_BALANCE_MIN = -10

# Optional comparison line for one glacier
CALIBRATION_REFERENCE = {
    "RHONE": {
        "ela": 3050,
        "slope_low": 0.00853,
        "slope_up": 0.00367,
    }
}


# ============================================================
# Data loading and filtering
# ============================================================

def load_data(input_file_band: Path, input_file_mb: Path, lookup_file: Path):
    df_band = pd.read_csv(input_file_band)
    df_mb = pd.read_csv(input_file_mb)
    lookup = pd.read_csv(lookup_file)
    return df_band, df_mb, lookup

def prepare_mass_balance_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["country"].isin(COUNTRIES)].copy()

    df = df.dropna(subset=["glacier_id", "glacier_name", "year"]).copy()

    df["year"] = df["year"].astype(int)

    df["annual_balance"] = pd.to_numeric(df["annual_balance"], errors="coerce")
    df["ela"] = pd.to_numeric(df["ela"], errors="coerce")

    df = df.dropna(subset=["annual_balance", "ela"]).copy()

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    years_per_glacier = df.groupby("glacier_id")["year"].nunique()
    valid_ids = years_per_glacier[years_per_glacier > MIN_UNIQUE_YEARS].index
    df = df[df["glacier_id"].isin(valid_ids)].copy()

    return df

def aggregate_annual_balance(df_mb: pd.DataFrame) -> pd.DataFrame:
    agg = df_mb.groupby("glacier_id").agg(
        annual_mass_balance_mean=("annual_balance", "mean"),
        annual_mass_balance_std=("annual_balance", "std"),
        n_years_mb=("annual_balance", "count"),
        ela_mean=("ela", "mean"),
        ela_std=("ela", "std"),
        n_years_ela=("ela", "count"),
    ).reset_index()

    return agg

def prepare_mass_balance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean WGMS mass balance band data."""
    df = df[df["country"].isin(COUNTRIES)].copy()

    df = df.dropna(
        subset=[
            "glacier_id",
            "glacier_name",
            "year",
            "lower_elevation",
            "upper_elevation",
        ]
    ).copy()

    df["year"] = df["year"].astype(int)
    df["mean_elevation"] = (df["lower_elevation"] + df["upper_elevation"]) / 2

    df["annual_balance"] = pd.to_numeric(df["annual_balance"], errors="coerce")
    df = df.dropna(subset=["annual_balance"]).copy()

    df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()

    years_per_glacier = df.groupby("glacier_id")["year"].nunique()
    valid_ids = years_per_glacier[years_per_glacier > MIN_UNIQUE_YEARS].index
    df = df[df["glacier_id"].isin(valid_ids)].copy()

    return df


# ============================================================
# Profile and fitting
# ============================================================

def build_mean_profile(glacier_df: pd.DataFrame) -> pd.DataFrame:
    """Build mean annual balance profile over all years for one glacier."""
    profile = (
        glacier_df.groupby("mean_elevation", as_index=False)["annual_balance"]
        .mean()
        .sort_values("mean_elevation")
    )

    profile = profile[profile["annual_balance"] >= ANNUAL_BALANCE_MIN].copy()
    return profile


def estimate_ela(elev: np.ndarray, mb: np.ndarray, glacier_name: str, country: str) -> float:
    """Estimate ELA from zero crossing or fallback linear fit."""
    sign_change = np.where(np.diff(np.sign(mb)) != 0)[0]

    if len(sign_change) > 0:
        i = sign_change[0]
        x1, x2 = elev[i], elev[i + 1]
        y1, y2 = mb[i], mb[i + 1]
        ela = x1 - y1 * (x2 - x1) / (y2 - y1)
    else:
        print(f"No sign change found for {country} {glacier_name}, using linear fallback.")
        slope, intercept = np.polyfit(elev, mb, 1)
        ela = -intercept / slope

    return ela


def fit_gradient_through_ela(elev: np.ndarray, mb: np.ndarray, ela: float) -> tuple[float | None, float | None]:
    """Fit slope through origin in ELA-relative coordinates."""
    if len(elev) < 2:
        return None, None

    x = elev - ela
    y = mb

    denom = np.sum(x ** 2)
    if denom == 0:
        return None, None

    slope = np.sum(x * y) / denom
    intercept = -slope * ela
    return slope, intercept


def analyze_glacier(glacier_df: pd.DataFrame, glacier_id: str, glacier_name: str) -> dict | None:
    """Estimate ELA and gradients for one glacier."""
    country = glacier_df["country"].iloc[0]

    profile = build_mean_profile(glacier_df)
    if len(profile) < MIN_PROFILE_POINTS:
        print(f"Profile too small for {country} {glacier_name}")
        return None

    elev = profile["mean_elevation"].to_numpy()
    mb = profile["annual_balance"].to_numpy()

    ela = estimate_ela(elev, mb, glacier_name, country)

    lower_mask = elev <= ela
    upper_mask = elev >= ela

    slope_low, intercept_low = fit_gradient_through_ela(elev[lower_mask], mb[lower_mask], ela)

    if np.sum(upper_mask) > 3:
        slope_up, intercept_up = fit_gradient_through_ela(elev[upper_mask], mb[upper_mask], ela)
    else:
        slope_up, intercept_up = None, None

    return {
        "country": country,
        "glacier_id": glacier_id,
        "glacier_name": glacier_name,
        "ELA_m": ela,
        "lower_gradient": slope_low * 1000 if slope_low is not None else None,
        "upper_gradient": slope_up * 1000 if slope_up is not None else None,
        "profile": profile,
        "slope_low": slope_low,
        "intercept_low": intercept_low,
        "slope_up": slope_up,
        "intercept_up": intercept_up,
        "raw_data": glacier_df,
    }


# ============================================================
# Plotting
# ============================================================

def safe_filename(name: str) -> str:
    """Create filesystem-safe glacier name."""
    return re.sub(r"[^\w\-_. ]", "", name).replace(" ", "_")


def plot_glacier_fit(result: dict, output_dir: Path) -> None:
    """Plot yearly profiles, mean profile, fitted gradients, and optional reference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    glacier_name = result["glacier_name"]
    glacier_id = result["glacier_id"]
    ela = result["ELA_m"]

    profile = result["profile"]
    raw_data = result["raw_data"]

    elev = profile["mean_elevation"].to_numpy()
    mb = profile["annual_balance"].to_numpy()

    slope_low = result["slope_low"]
    intercept_low = result["intercept_low"]
    slope_up = result["slope_up"]
    intercept_up = result["intercept_up"]

    plt.figure(figsize=(8, 6))

    years_sorted = sorted(raw_data["year"].unique())
    norm = mpl.colors.Normalize(vmin=min(years_sorted), vmax=max(years_sorted))
    cmap = plt.cm.Reds

    for year in years_sorted:
        yearly = raw_data[raw_data["year"] == year].sort_values("mean_elevation")
        plt.plot(
            yearly["mean_elevation"],
            yearly["annual_balance"],
            color=cmap(norm(year)),
            label=str(year),
            alpha=0.5,
        )

    plt.plot(elev, mb, color="black", linewidth=2, label="Mean SMB")

    lower_mask = elev <= ela
    upper_mask = elev >= ela

    if slope_low is not None:
        x_low = np.linspace(np.min(elev[lower_mask]), ela, 100)
        y_low = slope_low * x_low + intercept_low
        plt.plot(
            x_low,
            y_low,
            "-",
            linewidth=4,
            color="purple",
            label=f"Ablation gradient: {slope_low * 1000:.2f}",
        )

    if slope_up is not None:
        x_up = np.linspace(ela, np.max(elev[upper_mask]), 100)
        y_up = slope_up * x_up + intercept_up
        plt.plot(
            x_up,
            y_up,
            "--",
            linewidth=4,
            color="purple",
            label=f"Accumulation gradient: {slope_up * 1000:.2f}",
        )

    plt.axvline(
        ela,
        linestyle=":",
        linewidth=2,
        color="purple",
        label=f"ELA ≈ {ela:.0f} m",
    )

    if glacier_name in CALIBRATION_REFERENCE:
        ref = CALIBRATION_REFERENCE[glacier_name]
        ela_ref = ref["ela"]
        slope_low_ref = ref["slope_low"]
        slope_up_ref = ref["slope_up"]

        intercept_low_ref = -slope_low_ref * ela_ref
        intercept_up_ref = -slope_up_ref * ela_ref

        x_low_ref = np.linspace(np.min(elev), ela_ref, 100)
        y_low_ref = slope_low_ref * x_low_ref + intercept_low_ref
        plt.plot(
            x_low_ref,
            y_low_ref,
            "-",
            linewidth=4,
            color="orange",
            label=f"Calib. ablation: {slope_low_ref * 1000:.2f}",
        )

        x_up_ref = np.linspace(ela_ref, np.max(elev), 100)
        y_up_ref = slope_up_ref * x_up_ref + intercept_up_ref
        plt.plot(
            x_up_ref,
            y_up_ref,
            "--",
            linewidth=4,
            color="orange",
            label=f"Calib. accumulation: {slope_up_ref * 1000:.2f}",
        )

        plt.axvline(
            ela_ref,
            linestyle=":",
            linewidth=2,
            color="orange",
            label=f"Calib. ELA ≈ {ela_ref:.0f} m",
        )

    plt.xlabel("Elevation (m)")
    plt.ylabel("Annual mass balance (m w.e. yr$^{-1}$)")
    plt.title(glacier_name)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / f"{safe_filename(glacier_name)}_{glacier_id}_fit.png"
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_ela_diff_from_true_mean(mb_agg: pd.DataFrame,
                                  sample_sizes: list[int] = list(range(1, 21)),
                                  seed: int = 10) -> None:
    """
    For each sample size, draw n_draws times and compute the mean absolute
    difference between the sampled mean ELA and each glacier's true 20-year mean.
    Aggregated across all glaciers and shown as a boxplot.
    """
    rng = np.random.default_rng(seed)

    # Group by glacier, collect ELA values into a list
    glacier_ela = (
        mb_agg.groupby("glacier_id")["ela"]
        .apply(lambda x: x.dropna().tolist())
        .reset_index()
    )
    glacier_ela["n_years_ela"] = glacier_ela["ela"].apply(len)

    # Filter to glaciers with exactly 20 years
    glaciers_20 = glacier_ela[glacier_ela["n_years_ela"] == 20].copy()
    print(f"Glaciers with 20 years of ELA: {len(glaciers_20)}")

    ela_matrix = np.stack(glaciers_20["ela"].values)  # (n_glaciers, 20)
    true_means = ela_matrix.mean(axis=1)              # (n_glaciers,)

    # Bootstrap: for each sample size, compute mean abs diff across glaciers
    n_draws = 10  # e.g. 5, but 50 looks much smoother

    results = {}
    for n in sample_sizes:
        diffs = []

        for _ in range(n_draws):
            idx = rng.choice(20, size=n, replace=False)
            sampled_means = ela_matrix[:, idx].mean(axis=1)
            mean_abs_diff = np.abs(sampled_means - true_means)

            diffs.append(mean_abs_diff)  # (n_glaciers,)

        # stack → shape (n_draws, n_glaciers)
        diffs = np.stack(diffs)

        # flatten → one distribution per n
        results[n] = diffs.flatten()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(len(sample_sizes)))
    data_to_plot = [results[n] for n in sample_sizes]

    ax.boxplot(data_to_plot, positions=positions, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6),zorder=10,
               medianprops=dict(color="black", linewidth=2))

    #ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="True 20-year mean")

    ax.set_xticks(positions)
    ax.set_xticklabels(sample_sizes)
    ax.set_xlabel("Number of sampled years")
    ax.set_ylabel("Mean absolute error to 20-year mean (m)")
    ax.set_title(f"ELA sampling error")
    ax.grid(axis="y", color="black", linestyle="-", zorder=-1, alpha=.2)
    ax.grid(axis="x", color="black", linestyle="-", zorder=-1, alpha=.2)
    ax.xaxis.set_tick_params(bottom=False)
    ax.yaxis.set_tick_params(left=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "ela_sampling_error.pdf")

# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    df_band_raw, df_mb_raw, lookup = load_data(INPUT_FILE_MB_Bands, INPUT_FILE_MB, LOOKUP_FILE)
    df = prepare_mass_balance_data(df_band_raw)
    df_mb = prepare_mass_balance_timeseries(df_mb_raw)

    plot_ela_diff_from_true_mean(df_mb)
    mb_agg = aggregate_annual_balance(df_mb)


    results = []

    for (glacier_id, glacier_name), glacier_df in df.groupby(["glacier_id", "glacier_name"]):
        result = analyze_glacier(glacier_df, glacier_id, glacier_name)
        if result is None:
            continue

        plot_glacier_fit(result, PLOT_DIR)

        results.append(
            {
                "country": result["country"],
                "glacier_id": result["glacier_id"],
                "glacier_name": result["glacier_name"],
                "ELA_m": result["ELA_m"], # comment out
                "lower_gradient": result["lower_gradient"],
                "upper_gradient": result["upper_gradient"],
            }
        )


    results_df = pd.DataFrame(results)
    results_df = results_df.merge(mb_agg, on="glacier_id", how="left")
    results_df = results_df.merge(
        lookup,
        left_on="glacier_name",
        right_on="wgms_name",
        how="left",
    )
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(results_df["ela_mean"], results_df["ELA_m"])
    plt.xlabel("Observed ELA (m)")
    plt.ylabel("Computed ELA (m)")

    plt.show()
    # use GLAMOS name
    results_df["glacier_name"] = results_df["glamos_name"]

    # drop unwanted columns
    results_df = results_df.drop(columns=["wgms_name", "rgi_name", "glamos_name"])

    results_df.to_csv(OUTPUT_CSV, index=False, float_format="%.3f")
    print(results_df.head())
    print(f"Saved {len(results_df)} glacier records to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()