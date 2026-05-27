from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("../../data/raw/DOI-WGMS-FoG-2025-02b/data/mass_balance_band.csv")
OUTPUT_DIR = Path("Plots")

COUNTRIES = ["AT", "IT", "CH", "FR"]
YEAR_MIN = 2000
YEAR_MAX = 2019
MIN_UNIQUE_YEARS = 5


# ============================================================
# Data preparation
# ============================================================

def load_data(file_path: Path) -> pd.DataFrame:
    """Load WGMS mass balance band data."""
    return pd.read_csv(file_path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean the dataset."""
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
# Plotting
# ============================================================

def make_safe_filename(name: str) -> str:
    """Replace problematic filename characters."""
    return name.replace("/", "_")


def plot_glacier_timeseries(glacier_df: pd.DataFrame, glacier_id: str, glacier_name: str, output_dir: Path) -> None:
    """Plot annual balance profiles by year for one glacier."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    years_sorted = sorted(glacier_df["year"].unique())
    norm = mpl.colors.Normalize(vmin=min(years_sorted), vmax=max(years_sorted))
    cmap = plt.cm.Reds

    for year in years_sorted:
        yearly = glacier_df[glacier_df["year"] == year].sort_values("mean_elevation")
        plt.plot(
            yearly["mean_elevation"],
            yearly["annual_balance"],
            marker="o",
            color=cmap(norm(year)),
            label=str(year),
        )

    plt.xlabel("Mean elevation")
    plt.ylabel("Annual balance")
    plt.title(f"{glacier_name} ({glacier_id})")
    plt.legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    filename = output_dir / f"{make_safe_filename(glacier_name)}.png"
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Main workflow
# ============================================================

def main() -> None:
    df_raw = load_data(INPUT_FILE)
    df = prepare_data(df_raw)

    for (glacier_id, glacier_name), glacier_df in df.groupby(["glacier_id", "glacier_name"]):
        try:
            plot_glacier_timeseries(glacier_df, glacier_id, glacier_name, OUTPUT_DIR)
        except Exception as exc:
            print(f"Could not save plot for {glacier_name}: {exc}")


if __name__ == "__main__":
    main()