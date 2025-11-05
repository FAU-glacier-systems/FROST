import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

labels = {
    "area_km2": "Area\n(km²)",
    "lmax_m": "Glacier\nlength (m)",
    "zmax_m": "Max.\nelevation (m)",
    "EastWest": "East–West\norientation",
    "SouthNorth": "South–North\norientation",
    "zmean_m": "Mean\nelevation \n(m)",
    "cenlon": "Central\nlongitude \n(°E)",
    "cenlat": "Central\nlatitude\n (°N)",
    "slope_deg": "Mean\nslope\n (°)",

    "ela": "Equilibrium-line\naltitude (m)",
    "gradabl": "Ablation\ngradient\n(m yr⁻¹ km⁻¹)",
    "gradacc": "Accumulation\ngradient\n(m yr⁻¹ km⁻¹)",
    "ela_std": "ELA\nstd (m)",
    "gradabl_std": "Ablation\ngradient\nstd (m yr⁻¹ km⁻¹)",
    "gradacc_std": "Accumulation\ngradient\nstd (m yr⁻¹ km⁻¹)"
}

def plot_colored_correlation_points(data, factors, targets):
    num_factors = len(factors)
    num_targets = len(targets)

    fig, axes = plt.subplots(num_factors, num_targets,
                             figsize=(1.5 * num_targets, 1* num_factors),
                             squeeze=False)

    for i, factor in enumerate(factors):        # colums
        for j, target in enumerate(targets):    # rows
            ax = axes[i, j]
            x = data[factor]
            y = data[target]

            # Compute correlation
            corr, _ = spearmanr(x, y, nan_policy='omit')

            # Scaled area for better visibility
            s = data["area_km2"]

            ax.scatter(y, x, s=s, alpha=0.7, edgecolor='k')

            # Annotate correlation
            ax.annotate(f"r = {corr:.2f}",
                        xy=(0.05, 0.9), xycoords="axes fraction",
                        fontsize=9, ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if factor == "area_km2":
                ax.set_yscale("log")



            # Only first column gets y labels
            if j == 0:
                ax.set_ylabel(labels[factor])
            else:
                ax.set_yticklabels([])

            # Only bottom row gets x labels
            if i == num_factors - 1:
                ax.set_xlabel(labels[target])
            else:
                ax.set_xticklabels([])

            ax.grid(True, linestyle='--', alpha=0.1)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)  # smaller values = less space

    plt.savefig("../central_europe_submit/plots/correlation.pdf")
    plt.close(fig)


# Load and clean data
csv_path = "../central_europe_submit/tables/aggregated_results.csv"
data = pd.read_csv(csv_path)

factors = ["area_km2", "lmax_m", "zmax_m", "EastWest", "SouthNorth",
           "zmean_m", "cenlon", "cenlat", "slope_deg"]
targets = ["ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std"]

data[factors + targets] = data[factors + targets].apply(pd.to_numeric, errors="coerce")
filtered_data = data.dropna(subset=factors + targets)

plot_colored_correlation_points(filtered_data, factors, targets)
