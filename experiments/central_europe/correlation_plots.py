import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import spearmanr

labels = {
    "area_km2": "Area\n(km²)",
    "lmax_m": "Glacier\nlength (m)",
    "zmax_m": "Max.\nelevation (m)",
    "EastWest": "East–West\norientation",
    "SouthNorth": "South–North\norientation",
    "zmean_m": "Mean\nelevation (m)",
    "cenlon": "Central\nlongitude (°E)",
    "cenlat": "Central\nlatitude (°N)",
    "slope_deg": "Mean\nslope (°)",

    "ela": "Equilibrium-line\naltitude (m)",
    "gradabl": "Ablation\ngradient\n(m a⁻¹ km⁻¹)",
    "gradacc": "Accumulation\ngradient\n(m a⁻¹ km⁻¹)",
    "ela_std": "ELA\nstd (m)",
    "gradabl_std": "Ablation\ngradient\nstd (m a⁻¹ km⁻¹)",
    "gradacc_std": "Accumulation\ngradient\nstd (m a⁻¹ km⁻¹)"
}

def plot_colored_correlation_points(data, factors, targets):
    """
    Generate subplots correlating multiple factors with specific target variables,
    color points in the scatter plots based on the correlation coefficient, and
    adjust layout to show labels only on outer axes.

    :param data: DataFrame containing the data.
    :param factors: List of column names to correlate with targets (independent variables).
    :param targets: List of target column names (dependent variables).
    """
    num_factors = len(factors)
    num_targets = len(targets)

    fig, axes = plt.subplots(num_targets, num_factors,
                             figsize=(1.5*num_factors, 1.3 * num_targets),
                             squeeze=False)

    cmap = "RdBu_r"  # Colormap for points
    norm = plt.Normalize(-1, 1)  # Normalize correlation values between -1 and 1
    size = data['area_km2']
    for i, target in enumerate(targets):
        for j, factor in enumerate(factors):
            ax = axes[i][j]

            # Extract columns
            x = data[factor]
            y = data[target]

            # Calculate the Pearson correlation coefficient
            corr_coef, _ = spearmanr(x, y)

            # Scatter plot, with uniform point color based on `corr_coef`
            ax.scatter(x, y, #c=[corr_coef] * len(x), cmap=cmap,
                            norm=norm,
                            alpha=0.8, edgecolor='k', s=size)

            # Annotate correlation coefficient
            ax.annotate(f"r = {corr_coef:.2f}",
                        xy=(0.05, 0.90), xycoords="axes fraction",
                        fontsize=10, ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            if factor == "area_km2":
                ax.set_xscale("log")

            # Axis labels: Only set if on the outer rows/columns
            if i == num_targets - 1:  # Bottom row
                ax.set_xlabel(labels[factor])
            else:
                ax.set_xticklabels([])  # Remove inner plots' x-axis labels
            if j == 0:  # First column
                ax.set_ylabel(labels[target])
            else:
                ax.set_yticklabels([])  # Remove inner plots' y-axis labels

            # Titles for the first row based on the x-label (factor)

            # Clean up grid and style
            ax.grid(visible=True, linestyle='--', alpha=0.5)

    # # Add a single colorbar in the figure, not overlaying plots
    # # cbar = fig.colorbar(
    # #     plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    # #     ax=axes,
    # #     orientation='horizontal', location='top',
    # # )
    # cbar.ax.set_position([0.1, 0.92, 0.8, 0.02])  # adjust values as needed
    #
    # cbar.set_label("Correlation Coefficient (r)")
    # cbar.ax.xaxis.set_label_position('top')  # Move label above the colorbar
    # cbar.ax.xaxis.set_ticks_position('bottom')  # Keep ticks below
    # fig.subplots_adjust(
    #     wspace=0.5,  # width (horizontal) space between subplots
    #     hspace=0.5  # height (vertical) space between subplots
    # )
    # Tight layout for better spacing and adjustments
    # Adjust to include colorbar and titles
    #plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.tight_layout()
    plt.savefig("plots/correlation.pdf")


# Load data
csv_path = "../central_europe_1000/aggregated_results.csv"  # Update this with your CSV file path
data = pd.read_csv(csv_path)

# Define factors (independent variables) and targets (dependent variables)
factors = ["area_km2", "lmax_m", "zmax_m", "EastWest", "SouthNorth",
           "zmean_m", "cenlon", "cenlat", "slope_deg"]
targets = ["ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std"]



# Convert required columns to numeric, ignoring errors
columns_to_include = factors  # + targets
data[columns_to_include] = data[columns_to_include].apply(pd.to_numeric, errors="coerce")

# Filter data to avoid NaN issues
filtered_data = data.dropna(subset=factors)
# Duplicate target columns with a new name
for col in targets:
    new_col = f"{col}_"
    filtered_data[new_col] = filtered_data[col]

# Generate correlation plots with adjusted layout
plot_colored_correlation_points(filtered_data, factors, targets)
