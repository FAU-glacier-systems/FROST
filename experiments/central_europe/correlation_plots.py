import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import spearmanr


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
                             figsize=(2 * num_factors, 2 * num_targets +5),
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
            sc = ax.scatter(x, y, c=[corr_coef] * len(x), cmap=cmap, norm=norm,
                            alpha=0.8, edgecolor='k', s=size)

            # Annotate correlation coefficient
            ax.annotate(f"r = {corr_coef:.2f}",
                        xy=(0.05, 0.90), xycoords="axes fraction",
                        fontsize=10, ha="left", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            if factor == "area_km2":
                ax.set_xscale("log")

            # Axis labels: Only set if on the outer rows/columns
            if i == num_targets - 1:  # Bottom row
                ax.set_xlabel(factor, fontsize=12, labelpad=10)
            else:
                ax.set_xticklabels([])  # Remove inner plots' x-axis labels
            if j == 0:  # First column
                ax.set_ylabel(target)
            else:
                ax.set_yticklabels([])  # Remove inner plots' y-axis labels

            # Titles for the first row based on the x-label (factor)

            ax.set_title(f"{factor}\nvs {target}", fontsize=12, pad=15)

            # Clean up grid and style
            ax.grid(visible=True, linestyle='--', alpha=0.5)


    # Add a single colorbar in the figure, not overlaying plots
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes,
        orientation='horizontal', location='top',
    )
    cbar.ax.set_position([0.1, 0.92, 0.8, 0.02])  # adjust values as needed

    cbar.set_label("Correlation Coefficient (r)")
    cbar.ax.xaxis.set_label_position('top')  # Move label above the colorbar
    cbar.ax.xaxis.set_ticks_position('bottom')  # Keep ticks below
    fig.subplots_adjust(
        wspace=0.5,  # width (horizontal) space between subplots
        hspace=0.5  # height (vertical) space between subplots
    )
    # Tight layout for better spacing and adjustments
 # Adjust to include colorbar and titles
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig("plots/correlation.png", dpi=200)


# Load data
csv_path = "aggregated_results.csv"  # Update this with your CSV file path
data = pd.read_csv(csv_path)

# Define factors (independent variables) and targets (dependent variables)
factors = ["area_km2", "lmax_m", "zmax_m", "EastWest", "SouthNorth",
            "zmean_m", "cenlon", "cenlat", "slope_deg"]
targets = ["ela", "gradabl", "gradacc", "ela_std", "gradabl_std", "gradacc_std"]

# Convert required columns to numeric, ignoring errors
columns_to_include = factors + targets
data[columns_to_include] = data[columns_to_include].apply(pd.to_numeric,
                                                          errors="coerce")

# Filter data to avoid NaN issues
filtered_data = data.dropna(subset=factors + targets)
# Duplicate target columns with a new name
for col in targets:
    new_col = f"{col}_"
    filtered_data[new_col] = filtered_data[col]
    factors.append(new_col)  # Add to factors list


# Generate correlation plots with adjusted layout
plot_colored_correlation_points(filtered_data, factors, targets)
