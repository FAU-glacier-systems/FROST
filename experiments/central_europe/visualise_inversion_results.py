import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def scatter_plot(ax, x, y, xlabel, ylabel, title, ticks, glacier_names=None,
                 y_std=None, x_std=None, color=None):
    x_min = ticks[0]
    x_max = ticks[-1]

    ax.plot([x_min, x_max], [x_min, x_max], "--", color="black", alpha=0.3,
            zorder=-4,
            label="1:1 Correlation")

    # y = predictions, x = observations
    mae = np.mean(np.abs(y - x))
    bias = np.mean(y - x)
    y_corr = y - bias
    bc_didf = y_corr - x
    bcmae = np.mean(np.abs(bc_didf))  # Bias-corrected MAE
    correlation = np.corrcoef(x, y)[0, 1]

    ax.text(0.05, 0.95,
            f'MAE: {mae:.0f} Bias: {bias:.0f}\nBias-corrected-\nMAE:'
            f' {bcmae:.0f}\nPearson r: {correlation:.2f}',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')

    scatter_handles = []
    if glacier_names is None:
        if color is None:
            relative_error = abs(
                y - x) / x * 100  # Calculate relative error as percentage
            scatter = ax.scatter(x, y, c=relative_error)
        else:
            scatter = ax.scatter(x, y, c=np.log(color))
        plt.colorbar(scatter, ax=ax, label='Relative Error (%)', shrink=0.8)

    else:
        for i, label in enumerate(glacier_names):
            scatter = ax.scatter(x[i], y[i], label=label,
                                 color=color[i % len(color)])
            scatter_handles.append(scatter)
            if x_std is not None and y_std is not None:
                ellipse = Ellipse(
                    (x[i], y[i]),
                    width=2 * x_std[i],
                    height=2 * y_std[i],
                    edgecolor='none', facecolor=color[i % len(color)],
                    alpha=0.3, zorder=-2
                )
                ax.add_patch(ellipse)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_title(f"{title}")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    # ax.set_aspect('equal', adjustable='box')

    return scatter_handles


# Read both CSV files
velocity_df = pd.read_csv('../glamos_run/inversion_results.csv')
#aggregated_df = pd.read_csv('aggregated_results.csv')

# Merge the dataframes on RGI_ID
#merged_df = pd.merge(velocity_df, aggregated_df, on='rgi_id', how='inner')

# Compute the velocity error for each glacier
velocity_df['velocity_error'] = abs(
    velocity_df['Mean_velsurf_mag'] - velocity_df['Mean_velsurfobs_mag'])

# Find the glacier with the highest velocity error
highest_error_row = velocity_df.loc[velocity_df['velocity_error'].idxmax()]
highest_error_rgi_id = highest_error_row['rgi_id']
highest_error_value = highest_error_row['velocity_error']

print(f"The glacier with the highest velocity error is: {highest_error_rgi_id}")
print(f"The highest velocity error value is: {highest_error_value}")

# Save the merged results
print("Results merged and saved to merged_results.csv")

# Define velocity statistics to compare
vel_stats = [
    ('Mean_velsurf_mag', 'Mean_velsurfobs_mag', 'Mean'),
    ('Q1_velsurf_mag', 'Q1_velsurfobs_mag', 'Q1'),
    ('Max_velsurf_mag', 'Max_velsurfobs_mag', 'Max'),
    ('Std_velsurf_mag', 'Std_velsurfobs_mag', 'Std'),
    ('Q3_velsurf_mag', 'Q3_velsurfobs_mag', 'Q3'),
    ('Median_velsurf_mag', 'Median_velsurfobs_mag', 'Median'),

]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for idx, (mod_col, obs_col, stat_name) in enumerate(vel_stats):
    velsurf = velocity_df[mod_col].to_numpy()
    velsurf_obs = velocity_df[obs_col].to_numpy()

    scatter_handles = scatter_plot(ax=axes[idx],
                                   x=velsurf_obs,
                                   y=velsurf,
                                   ylabel=f"Modelled {stat_name} Surface Velocity (m/yr)",
                                   xlabel=f"Observed {stat_name} Surface Velocity (m/yr)",
                                   title=f"{stat_name} Velocity",
                                   ticks=np.logspace(0, 3, 4))

import string

labels_subplot = [f"{letter})" for letter in
                  string.ascii_lowercase[:len(axes)]]
for ax, label in zip(axes, labels_subplot):
    # Add label to lower-left corner (relative coordinates)
    ax.text(0, 1.02, label, transform=ax.transAxes,
            fontsize=12, va='bottom', ha='left', fontweight='bold')
fig.tight_layout()
plt.savefig("../glamos_run/plots/inversion_results.pdf",
            bbox_inches="tight")
