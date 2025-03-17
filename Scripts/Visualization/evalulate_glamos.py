#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
root_folder = "../../Experiments"
experiment_folders = os.listdir(root_folder)
rgi_name_table = "../../Data/GLAMOS/GLAMOS_RGI.csv"
rgi_df = pd.read_csv(rgi_name_table)

# Data Storage
glamos_results = {"final_mean": [], "final_std": [], "reference_smb": []}
glacier_names = []

# Load Data
for rgi_id in experiment_folders:
    if not rgi_id.startswith("RGI2000-v7.0-G-11"):
        continue

    row = rgi_df[rgi_df["RGI_ID"] == rgi_id]
    glacier_names.append(row["GLAMOS Name"].values[0])

    results_file = os.path.join(root_folder, rgi_id, "Experiment_50_50_1.0_1", "result_seed_1_1.0_50.json")
    reference_file = os.path.join(root_folder, rgi_id, "params_calibration.json")

    with open(results_file) as f:
        results = json.load(f)
    with open(reference_file) as f:
        params = json.load(f)

    glamos_results["final_mean"].append(results["final_mean"])
    glamos_results["final_std"].append(results["final_std"])
    glamos_results["reference_smb"].append(params["reference_smb"])

# Convert to NumPy Arrays
predicted_smb = np.array(glamos_results["final_mean"])
predicted_smb_std = np.array(glamos_results["final_std"])

predicted_ela = predicted_smb[:, 0]
predicted_grad_abl = predicted_smb[:, 1]
predicted_grad_acc = predicted_smb[:, 2] * 0.55

predicted_ela_std = predicted_smb_std[:, 0]
predicted_grad_abl_std = predicted_smb_std[:, 1]
predicted_grad_acc_std = predicted_smb_std[:, 2]

# Extract Reference Data
reference_smb = glamos_results["reference_smb"]
reference_ela, reference_grad_abl, reference_grad_acc = [], [], []
for smb in reference_smb:
    reference_ela.append(smb["ela"])
    reference_grad_abl.append(smb["gradabl"])
    reference_grad_acc.append(smb["gradacc"])

# Color Mapping
colors = plt.get_cmap("tab20").colors

# **Helper Function for Scatter Plots**
def scatter_plot(ax, x, y, x_std, y_std, xlabel, ylabel, title, xlim, ylim, diag_range):
    """Creates a scatter plot with a regression line, 1:1 reference line, and shaded error regions."""
    ax.plot(diag_range, diag_range, "--", color="black", alpha=0.3, zorder=-4, label="1:1 Correlation")

    # Scatter points
    scatter_handles = []
    for i, label in enumerate(glacier_names):
        scatter = ax.scatter(x[i], y[i], label=label, color=colors[i])
        scatter_handles.append(scatter)
        ax.fill_between([x[i] - x_std, x[i] + x_std],
                        y[i] - y_std[i], y[i] + y_std[i],
                        color=colors[i], alpha=0.5, zorder=-2)

    # Compute and plot regression line
    slope, intercept = np.polyfit(x, y, 1)
    regression_x = np.linspace(min(x), max(x), 100)
    regression_y = slope * regression_x + intercept
    regression_line, = ax.plot(regression_x, regression_y, color="Black",
                               linestyle="-", linewidth=2, zorder=-3)



    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

    # Add legend for regression line
    legend_reg = ax.legend([regression_line], [f"Regression Line\n(slope={slope:.2f})"], loc="lower right", fontsize=10, frameon=True)
    ax.add_artist(legend_reg)

    return scatter_handles

# **Create Figure with 3 Subplots**
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# **1️⃣ ELA Scatter Plot**
scatter_handles_ela = scatter_plot(
    axes[0], reference_ela, predicted_ela, 1, predicted_ela_std,
    "Reference ELA (m)", "Predicted ELA (m)", "ELA Comparison",
    (2780, 3620), (2780, 3620), np.arange(2800, 3601)
)

# **2️⃣ Ablation Gradient Scatter Plot**
scatter_handles_abl = scatter_plot(
    axes[1], reference_grad_abl, predicted_grad_abl, 0.01, predicted_grad_abl_std,
    "Reference Ablation Gradient (m/yr/km)", "Predicted Ablation Gradient (m/yr/km)", "Ablation Gradient Comparison",
    (1.7, 12.3), (1.7, 12.3), np.arange(2, 13)
)

# **3️⃣ Accumulation Gradient Scatter Plot**
scatter_handles_acc = scatter_plot(
    axes[2], reference_grad_acc, predicted_grad_acc, 0.001, predicted_grad_acc_std,
    "Reference Accumulation Gradient (m/yr/km)", "Predicted Accumulation Gradient (m/yr/km)", "Accumulation Gradient Comparison",
    (-2.2, 6.2), (-2.2, 6.2), np.arange(-2, 7)
)

# **Legend for Glacier Names (Outside Figure)**
fig.legend(scatter_handles_ela, glacier_names, loc="upper left", bbox_to_anchor=(1, 0.93), fontsize=10)

fig.tight_layout()

# Save the figure
plt.savefig("../../Plots/GLAMOS_regional_run.png", dpi=300, bbox_inches="tight")

