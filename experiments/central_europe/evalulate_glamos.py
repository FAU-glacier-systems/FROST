#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from frost.visualization.utils import scatter_plot

# Paths
glamos_results = "../../data/raw/glamos/GLAMOS_analysis_results.csv"
predicted_results = "../../experiments/central_europe/aggregated_results.csv"
#predicted_results = "../../Scripts/CentralEurope/velocity_results_merged.csv"
sla_path = ("../../data/raw/central_europe/Alps_EOS_SLA_2000-2019_mean.csv")

glamos_df = pd.read_csv(glamos_results)
predicted_df = pd.read_csv(predicted_results)
sla_df = pd.read_csv(sla_path)

merged_df_glamos = pd.merge(glamos_df, predicted_df, on="rgi_id", how="left",
                            suffixes=('', '_drop'))
merged_df_glamos = merged_df_glamos.loc[:,
                   ~merged_df_glamos.columns.str.endswith('_drop')]
merged_df_glamos = merged_df_glamos.dropna(subset=["ela"])
merged_df_glamos = merged_df_glamos.sort_values(by="area_km2",
                                                ascending=True).reset_index(
    drop=True)

merged_df_sla = pd.merge(predicted_df, sla_df, on="rgi_id", how="inner",
                         suffixes=('', '_drop'))
merged_df_sla = merged_df_sla.sort_values(by="area_km2",
                                          ascending=True).reset_index(
    drop=True)

merged_df_glamos_sla = pd.merge(merged_df_glamos, sla_df, on="rgi_id", how="inner",
                                suffixes=('', '_drop'))

predicted_ela = merged_df_glamos['ela'].to_numpy()
predicted_grad_abl = merged_df_glamos['gradabl'].to_numpy()
predicted_grad_acc = merged_df_glamos['gradacc'].to_numpy()

predicted_ela_std = merged_df_glamos['ela_std'].to_numpy()
predicted_grad_abl_std = merged_df_glamos['gradabl_std'].to_numpy()
predicted_grad_acc_std = merged_df_glamos['gradacc_std'].to_numpy()

reference_ela = merged_df_glamos['Mean_ELA'].to_numpy()
reference_grad_abl = merged_df_glamos['Mean_Ablation_Gradient'].to_numpy()
reference_grad_acc = merged_df_glamos['Mean_Accumulation_Gradient'].to_numpy()

reference_ela_std = merged_df_glamos['Annual_Variability_ELA'].to_numpy()
reference_grad_abl_std = merged_df_glamos[
    'Annual_Variability_Ablation_Gradient'].to_numpy()
reference_grad_acc_std = merged_df_glamos[
    'Annual_Variability_Accumulation_Gradient'].to_numpy()

# Combine glacier name with shortened RGI ID for labels
glacier_names = [f"{name}" for name in merged_df_glamos['Glacier_Name']]
# \n({rgi.split('-')[-1]})" for name, rgi in zip(
# merged_df_glamos[
#                                               'Glacier_Name'],
#                                           merged_df_glamos[
#                                           'rgi_id'])]


fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes = axes.flatten()

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

axes[1].axis('off')
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
    ticks=np.arange(0, 16, 3)
)

fig.legend(scatter_handles_ela, glacier_names, loc="upper left",
           bbox_to_anchor=(0.6, 0.95), fontsize=10)

import string

axes_with_label = [axes[0], axes[2], axes[3]]
labels_subplot = [f"{letter})" for letter in
                  string.ascii_lowercase[:len(axes_with_label)]]
for ax, label in zip(axes_with_label, labels_subplot):
    # Add label to lower-left corner (relative coordinates)
    ax.text(0, 1.02, label, transform=ax.transAxes,
            fontsize=12, va='bottom', ha='left', fontweight='bold')
fig.tight_layout()
plt.savefig("plots/GLAMOS_regional_run.pdf", bbox_inches="tight")
plt.clf()

fig, axes = plt.subplots(1, 1, figsize=(5, 5))

ae_ela = np.abs(merged_df_sla['sla'].to_numpy() - merged_df_sla['ela'].to_numpy())
scatter_handles_sla = scatter_plot(ax=axes,
                                   x=merged_df_sla['sla'].to_numpy(),
                                   y=merged_df_sla['ela'].to_numpy(),
                                   xlabel="End of summer snow line altitude (m)",
                                   ylabel="Predicted ELA (m)",
                                   title="End of summer snow line altitude\nvs. "
                                         "Predicted ELA",
                                   ticks=np.arange(2200, 4000, 500),
                                   color=ae_ela)

plt.savefig("plots/SLA_comparison.pdf", bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(7, 4))
axes = axes.flatten()
scatter_handles_glamos_sla = scatter_plot(ax=axes[0],
                                          y=merged_df_glamos_sla['sla'].to_numpy(),
                                          x=merged_df_glamos_sla[
                                              'Mean_ELA'].to_numpy(),
                                          ylabel="End of summer snow line altitude (m)",
                                          xlabel="GLAMOS ELA (m)",
                                          title="",
                                          glacier_names=glacier_names,
                                          ticks=np.arange(2200, 4000, 500))

scatter_handlessla = scatter_plot(ax=axes[1],
                                  x=merged_df_glamos_sla['sla'].to_numpy(),
                                  y=merged_df_glamos_sla[
                                      'ela'].to_numpy(),
                                  xlabel="End of summer snow line altitude (m)",
                                  ylabel="Predicted ELA (m)",
                                  title="SLA ELA Comparison on Selected Glaciers",
                                  glacier_names=glacier_names,
                                  ticks=np.arange(2200, 4000, 500))

# Calculate absolute differences between SLA and ELA
merged_df_glamos_sla['difference'] = abs(
    merged_df_sla['sla'] - merged_df_sla['ela'])

# Get top 10 differences
top_10_diff = merged_df_glamos_sla.nlargest(10, 'difference')[
    ['rgi_id', 'Glacier_Name', 'sla', 'ela', 'difference']]

print("\nTop 10 glaciers with highest SLA-ELA difference:")
print("=" * 60)
for _, row in top_10_diff.iterrows():
    print(f"RGI ID: {row['rgi_id']}")
    print(f"Glacier Name: {row['Glacier_Name']}")
    print(f"SLA: {row['sla']:.0f} m")
    print(f"ELA: {row['ela']:.0f} m")
    print(f"Difference: {row['difference']:.0f} m")
    print("-" * 30)

fig.legend(scatter_handles_ela, glacier_names, loc="upper left",
           bbox_to_anchor=(1, 0.9), fontsize=10)

fig.tight_layout()
plt.savefig("plots/SLA_comparison_selection.pdf", bbox_inches="tight")
