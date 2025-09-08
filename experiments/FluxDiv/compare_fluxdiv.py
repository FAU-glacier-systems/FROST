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
predicted_results = "../central_europe/aggregated_results.csv"
glamos_name_rgi_path = "../../data/raw/glamos/GLAMOS_RGI.csv"
fluxdiv_path = "smb_params_results_Anna.csv"

predicted_df = pd.read_csv(predicted_results)
glamos_name_rgi_df = pd.read_csv(glamos_name_rgi_path)
fluxdiv_df = pd.read_csv(fluxdiv_path, header=1)

merge_fluxdiv_rgi = pd.merge(fluxdiv_df, glamos_name_rgi_df, on="glamos_name",
                             how="inner")
merged_df = pd.merge(predicted_df, merge_fluxdiv_rgi, on="glamos_name")

merged_df = merged_df.sort_values(by="area_km2", ascending=True).reset_index(
    drop=True)

predicted_ela = merged_df['ela_x'].to_numpy()
predicted_grad_abl = merged_df['gradabl'].to_numpy()
predicted_grad_acc = merged_df['gradacc'].to_numpy()

predicted_ela_std = merged_df['ela_std'].to_numpy()
predicted_grad_abl_std = merged_df['gradabl_std'].to_numpy()
predicted_grad_acc_std = merged_df['gradacc_std'].to_numpy()

reference_ela = merged_df['ela_y'].to_numpy()
reference_grad_abl = merged_df['abl_grad'].to_numpy()*1000
reference_grad_acc = merged_df['acc_grad'].to_numpy()*1000

reference_ela_std = merged_df['iqr_ela'].to_numpy()
reference_grad_abl_std = merged_df['iqr_abl_grad'].to_numpy()*1000
reference_grad_acc_std = merged_df['iqr_acc_grad'].to_numpy()*1000

# Combine glacier name with shortened RGI ID for labels
glacier_names = [f"{name}" for name in merged_df['glamos_name']]
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
    xlabel="FluxDiv ELA (m)",
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
    xlabel="FluxDiv Ablation Gradient (m/yr/km)",
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
    xlabel="FluxDiv Accumulation Gradient (m/yr/km)",
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
plt.savefig("../central_europe/Flux_Dif_comp.pdf", bbox_inches="tight")
plt.clf()
