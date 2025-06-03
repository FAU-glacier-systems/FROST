#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Paths
glamos_results = "../../Data/GLAMOS/GLAMOS_analysis_results.csv"
predicted_results = "../../Scripts/CentralEurope/aggregated_results.csv"
sla_path = "../../Data/CentralEurope/Alps_EOS_SLA_2000-2019_mean.csv"

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
glacier_names = [
    f"{name} ({rgi.split('-')[-1]})" for name, rgi in zip(merged_df_glamos[
                                                              'Glacier_Name'],
                                                          merged_df_glamos['rgi_id'])
]

cmap = plt.get_cmap("tab20").colors
colors = tuple(cmap[i] for i in range(0, 20, 2)) + tuple(
    cmap[i] for i in range(1, 20, 2))


def scatter_plot(ax, x, y, xlabel, ylabel, title, ticks, glacier_names=None,
                 y_std=None, x_std=None, color=None):
    margin = (ticks[-1] - ticks[0]) * 0.05
    x_min = ticks[0] - margin
    x_max = ticks[-1] + margin


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
        scatter = ax.scatter(x, y, c=np.log(color))
        plt.colorbar(scatter, ax=ax, label='Log(Area) km²', shrink=0.8,)

    else:
        for i, label in enumerate(glacier_names):
            scatter = ax.scatter(x[i], y[i], label=label,
                                 color=colors[i % len(colors)])
            scatter_handles.append(scatter)
            if x_std is not None and y_std is not None:
                ellipse = Ellipse(
                    (x[i], y[i]),
                    width=2 * x_std[i],
                    height=2 * y_std[i],
                    edgecolor='none', facecolor=colors[i % len(colors)],
                    alpha=0.3, zorder=-2
                )
                ax.add_patch(ellipse)



    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect('equal', adjustable='box')

    return scatter_handles


fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes = axes.flatten()

scatter_handles_ela = scatter_plot(
    ax=axes[0],
    x=reference_ela,
    y=predicted_ela,
    x_std=reference_ela_std,
    y_std=predicted_ela_std,
    xlabel="GLAMOS ELA (m)",
    ylabel="Predicted ELA (m)",
    title="ELA Comparison",
    glacier_names=glacier_names,
    ticks=np.arange(2500, 3501, 250),
)

scatter_handles_abl = scatter_plot(
    ax=axes[1],
    x=reference_grad_abl,
    y=predicted_grad_abl,
    x_std=reference_grad_abl_std,
    y_std=predicted_grad_abl_std,
    xlabel="GLAMOS Ablation Gradient (m/yr/km)",
    ylabel="Predicted Ablation Gradient (m/yr/km)",
    title="Ablation Gradient Comparison",
    glacier_names=glacier_names,
    ticks=np.arange(0, 26, 5),

)

scatter_handles_acc = scatter_plot(
    ax=axes[2],
    x=reference_grad_acc,
    y=predicted_grad_acc,
    x_std=reference_grad_acc_std,
    y_std=predicted_grad_acc_std,
    xlabel="GLAMOS Accumulation Gradient (m/yr/km)",
    ylabel="Predicted Accumulation Gradient (m/yr/km)",
    title="Accumulation Gradient Comparison",
    glacier_names=glacier_names,
    ticks=np.arange(0, 16, 3)
)

scatter_handles_sla = scatter_plot(ax=axes[3],
                                   x=merged_df_sla['sla'].to_numpy(),
                                   y=merged_df_sla['ela'].to_numpy(),
                                   xlabel="End of summer snow line altitude (m)",
                                   ylabel="Predicted ELA (m)",
                                   title="SLA Comparison",
                                   ticks=np.arange(2200, 4000, 500),
                                   color=merged_df_sla['area_km2'].to_numpy())

scatter_handles_glamos_sla = scatter_plot(ax=axes[4],
                                          y=merged_df_glamos_sla['sla'].to_numpy(),
                                          x=merged_df_glamos_sla[
                                              'Mean_ELA'].to_numpy(),
                                          ylabel="End of summer snow line altitude (m)",
                                          xlabel="GLAMOS ELA (m)",
                                          title="SLA GLAMOS Comparison",
                                          glacier_names=glacier_names,
                                          ticks=np.arange(2200, 4000, 500))

scatter_handlessla = scatter_plot(ax=axes[5],
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
           bbox_to_anchor=(1, 0.93), fontsize=10)

fig.tight_layout()
plt.savefig("../../Plots/GLAMOS_regional_run.png", dpi=300, bbox_inches="tight")
