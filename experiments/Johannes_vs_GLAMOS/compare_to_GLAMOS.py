from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
glamos_file = Path("../WGMS/tables/combined_ela_gradients.csv")
calibrated_file = Path("MB_compare_output_v06.csv")

glamos_df = pd.read_csv(glamos_file)
calibrated_df = pd.read_csv(calibrated_file)

merged = glamos_df.merge(
    calibrated_df,
    left_on="rgi_id",
    right_on="RGI_ID",
    how="left"
)
merged = merged.dropna(subset=["spec. SMBs_mod [m i.e. / yr]", "annual_mass_balance"]).copy()
merged = merged.sort_values("country", ascending=True)

# --- Prepare data ---
x = merged['spec. SMBs_mod [m i.e. / yr]']
y = merged["annual_mass_balance"]
glacier_names = merged["glacier_name"]

# correlation
r = x.corr(y)
country_to_marker = {
    "CH": "o",
    "AT": "^",
    "IT": "s",
    "FR": "D",
    "DE": "v",
}
# --- Plot ---
fig, ax = plt.subplots(figsize=(4, 4))
glacier_info = (
    merged[["rgi_id", "country", "glacier_name"]]
    .drop_duplicates(subset="rgi_id")
    .copy()
)
# rank by area within each country, largest = 0, second largest = 1, ...
glacier_info = glacier_info.sort_values(
    ["country"], ascending=[True]
)
glacier_info["country_rank"] = glacier_info.groupby("country").cumcount()

# choose a colormap with enough distinguishable colors
cmap = plt.cm.tab20
max_rank = glacier_info["country_rank"].max() + 1

# rank -> color
rank_to_color = {
    rank: cmap(rank % 20)
    for rank in range(max_rank)
}

# glacier_id -> color / marker
rgi_to_color = {
    row["rgi_id"]: rank_to_color[row["country_rank"]]
    for _, row in glacier_info.iterrows()
}

rgi_to_marker = {
    row["rgi_id"]: country_to_marker.get(row["country"], "o")
    for _, row in glacier_info.iterrows()
}

# arrays in plotting order
colors = [rgi_to_color[gid] for gid in merged["rgi_id"]]
markers = [rgi_to_marker[gid] for gid in merged["rgi_id"]]
for i, label in enumerate(glacier_names):
    ax.scatter(
        y.iloc[i],
        x.iloc[i],
        label=label,
        zorder=10,
        color=colors[i],
        marker=markers[i],
    )

# 1:1 line
ticks = np.arange(-2, 0.1, 0.5)
margin = (ticks[-1] - ticks[0]) * 0.05
x_min = ticks[0] - margin
x_max = ticks[-1] + margin
ax.plot([x_min, x_max], [x_min, x_max], "--", color="black", label="1:1 Correlation")

# labels & title
ax.set_ylabel("Calibrated mass balance [m i.e. / yr]")
ax.set_xlabel("GLAMOS + WGMS mass balance [m i.e. / yr]")
ax.set_title(f"Correlation r = {r:.2f}")

# grid and spines
ax.grid(axis="y", color="lightgray", linestyle="-", zorder=-10)
ax.grid(axis="x", color="lightgray", linestyle="-", zorder=-10)
for spine in ax.spines.values():
    spine.set_visible(False)

# hide ticks
ax.tick_params(left=False, bottom=False)

# equal aspect
ax.set_aspect("equal", adjustable="box")

# legend
ax.legend(
    loc="upper right",
    bbox_to_anchor=(2.5, 1),
    fontsize=10,
    ncol=2,
    columnspacing=1.2,
    handletextpad=0.5
)
# save figure
fig.savefig("compare_to_GLAMOS_WGMS.pdf", bbox_inches="tight")
