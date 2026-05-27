import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# load + filter
df = pd.read_csv("../../data/raw/DOI-WGMS-FoG-2025-02b/data/mass_balance_point.csv")
df = df[df["glacier_name"].str.contains("GROSSER ALETSCH", case=False, na=False)]

# dates
df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

# september to september
df = df[
    (df["begin_date"].dt.month == 9) &
    (df["end_date"].dt.month == 9)
]

# clean
df = df.dropna(subset=["elevation", "balance", "begin_date", "end_date", "longitude", "latitude"])

# mid year
df["mid_date"] = df["begin_date"] + (df["end_date"] - df["begin_date"]) / 2
df["mid_year"] = df["mid_date"].dt.year

# optional filter
df = df[df["mid_year"].between(1970, 1972)]
df = df[df["elevation"].between(2000, 3000)]

# one label per observation period
df["period_label"] = (
    df["begin_date"].dt.strftime("%Y-%m-%d")
    + " to "
    + df["end_date"].dt.strftime("%Y-%m-%d")
)

# --- 50 m binning for left panel ---
bin_size = 25
zmin = np.floor(df["elevation"].min() / bin_size) * bin_size
zmax = np.ceil(df["elevation"].max() / bin_size) * bin_size
bins = np.arange(zmin, zmax + bin_size, bin_size)

df["elev_bin"] = pd.cut(df["elevation"], bins=bins, right=False)
df["z"] = df["elev_bin"].apply(lambda x: x.left + bin_size / 2 if pd.notnull(x) else np.nan)

# mean and std per year and elevation bin
df_year = (
    df.groupby(["mid_year", "z"], observed=True)["balance"]
    .agg(["mean", "min", "max"])
    .reset_index()
)

# distinct colors for left panel
years = sorted(df["mid_year"].unique())
line_cmap = plt.cm.tab10
line_colors = line_cmap(np.linspace(0, 1, len(years)))
color_dict_year = dict(zip(years, line_colors))

# markers for right panel
marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
marker_dict = {year: marker_list[i % len(marker_list)] for i, year in enumerate(years)}

# --- figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# left: yearly mean profiles with std shading
handles = []
labels = []

for year in years:
    group = df_year[df_year["mid_year"] == year].sort_values("z")
    color = color_dict_year[year]

    # mean line
    line, = ax1.plot(
        group["z"],
        group["mean"],
        color=color,
        linewidth=1.8
    )

    # min–max envelope
    ax1.fill_between(
        group["z"],
        group["min"],
        group["max"],
        color=color,
        alpha=0.2
    )

    handles.append(line)
    labels.append(str(year))

ax1.set_xlabel("Elevation (m)")
ax1.set_ylabel("Mass balance (m w.e. yr$^{-1}$)")
ax1.set_title(f"Yearly mean profiles ({bin_size} m bins)")
ax1.legend(handles, labels, frameon=False, fontsize=8, title="Year")

# right: raw positions, color = elevation, marker = year
elev_norm = mcolors.Normalize(df["elevation"].min(), df["elevation"].max())
cmap = plt.cm.viridis

for year in years:
    group = df[df["mid_year"] == year]

    ax2.scatter(
        group["longitude"],
        group["latitude"],
        facecolors='none',
        edgecolors=cmap(elev_norm(group["elevation"])),
        marker=marker_dict[year],
        s=40,
        linewidth=1
    )

# colorbar for elevation
sm = plt.cm.ScalarMappable(norm=elev_norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2)
cbar.set_label("Elevation (m)")

ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
ax2.set_title("Measurement locations")

# right legend only for marker/year
handles = [
    plt.Line2D(
        [0], [0],
        marker=marker_dict[year],
        color="black",
        linestyle="",
        markersize=6,
        label=str(year)
    )
    for year in years
]
ax2.legend(handles=handles, frameon=False, title="Year")

# styling
for ax in [ax1, ax2]:
    ax.grid(axis="y", color="black", linestyle="-", zorder=-1, alpha=0.2)
    ax.grid(axis="x", color="black", linestyle="-", zorder=-1, alpha=0.2)
    ax.xaxis.set_tick_params(bottom=False)
    ax.yaxis.set_tick_params(left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

plt.tight_layout()
plt.savefig("Plots/combined_year_profiles_positions.png", bbox_inches="tight", dpi=300)
print("Saved figure to Plots/combined_year_profiles_positions.png")
plt.show()