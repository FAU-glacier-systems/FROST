import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "monospace"      # or "serif", "sans-serif"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono"]

# =========================
# 1. LOAD PAGES2k DATA
# =========================
data = np.genfromtxt(
    "Full_ensemble_median_and 95pct_range.txt",
    delimiter="\t",
    comments="#",
    missing_values="NA",
    filling_values=np.nan,
    skip_header=1
)

# Columns from PAGES2k file
year_pages = data[:, 0]
median_pages = data[:, 2]   # Full ensemble median (°C rel. 1961–1990)

# =========================
# 2. LOAD HADCRUT5 DATA
# =========================
df = pd.read_csv("gmt_HadCRUT5.csv")
df = df.rename(columns={
    "HadCRUT5 (degC)": "temp",
    "HadCRUT5 uncertainty": "unc"
})

years_had = df["Year"].values
temps_had = df["temp"].values

# =========================
# 3. REBASE BOTH TO 1850–1900
# =========================
# PAGES2k rebasing
mask_1850_1900 = (year_pages >= 1850) & (year_pages <= 1900)
baseline_pages = np.nanmean(median_pages[mask_1850_1900])
median_pages_rebased = median_pages - baseline_pages

# HADCRUT5 rebasing
baseline_had = np.nanmean(temps_had[(years_had >= 1850) & (years_had <= 1900)])
temps_had_rebased = temps_had - baseline_had

# =========================
# 4. CONCATENATE THE TWO SERIES
#    PAGES2k covers 1–1900, HadCRUT5 covers 1850–today
# =========================
# Keep PAGES2k until 1900, then use HadCRUT5 afterwards
cut_index = np.where(years_had >= 1900)[0][0]

concat_years = np.concatenate((year_pages[year_pages <= 1900],
                               years_had[cut_index:]))

concat_temps = np.concatenate((median_pages_rebased[year_pages <= 1900],
                               temps_had_rebased[cut_index:]))

# =========================
# 5. NORMALIZE TO [-1, 1] FOR COLOR MAPPING
# =========================
abs_max = np.nanmax(np.abs(concat_temps))
norm = (concat_temps / abs_max + 1) / 2  # convert to [0, 1]

cmap = plt.get_cmap("RdBu_r")
colors = cmap(norm)

# =========================
# 6. PLOT STRIPES
# =========================
fig, ax = plt.subplots(figsize=(4, 4))

ax.bar(concat_years, concat_temps, color=colors, width=1.0, edgecolor='none', zorder=10)
#ax.plot(concat_years,concat_temps, color='black', linewidth=0.1, zorder=10)
ax.plot(year_pages[year_pages <= 1900], median_pages_rebased[year_pages <= 1900], color='gray', linewidth=0.3, zorder=10)

ax.plot(years_had[cut_index:], temps_had_rebased[cut_index:], color='black', linewidth=0.3, zorder=10)
# Labels and title
# Main text (black)
ax.axhline(0, color='black', linewidth=0.6, linestyle='-', zorder=50)
ax.scatter(2025, temps_had_rebased[-2], color=colors[-2], zorder=100, edgecolors='black', )
point_x = 2000
point_y = temps_had_rebased[-2]-0.01

ax.annotate(
    "heading for +3°C by 2100\n great for mosquitoes,\n bad for literally everything else.",
    xy=(point_x, point_y),          # point to annotate
    xytext=(point_x-150, point_y-0.3),   # text position (adjust as needed)
    arrowprops=dict(
        arrowstyle="->",
        lw=1,
        color="black"
    ),
    ha="right",
    va="bottom",
    fontsize=6,
    bbox=dict(facecolor="white", edgecolor="none", pad=2)
)

ax.text(
    0.5, 0.61,
    "Global Temperature Rise",
    ha="center", va="center",
    fontsize=13, zorder=8,
    transform=ax.transAxes,
bbox = dict(facecolor="white", edgecolor="none", pad=4)

)
ax.text(
    0.5, 0.55,
    "Baseline: 1850 – 1900" ,     # leave space for PAGES2k
    ha="center", va="center",
    fontsize=10, zorder=10,
    transform=ax.transAxes,
bbox = dict(facecolor="white", edgecolor="none", pad=4)
)
ax.text(
    0.5, 0.5,
    "[        + HadCRUT5]" ,     # leave space for PAGES2k
    ha="center", va="center",
    fontsize=8, zorder=12,
    transform=ax.transAxes,
bbox = dict(facecolor="white", edgecolor="none", pad=4)
)

# Overlay the gray "PAGES2k" at the right spot
ax.text(
    0.5 - 0.1, 0.5,     # small manual offset tuning
    "PAGES2k",
    ha="center", va="center",
    fontsize=8, zorder=13,
    color="gray",
    transform=ax.transAxes
)

ax.set_xlabel("Year", fontsize=8)

# Clean look: minimal axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_ylim(np.nanmin(concat_temps))

ax.grid(False)
plt.xticks([0, 500, 1000, 1500,  2025])
plt.yticks(ticks=[-0.2, 0.5, 1, 1.5], labels=["-0.2 °C", "+0.5 °C", "+1 °C", "+1.5 °C"])

# Place horizontal line at 0°C anomaly
ax.grid(axis="y", color="lightgray", linestyle="-", )
ax.grid(axis="x", color="lightgray", linestyle="-",)
ax.xaxis.set_tick_params(bottom=False)
ax.yaxis.set_tick_params(left=False)

# Move labels inside
ax.tick_params(axis='y', direction='in', pad=-2)
for label in ax.get_yticklabels():
    label.set_ha("left")
    label.set_bbox(dict(facecolor="white", edgecolor="none", pad=2))
    label.set_zorder(50)





plt.tight_layout()
plt.savefig("climate_stripes_pages2k.svg", dpi=300, bbox_inches="tight")
plt.savefig("climate_stripes_pages2k.png", dpi=300, bbox_inches="tight")
plt.savefig("climate_stripes_pages2k.pdf", dpi=300, bbox_inches="tight")

