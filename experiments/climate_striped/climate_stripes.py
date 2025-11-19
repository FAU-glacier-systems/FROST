import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file
df = pd.read_csv("gmt_HadCRUT5.csv")

# Rename columns for convenience
df = df.rename(columns={
    "HadCRUT5 (degC)": "temp",
    "HadCRUT5 uncertainty": "unc"
})

years = df["Year"].values
temps = df["temp"].values

# Normalize temps to [-1, 1] for colormap
tmin = temps.min()
tmax = temps.max()
abs_max = np.max(np.abs(temps))  # largest magnitude (pos/neg)

# norm in [0, 1], where:
#   0 -> -abs_max
#   0.5 -> 0
#   1 -> +abs_max
norm = (temps / abs_max + 1) / 2

# Choose color map (blue->white->red)
cmap = plt.get_cmap("RdBu_r")

colors = cmap(norm)

# Create the figure
fig, ax = plt.subplots(figsize=(8, 4))  # wide & short like climate stripes

# Plot vertical bars
ax.bar(years, temps, color=colors, width=1.0, edgecolor='none')

# Remove everything except the stripes

ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
ax.set_frame_on(False)

plt.tight_layout()
plt.savefig("climate_stripes.png", dpi=300)
