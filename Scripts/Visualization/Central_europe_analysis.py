import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
csv_path = "../../Data/CentralEurope/RGI_SELECT.csv"
df = pd.read_csv(csv_path)

# Clean area data
df = df[pd.to_numeric(df['area_km2'], errors='coerce').notna()]
df = df[df['area_km2'] > 0]
df['area_km2'] = df['area_km2'].astype(float)

# Clean zmax_m data
df = df[pd.to_numeric(df['zmax_m'], errors='coerce').notna()]
df['zmax_m'] = df['zmax_m'].astype(float)

# Sort by area descending
df_sorted = df.sort_values(by='area_km2', ascending=True).reset_index(drop=True)
areas_sorted = df_sorted['area_km2'].values
zmax_sorted = df_sorted['cenlat'].values
zmin_sorted = df_sorted['cenlon'].values
ranks = np.arange(1, len(areas_sorted) + 1)

# Plot with color mapping to zmax
plt.figure(figsize=(10, 6))
# Scale area to a reasonable range for plotting marker size
# sqrt is common for visual area scaling (since area ~ radius²)
marker_sizes = areas_sorted*10  # scale factor can be adjusted

sc = plt.scatter(zmin_sorted, zmax_sorted, c=areas_sorted, cmap='viridis',
                 edgecolor='k', s=marker_sizes, norm=plt.Normalize(vmin=np.min(areas_sorted),
                                                        vmax=np.max(areas_sorted)))

#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('Rank (1 = largest)')
plt.ylabel('Glacier Area (km²)')
plt.title('Pareto Plot of Glacier Area (Colored by Max Elevation)')
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# Colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Maximum Elevation (m)')

plt.tight_layout()
plt.savefig('../../Plots/Pareto_Plot_Area_Zmax.png')
