import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv("tables/aggregated_results.csv")

# Glacier names to filter
glacier_names = ["Kanderfirn", "Rhone", "Grosser Aletsch"]

# Filter the data for the specific glaciers
selected_glaciers = data[data["glac_name"].str.contains("|".join(glacier_names), case=False, na=False)]

# Function to compute SMB as a function of elevation
def compute_smb(elevation, ela, gradabl, gradacc):
    smb = np.where(
        elevation >= ela,
        gradacc * (elevation - ela),  # Accumulation zone
        gradabl * (ela - elevation)*-1,  # Ablation zone
    )
    return smb/1000

# Plotting
plt.figure(figsize=(5, 5))

# Loop over the top 10 glaciers and plot their SMB curve
for i, row in selected_glaciers.iterrows():
    ela = row["ela"]
    ela_std = row["ela_std"]
    gradabl = row["gradabl"]
    gradabl_std = row["gradabl_std"]
    gradacc = row["gradacc"]
    gradacc_std = row["gradacc_std"]
    zmin = row["zmin_m"]
    zmax = row["zmax_m"]

    # Generate elevation values
    elevation = np.linspace(zmin, zmax, 500)

    # Compute SMB
    smb_mean = compute_smb(elevation, ela, gradabl, gradacc)
    smb_plus = compute_smb(elevation, ela+ela_std, gradabl+gradabl_std,
                           gradacc+gradacc_std)
    smb_minus = compute_smb(elevation, ela-ela_std, gradabl-gradabl_std,
                            gradacc-gradacc_std)
    # Plot SMB curve
    plt.plot(elevation, smb_mean, label=f"{row['glac_name']} (ELA={int(ela)})")
    plt.fill_between(elevation, smb_minus, smb_plus, alpha=0.2)

# Label the axes
plt.axhline(0, color="black", linestyle="--", linewidth=1, label="0 SMB Line (ELA)")
plt.title("SMB as a Function of Elevation", fontsize=14)
plt.xlabel("Elevation (m)", fontsize=12)
plt.ylabel("SMB (meters per year)", fontsize=12)
plt.legend(fontsize=9)
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig("../../Plots/smb_as_a_function_of_elevation.png", dpi=200)
