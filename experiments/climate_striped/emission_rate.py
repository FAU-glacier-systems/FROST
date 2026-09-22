import numpy as np
import matplotlib.pyplot as plt

# Path to your NOAA .txt file
fname = "emission_rate.txt"  # <-- change if needed

# Load the table while skipping comment lines
years = []
growth = []
unc = []

with open(fname, "r") as f:
    for line in f:
        # skip header or empty lines
        if line.strip().startswith("#") or line.strip() == "":
            continue

        # split into columns
        parts = line.split()
        if len(parts) == 3:
            y, g, u = parts
            years.append(int(y))
            growth.append(float(g))
            unc.append(float(u))

years = np.array(years)
growth = np.array(growth)
unc = np.array(unc)

# --- Plot ---
plt.figure(figsize=(10, 5))

plt.plot(years, growth, lw=2)
plt.fill_between(years, growth - unc, growth + unc, alpha=0.2, label="Uncertainty")

plt.xlabel("Year")
plt.ylabel("CO$_2$ annual increase (ppm yr$^{-1}$)")
plt.title("NOAA Global Monitoring Laboratory – Annual CO$_2$ Growth Rate")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
