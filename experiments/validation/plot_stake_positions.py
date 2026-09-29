import os
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# run from the repository root, which holds the frost package
sys.path.insert(0, os.getcwd())
from frost.paths import DATA_RAW

HERE = Path(__file__).resolve().parent

# load
df = pd.read_csv(DATA_RAW / "DOI-WGMS-FoG-2025-02b" / "data" / "mass_balance_point.csv")

# filter glacier
df = df[df["glacier_name"].str.contains("GROSSER ALETSCH", case=False, na=False)]

# parse dates
df["begin_date"] = pd.to_datetime(df["begin_date"], errors="coerce")
df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

#keep september to september measurements
df = df[
    (df["begin_date"].dt.month == 9) &
    (df["end_date"].dt.month == 9)
]

# optional: only valid coordinates
df = df.dropna(subset=["longitude", "latitude"])
df["mid_date"] = df["begin_date"] + (df["end_date"] - df["begin_date"]) / 2
df["mid_year"] = df["mid_date"].dt.year

# plot
plt.figure(figsize=(4, 4))
sc = plt.scatter(df["longitude"], df["latitude"], c=df["elevation"], s=12, marker="o")

cbar = plt.colorbar(sc)
cbar.set_label("Year")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Measurement locations")
plt.tight_layout()
plt.savefig(HERE / "plots" / "wgms" / "stake_positions.pdf")