#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
from netCDF4 import Dataset
import pandas as pd
import utm
import numpy as np
import matplotlib.pyplot as plt

rgi__dir =  '../../Data/Glaciers/RGI2000-v7.0-G-17-14719/'
oggm_dataset_path = os.path.join(rgi__dir, 'OGGM_shop', 'input_saved.nc')

thk_measure_path = os.path.join(rgi__dir, 'icethk_measurement.csv')
thk_measure = pd.read_csv(thk_measure_path)
lon = thk_measure["LON"].values[::50]
lat = thk_measure["LAT"].values[::50]
thk = thk_measure["icthckn"].values[::50]


with Dataset(oggm_dataset_path, 'r') as oggm_shop_dataset:
    icemask = oggm_shop_dataset.variables['icemask'][:]
    x_coords = oggm_shop_dataset['x'][:]
    y_coords = oggm_shop_dataset['y'][:]
    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()

zone_number = 18
zone_letter = "S"

easting, northing, zn, zl = utm.from_latlon(lat, lon, zone_number, zone_letter)

# Translate easting/northing to icemask grid indices
x_indices = ((easting - min_x) / (x_coords[1] - x_coords[0])).astype(int)
y_indices = ((northing  - min_y) / (y_coords[1] - y_coords[0])).astype(int)
y_indices -=  100000

thk_raster = np.full(icemask.shape, np.nan)  # Fill with NaN initially

# Assign thickness values to the raster
for x, y, t in zip(x_indices, y_indices, thk):
    if 0 <= x < thk_raster.shape[1] and 0 <= y < thk_raster.shape[0]:  # Ensure indices are within bounds
        if np.isnan(thk_raster[y, x]):  # If no value exists yet
            thk_raster[y, x] = t
        else:
            # Handle overlaps: Example - Average existing value and new value
            thk_raster[y, x] = (thk_raster[y, x] + t) / 2

# Plot icemask and scatter points
plt.figure(figsize=(10, 8))
plt.imshow(icemask, cmap='gray', origin='lower',)

with Dataset(oggm_dataset_path, mode='r+') as nc_file:
    # Check if the variable 'thkobs' already exists
    if 'thkobs' in nc_file.variables:
        print("Variable 'thkobs' already exists. Overwriting it.")
        thkobs_var = nc_file.variables['thkobs']
    else:
        # Create the new variable 'thkobs'
        thkobs_var = nc_file.createVariable(
            'thkobs',  # Variable name
            'f4',      # Data type (float32)
            ('y', 'x'),  # Dimensions (matching icemask)
            fill_value=-9999  # Fill value for missing data
        )
        thkobs_var.units = "m"  # Add metadata
        thkobs_var.long_name = "Observed Ice Thickness"

    # Write the raster data to the 'thkobs' variable
    thkobs_var[:] = thk_raster