import os
import xarray as xr
import csv
import numpy as np
import pandas as pd

# Define directories
data_dir = "../../data/results/central_europe_sliding/glaciers"

# Create CSV file for results
csv_file_path = "../central_europe_sliding/inversion_results.csv"
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        'rgi_id', 'Mean_velsurf_mag', 'Std_velsurf_mag', 'Min_velsurf_mag',
        'Q1_velsurf_mag', 'Median_velsurf_mag', 'Q3_velsurf_mag', 'Max_velsurf_mag',
        'Mean_velsurfobs_mag', 'Std_velsurfobs_mag', 'Min_velsurfobs_mag',
        'Q1_velsurfobs_mag', 'Median_velsurfobs_mag', 'Q3_velsurfobs_mag',
        'Max_velsurfobs_mag'
    ])

# Iterate over all RGI-NetCDF files in the specified directory
for rgi_file in os.listdir(data_dir):
    rgi_path = os.path.join(data_dir, rgi_file)

    if os.path.isdir(rgi_path):  # Ensure we only process directories
        print(f"Processing RGI: {rgi_file}")

        # Construct the path to the Inversion folder
        inversion_path = os.path.join(rgi_path, "Preprocess", "outputs",)

        if os.path.exists(inversion_path):
            # Load the geology-optimized.nc file
            geo_file_path = os.path.join(inversion_path, "output.nc")

            if os.path.isfile(geo_file_path):
                print(f"  Loading {geo_file_path}")

                try:
                    # Open the NetCDF file
                    ds = xr.open_dataset(geo_file_path)

                    # Ensure required variables exist
                    if all(var in ds.variables for var in
                           ["velsurf_mag", "velsurfobs_mag", "icemask"]):
                        # Apply the ice mask
                        velsurf_mag = ds["velsurf_mag"].where(ds["icemask"] == 1)
                        velsurfobs_mag = ds["velsurfobs_mag"].where(
                            ds["icemask"] == 1)

                        # Compute statistics for velsurf_mag
                        mean_velsurf_mag = velsurf_mag.mean(skipna=True).item()
                        std_velsurf_mag = velsurf_mag.std(skipna=True).item()
                        min_velsurf_mag = velsurf_mag.min(skipna=True).item()
                        max_velsurf_mag = velsurf_mag.max(skipna=True).item()
                        q1_velsurf_mag = velsurf_mag.quantile(0.25).item()
                        median_velsurf_mag = velsurf_mag.quantile(0.5).item()
                        q3_velsurf_mag = velsurf_mag.quantile(0.75).item()

                        # Compute statistics for velsurfobs_mag
                        mean_velsurfobs_mag = velsurfobs_mag.mean(skipna=True).item()
                        std_velsurfobs_mag = velsurfobs_mag.std(skipna=True).item()
                        min_velsurfobs_mag = velsurfobs_mag.min(skipna=True).item()
                        max_velsurfobs_mag = velsurfobs_mag.max(skipna=True).item()
                        q1_velsurfobs_mag = velsurfobs_mag.quantile(0.25).item()
                        median_velsurfobs_mag = velsurfobs_mag.quantile(0.5).item()
                        q3_velsurfobs_mag = velsurfobs_mag.quantile(0.75).item()

                        print(f"    Mean velsurf_mag: {mean_velsurf_mag}")
                        print(f"    Mean velsurfobs_mag: {mean_velsurfobs_mag}")

                        # Write results to CSV
                        with open(csv_file_path, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow([
                                rgi_file, mean_velsurf_mag, std_velsurf_mag,
                                min_velsurf_mag, q1_velsurf_mag, median_velsurf_mag,
                                q3_velsurf_mag, max_velsurf_mag,
                                mean_velsurfobs_mag, std_velsurfobs_mag,
                                min_velsurfobs_mag, q1_velsurfobs_mag,
                                median_velsurfobs_mag, q3_velsurfobs_mag,
                                max_velsurfobs_mag
                            ])
                    else:
                        print(
                            "    Required variables (velsurf_mag, velsurfobs_mag, or icemask) are missing in the dataset!")

                    # Close the dataset
                    ds.close()
                except Exception as e:
                    print(f"    Error processing {geo_file_path}: {e}")
            else:
                print(f"  geology-optimized.nc not found in {inversion_path}")
        else:
            print(f"  Inversion folder not found for {rgi_file}")

