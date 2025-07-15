#!/usr/bin python3

# Copyright (C) 2024-2026 Oskar Herrmann
# Published under the GNU GPL (Version 3), check the LICENSE file

import os
import shutil
import json
import pandas as pd
import argparse


def main(rgi_id):
    # Define paths
    SOURCE_FOLDER = "../../Experiments/default"
    CSV_FILE = "../../Data/GLAMOS/GLAMOS_analysis_results.csv"

    # Create destination folder
    DEST_FOLDER = f"../../Experiments/{rgi_id}"
    os.makedirs(DEST_FOLDER, exist_ok=True)

    # Copy all JSON files from SOURCE_FOLDER to DEST_FOLDER
    for file in os.listdir(SOURCE_FOLDER):
        if file.endswith(".json"):
            shutil.copy(os.path.join(SOURCE_FOLDER, file), DEST_FOLDER)
        if file.endswith(".yaml"):
            shutil.copy(os.path.join(SOURCE_FOLDER, file), DEST_FOLDER)


    # Read the CSV file and extract values for the given rgi_id
    df = pd.read_csv(CSV_FILE)
    row = df[df["rgi_id"] == rgi_id]

    if row.empty:
        print(f"Error: No matching RGI ID found in {CSV_FILE}")
        exit(1)

    # Extract values from the CSV row
    ela_var = float(row["Annual_Variability_ELA"].values[0])
    gradabl_var = float(row["Annual_Variability_Ablation_Gradient"].values[0])
    gradacc_var = float(row["Annual_Variability_Accumulation_Gradient"].values[0])

    ela = float(row["Mean_ELA"].values[0])
    gradabl = float(row["Mean_Ablation_Gradient"].values[0])
    gradacc = float(row["Mean_Accumulation_Gradient"].values[0])

    # Define the JSON file path
    json_file = os.path.join(DEST_FOLDER, "params_calibration.json")

    # Update the JSON file
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        # Update reference_smb values
        if "reference_smb" in data:
            data["reference_smb"]["ela"] = ela
            data["reference_smb"]["gradabl"] = gradabl
            data["reference_smb"]["gradacc"] = gradacc
            data["reference_variability"]["ela"] = ela_var
            data["reference_variability"]["gradabl"] = gradabl_var
            data["reference_variability"]["gradacc"] = gradacc_var


        # Write the updated JSON file
        with open(json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Updated {json_file} successfully!")
    else:
        print(f"Error: {json_file} not found!")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Update JSON files based on RGI ID from CSV.")
    parser.add_argument("--rgi_id", type=str, help="The RGI ID to process.")
    args = parser.parse_args()
    main(args.rgi_id)
