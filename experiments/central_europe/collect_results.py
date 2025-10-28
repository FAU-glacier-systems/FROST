import os
import json
from cmath import isnan

import pandas as pd
import numpy as np
import xarray as xr
from os.path import join, dirname, exists

# Global paths
#RGI_FILES_PATH = "../../data/raw/central_europe/Split_Files"
RGI_FILES_PATH = "../../data/raw/central_europe/Split_Files"
SLA_PATH ="../../data/raw/central_europe/Alps_Glacier_EoS_SLA_2000-2019_stats.csv"
GLAMOS_PATH ="../../data/raw/glamos/GLAMOS_analysis_results.csv"


INVERSION_PATH = "../central_europe_1000/tables/inversion_results.csv"
EXPERIMENTS_PATH = "../../data/results/central_europe_1000/glaciers"
OUTPUT_CSV = "../central_europe_1000/aggregated_results.csv"


def load_and_collect_results(parts):
    """
    Load RGI data and collect relevant results with JSON data for specified parts.
    Returns a combined DataFrame.
    """
    all_results = []

    for part in parts:
        print(part)
        # Load RGI CSV
        file_path = os.path.join(RGI_FILES_PATH, f"RGI_SELECT_PART_{part}.csv")
        if not os.path.exists(file_path):
            print(f"Error: CSV file {file_path} not found.")
            continue

        rgi_data = pd.read_csv(file_path)
        if rgi_data.empty:
            print(f"Warning: RGI file for part {part} is empty.")
            continue

        # Collect results for each row
        for i, row in rgi_data.iterrows():
            rgi_id = row["rgi_id"]
            result_path = os.path.join(
                EXPERIMENTS_PATH, rgi_id, "calibration_results.json"
            )
            if not os.path.exists(result_path):
                print(f"Warning: Missing results.json for RGI ID {rgi_id}.")
                continue

            try:
                # Read JSON results
                with open(result_path, "r") as file:
                    print(f"Found results.json for RGI ID {rgi_id}.")
                    data = json.load(file)
                    final_mean = data.get("final_mean", [])
                    final_std = data.get("final_std", [])

                    # Combine all RGI data columns + computed fields
                    row_data = row.to_dict()
                    aspect_deg = row_data.pop("aspect_deg")
                    
                    # Convert degrees to radians for calculations
                    aspect_rad = np.radians(aspect_deg)

                    ### collect ensemble data
                    ensemble_path = join(dirname(result_path), 'Ensemble')
                    if exists(ensemble_path):
                        vel_data_ensemble = []
                        smb_data_ensemble = []
                        dhdt_data_ensemble = []
                        for member in os.listdir(ensemble_path)[::10]:
                            output_path = join(ensemble_path, member, 'outputs', 'output.nc')
                            if exists(output_path):
                                try:
                                    with xr.open_dataset(output_path) as ds:
                                        if 'velsurf_mag' in ds:
                                            vel = ds['velsurf_mag']
                                            smb = ds['smb']
                                            usurf = ds['usurf']

                                            # Get values at specific indices
                                            vel_data_member = []
                                            smb_data_member = []
                                            for idx in [0, 1, 2]:
                                                if idx < len(vel):
                                                    vel_year = vel[idx].values[ds['thk'][idx] > 1]
                                                    vel_data_member.append(np.nanmean(vel_year))

                                                    smb_year = smb[idx].values[ds['thk'][idx] > 1]
                                                    smb_data_member.append(np.nanmean(smb_year))


                                            dhdt = (usurf[2].values - usurf[0].values)/20

                                            dhdt_data_ensemble.append(np.mean(dhdt[ds['thk'][0] > 1]))

                                            vel_data_ensemble.append(vel_data_member)
                                            smb_data_ensemble.append(smb_data_member)


                                except Exception as e:
                                    print(f"Error reading netCDF file {output_path}: {e}")

                        vel_data_ensemble = np.array(vel_data_ensemble)
                        vel_data_ensemble = np.nanmean(vel_data_ensemble, axis=0)

                        smb_data_ensemble_std = np.nanstd(smb_data_ensemble, axis=0)
                        smb_data_ensemble_mean = np.nanmean(smb_data_ensemble, axis=0)
                        dhdt_data_ensemble_mean = np.nanmean(dhdt_data_ensemble, axis=0)
                        dhdt_data_ensemble_std = np.nanstd(dhdt_data_ensemble, axis=0)


                    observations_path = join(dirname(result_path), 'observations.nc')

                    with xr.open_dataset(observations_path) as ds:
                        usurf_obs = ds['usurf']
                        thk = ds['thk'][0]

                        dhdt = (usurf_obs[1].values - usurf_obs[0].values)/20
                        dhdt_mean = np.mean(dhdt[thk > 1])


                        thk_mask = ds['thk'][0]>1
                        dhdt_err = ds['dhdt_err'][1].to_numpy()
                        dhdt_err = dhdt_err[thk_mask.to_numpy()]
                        dhdt_std = dhdt_err.mean()

                    if isnan(np.mean(smb_data_ensemble_mean)):
                        print(rgi_id)
                    # Add the sine and cosine values
                    row_data.update({
                        "EastWest": np.sin(aspect_rad),
                        "SouthNorth": np.cos(aspect_rad),
                        "ela": final_mean[0] if len(final_mean) > 0 else None,
                        "gradabl": final_mean[1] if len(final_mean) > 1 else None,
                        "gradacc": final_mean[2] if len(final_mean) > 2 else None,
                        "ela_std": final_std[0] if len(final_std) > 0 else None,
                        "gradabl_std": final_std[1] if len(final_std) > 1 else None,
                        "gradacc_std": final_std[2] if len(final_std) > 2 else None,
                        "vel_ensemble_year0": vel_data_ensemble[0],
                        "vel_ensemble_year10": vel_data_ensemble[1],
                        "vel_ensemble_year20": vel_data_ensemble[2],
                        "smb_ensemble_year0": smb_data_ensemble_mean[0],
                        "smb_ensemble_year10": smb_data_ensemble_mean[1],
                        "smb_ensemble_year20": smb_data_ensemble_mean[2],
                        "smb_ensemble_mean": np.mean(smb_data_ensemble_mean),
                        "smb_ensemble_std": np.mean(smb_data_ensemble_std),
                        "dhdt_mean": dhdt_mean,
                        "dhdt_std": dhdt_std,
                        "dhdt_ensemble_mean": dhdt_data_ensemble_mean,
                        "dhdt_ensemble_std": dhdt_data_ensemble_std,
                    })
                    all_results.append(row_data)





            except json.JSONDecodeError:
                print(
                    f"Error: Failed to parse JSON for RGI ID {rgi_id}. Skipping...")

    # Convert the list of results to a DataFrame
    return pd.DataFrame(all_results)


def _merge_drop_duplicates(left: pd.DataFrame, right: pd.DataFrame, on: str, suffix: str) -> pd.DataFrame:
    """
    Merge two dataframes and drop columns from 'right' that duplicate columns in 'left'.
    Keeps values from 'left' when name collisions occur.
    """
    merged = pd.merge(left, right, on=on, how="left", suffixes=("", suffix))
    # Drop only the suffixed duplicates where the unsuffixed already exists
    to_drop = []
    for col in merged.columns:
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in merged.columns:
                to_drop.append(col)
    if to_drop:
        merged = merged.drop(columns=to_drop)
    return merged


def main():
    """
    Main function to run the script.
    """
    parts = np.arange(1, 42)  # Update as needed
    combined_results = load_and_collect_results(parts)

    if not combined_results.empty:
        # Load additional sources
        # 1) Inversion: include all columns
        if os.path.exists(INVERSION_PATH):
            inversion_df = pd.read_csv(INVERSION_PATH)
        else:
            print(f"Warning: INVERSION_PATH '{INVERSION_PATH}' not found. Skipping inversion merge.")
            inversion_df = pd.DataFrame(columns=["rgi_id"])

        # 2) SLA: only 'rgi_id' and 'sla'
        if os.path.exists(SLA_PATH):
            sla_df_all = pd.read_csv(SLA_PATH)
            sla_cols = [c for c in sla_df_all.columns if c in ("rgi_id", "sla_mean", "sla_n")]
            sla_df = sla_df_all.loc[:, sla_cols].copy() if sla_cols else pd.DataFrame(columns=["rgi_id", "sla_mean", "sla_n"])
        else:
            print(f"Warning: SLA_PATH '{SLA_PATH}' not found. Skipping SLA merge.")
            sla_df = pd.DataFrame(columns=["rgi_id", "sla_mean", "sla_n"])

        # 3) GLAMOS: ELA, accumulation gradient, ablation gradient + their variability
        if os.path.exists(GLAMOS_PATH):
            glamos_all = pd.read_csv(GLAMOS_PATH)
            wanted_glamos_cols = [
                "rgi_id",
                "glamos_name",
                "Mean_ELA",
                "Annual_Variability_ELA",
                "Mean_Ablation_Gradient",
                "Annual_Variability_Ablation_Gradient",
                "Mean_Accumulation_Gradient",
                "Annual_Variability_Accumulation_Gradient",
                "annual_mass_balance",
                "annual_mass_balance_std",
            ]
            glamos_cols = [c for c in wanted_glamos_cols if c in glamos_all.columns]
            glamos_df = glamos_all.loc[:, glamos_cols].copy() if glamos_cols else pd.DataFrame(columns=["rgi_id"])
        else:
            print(f"Warning: GLAMOS_PATH '{GLAMOS_PATH}' not found. Skipping GLAMOS merge.")
            glamos_df = pd.DataFrame(columns=["rgi_id"])

        # Merge by RGI ID
        merged = combined_results.copy()
        if not inversion_df.empty:
            merged = _merge_drop_duplicates(merged, inversion_df, on="rgi_id", suffix="_inv")
        if not sla_df.empty:
            merged = _merge_drop_duplicates(merged, sla_df, on="rgi_id", suffix="_drop")
        if not glamos_df.empty:
            merged = _merge_drop_duplicates(merged, glamos_df, on="rgi_id", suffix="_drop")

        # Save results to CSV
        merged.to_csv(OUTPUT_CSV, index=False)
        print(f"Results successfully saved to: {OUTPUT_CSV}")
    else:
        print("No results to save. Exiting...")


if __name__ == "__main__":
    main()