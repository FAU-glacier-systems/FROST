import os
import json
import pandas as pd
import numpy as np

# Global paths
RGI_FILES_PATH = "../../Data/CentralEurope/Split_Files"
EXPERIMENTS_PATH = "../../Experiments"
OUTPUT_CSV = "../../Scripts/CentralEurope/aggregated_results.csv"


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
        for _, row in rgi_data.iterrows():
            rgi_id = row["rgi_id"]
            result_path = os.path.join(
                EXPERIMENTS_PATH, f"RGI_SELECT_PART_{part}", rgi_id,
                "regional_run_v1", "result.json"
            )
            if not os.path.exists(result_path):
                print(f"Warning: Missing results.json for RGI ID {rgi_id}.")
                continue

            try:
                # Read JSON results
                with open(result_path, "r") as file:
                    data = json.load(file)
                    final_mean = data.get("final_mean", [])
                    final_std = data.get("final_std", [])

                    # Combine all RGI data columns + computed fields
                    row_data = row.to_dict()
                    aspect_deg = row_data.pop("aspect_deg")
                    
                    # Convert degrees to radians for calculations
                    aspect_rad = np.radians(aspect_deg)
                    
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
                    })
                    all_results.append(row_data)

            except json.JSONDecodeError:
                print(
                    f"Error: Failed to parse JSON for RGI ID {rgi_id}. Skipping...")

    # Convert the list of results to a DataFrame
    return pd.DataFrame(all_results)


def main():
    """
    Main function to run the script.
    """
    parts = np.arange(1, 27)  # Update as needed
    combined_results = load_and_collect_results(parts)

    if not combined_results.empty:
        # Ensure directory for OUTPUT_CSV exists
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

        # Save results to CSV
        combined_results.to_csv(OUTPUT_CSV, index=False)
        print(f"Results successfully saved to: {OUTPUT_CSV}")
    else:
        print("No results to save. Exiting...")


if __name__ == "__main__":
    main()