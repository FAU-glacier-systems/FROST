import pandas as pd

# Define file paths
CSV_FILE_RGI = ("../../Data/RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G"
                "-11_central_europe-attributes.csv")  # First CSV
CSV_FILE_GLAMOS = "../../Data/GLAMOS/GLAMOS_RGI.csv"  # Second CSV
CSV_FILE_ELA = "../../Data/GLAMOS/glacier_analysis_results.csv"
OUTPUT_FILE = "../../Data/GLAMOS/GLAMOS_SELECT.csv"  # Output CSV

# Load both CSV files
rgi_df = pd.read_csv(CSV_FILE_RGI)
glamos_df = pd.read_csv(CSV_FILE_GLAMOS)
ela_df = pd.read_csv(CSV_FILE_ELA)

# Merge on 'RGI_ID' (Assuming both have this column)
merged_df = pd.merge(rgi_df, glamos_df, left_on="rgi_id", right_on="RGI_ID",
                     how="inner")
merged2_df = pd.merge(merged_df, ela_df, left_on="GLAMOS Name",
                      right_on="Glacier Name",
                      how="inner")
# keeps only
# matching RGI_IDs

filtered_df = merged2_df[merged_df["area_km2"] >= 1]
filtered_df = filtered_df[filtered_df["Years with ELA"] >= 10]

# Save the merged data to a new CSV file
filtered_df.to_csv(OUTPUT_FILE, index=False, sep=",")

print(f"Merged CSV saved to: {OUTPUT_FILE}")
