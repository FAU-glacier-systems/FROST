import pandas as pd
import os

# Load the CSV file
df = pd.read_csv("../../Data/RGI2000-v7.0-G-11_central_europe/RGI2000-v7.0-G-11_central_europe-attributes.csv")

# Filter rows where 'area_km2' is greater than 1
df_filtered = df[df["area_km2"] > 1]

# Sort by area_km2 (descending or ascending – choose your preference)
df_filtered = df_filtered.sort_values(by="area_km2", ascending=False)

# Output directory
output_dir = "../../Data/CentralEurope/Split_Files"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Split DataFrame into chunks of 10 rows
for i in range(0, len(df_filtered), 10):
    chunk = df_filtered.iloc[i:i+10]
    file_name = os.path.join(output_dir, f"RGI_SELECT_PART_{i//10 + 1}.csv")
    chunk.to_csv(file_name, index=False)

print(f"Split complete! {len(df_filtered) // 10 + 1} files created in {output_dir}")
