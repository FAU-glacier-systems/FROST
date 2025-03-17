#!/bin/bash

# Define the CSV file (update the path if necessary)
CSV_FILE="Data/GLAMOS/GLAMOS_SELECT.csv"

# Read CSV file and extract only the RGI_ID column (skip header)
tail -n +2 "$CSV_FILE" | cut -d, -f1 | while read -r RGI_ID; do
    # Submit the job
    echo "Submitting job for RGI ID: $RGI_ID"
    sbatch FROST_pipeline.sh --rgi_id "$RGI_ID"  --calibrate
    sleep 1  # Small delay to prevent overwhelming the scheduler
done

echo "All jobs submitted!"
