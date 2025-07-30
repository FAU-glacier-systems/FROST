#!/bin/bash

# Define the CSV file from the first argument
CSV_FILE="$1"
CSV_BASENAME=$(basename "$CSV_FILE" .csv)

# Read CSV file and extract only the RGI_ID column (skip header)
tail -n +2 "$CSV_FILE" | cut -d, -f1 | while read -r RGI_ID; do
    # Submit the job
    SHORT_ID="${RGI_ID: -8}"

    echo "Submitting job for RGI ID: $RGI_ID"
   sbatch --output="Experiments/${CSV_BASENAME}/Log/${SHORT_ID}.out"\
   --job-name="${SHORT_ID}"\
   --error="Experiments/${CSV_BASENAME}/Log/${SHORT_ID}.err" \
   FROST_pipeline.sh \
   --rgi_id "$RGI_ID"  \
   --inversion \
   --results_dir "Experiments/${CSV_BASENAME}/${RGI_ID}/regional_run_samuel"

done

echo "All jobs submitted!"
