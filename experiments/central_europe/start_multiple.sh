#!/bin/bash

cd ../../
# Define the CSV file from the first argument
CSV_FILE="$1"
CSV_BASENAME=$(basename "$CSV_FILE" .csv)


# Read CSV file and extract only the RGI_ID column (skip header)
tail -n +2 "$CSV_FILE" | cut -d, -f1 | while read -r RGI_ID; do
    # Submit the job
    SHORT_ID="${RGI_ID: -8}"

    echo "Submitting job for RGI ID: $RGI_ID"
   sbatch --output="data/raw/log/${SHORT_ID}.out"\
   --job-name="${SHORT_ID}"\
   --error="data/raw/log/${SHORT_ID}.err" \
   hpc_sbatch.sh "experiments/central_europe/config.yml" "$RGI_ID" \
   "central_europe"


done

echo "All jobs submitted!"
