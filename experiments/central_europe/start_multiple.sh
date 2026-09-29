#!/bin/bash
# Submits one FROST job per glacier of a split file. Run from the
# repository root:
#   bash experiments/central_europe/start_multiple.sh \
#       data/raw/central_europe/Split_Files/RGI_SELECT_PART_1.csv

# Define the CSV file from the first argument
CSV_FILE="$1"
CSV_BASENAME=$(basename "$CSV_FILE" .csv)


# Read CSV file and extract only the RGI_ID column (skip header)
tail -n +2 "$CSV_FILE" | cut -d, -f1 | while read -r RGI_ID; do
    # Submit the job
    SHORT_ID="${RGI_ID: -8}"

    echo "Submitting job for RGI ID: $RGI_ID"
   sbatch --output="logs/${SHORT_ID}.out"\
   --job-name="${SHORT_ID}"\
   --error="logs/${SHORT_ID}.err" \
   hpc_sbatch.sh "experiments/central_europe/pipeline_config.yaml" "$RGI_ID"


done

echo "All jobs submitted!"
