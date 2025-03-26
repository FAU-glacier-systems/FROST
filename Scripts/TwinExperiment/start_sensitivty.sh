RGI_ID="RGI2000-v7.0-G-11-01706"
SHORT_ID="${RGI_ID: -8}"

sbatch --output="../../Experiments/Log/${SHORT_ID}.out" \
--job-name="${SHORT_ID}" \
--error="../../Experiments/Log/${SHORT_ID}.err" elevation_step_sensitivity.sh \
--rgi_id "$RGI_ID" --calibrate

#sbatch --output="../../Experiments/Log/${SHORT_ID}.out" --job-name="${SHORT_ID}" \
#--error="../../Experiments/Log/${SHORT_ID}.err" obs_uncertainty_sensitivity.sh \
#--rgi_id "$RGI_ID" --calibrate
#
#sbatch --output="../../Experiments/Log/${SHORT_ID}.out" --job-name="${SHORT_ID}" \
#--error="../../Experiments/Log/${SHORT_ID}.err" obs_uncertainty_sensitivity.sh \
#--rgi_id "$RGI_ID" --calibrate