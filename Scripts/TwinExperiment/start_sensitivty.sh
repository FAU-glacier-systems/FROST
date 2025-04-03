name="elevation_step"
sbatch --output="../../Experiments/Log/${name}.out" --job-name="${name}" \
--error="../../Experiments/Log/${name}.err" elevation_step_sensitivity.sh

name="init_offset"
sbatch --output="../../Experiments/Log/${name}.out" --job-name="${name}" \
--error="../../Experiments/Log/${name}.err" initial_offset_sensitivity.sh

name="obs_uncertainty"
sbatch --output="../../Experiments/Log/${name}.out" --job-name="${name}" \
--error="../../Experiments/Log/${name}.err" obs_uncertainty_sensitivity.sh

name="ensemble_size"
sbatch --output="../../Experiments/Log/${name}.out" --job-name="${name}" \
--error="../../Experiments/Log/${name}.err" ensemble_size_sensitivity.sh

name="iterations"
sbatch --output="../../Experiments/Log/${name}.out" --job-name="${name}" \
--error="../../Experiments/Log/${name}.err" iterations_sensitivity.sh