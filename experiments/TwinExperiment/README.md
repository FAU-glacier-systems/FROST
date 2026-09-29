# Twin experiment: sensitivity of the EnKF calibration

Synthetic twin experiments on Rhone: the calibration is run against synthetic
observations of a known SMB, varying one setting at a time (ensemble size,
iterations, elevation-band width, observation uncertainty, initial offset,
inflation), each over several seeds.

**Status: not runnable as is.** The scripts belong to an older FROST layout: they
call `FROST_calibration.py`, `run_calibration.py` and `../Preprocess/*.py`, write
to `Experiments/`, and use glacier IDs with suffixes (`_v1`, `_v3`). The six
sweep scripts differ only in the swept setting; a rewrite would be one sweep
script with the setting and values as arguments.

## Run

`start_sensitivty.sh` submitted the five sweeps; `Inflation_analysis.sh` the
inflation runs (see Status).

## Inputs

- Synthetic observations of a FROST run on Rhone

## Outputs

- `Experiments/<rgi_id>/<setting>/<value>/Seed_<seed>/` (old layout)

## Files

| File | Purpose |
|---|---|
| `start_sensitivty.sh` | Submits the five sweeps |
| `ensemble_size_sensitivity.sh`, `iterations_sensitivity.sh`, `elevation_step_sensitivity.sh`, `obs_uncertainty_sensitivity.sh`, `initial_offset_sensitivity.sh` | One sweep each |
| `Inflation_analysis.sh` | Inflation factor sweep |
