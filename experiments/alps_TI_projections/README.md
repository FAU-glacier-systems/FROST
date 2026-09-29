# Alps TI projections

Temperature-index (TI) SMB calibration and CMIP/CORDEX projections for glaciers in
the European Alps (eALPS), with Johannes J. Fürst. For each glacier in a to-do
list, FROST calibrates the TI model, selects the best ensemble member and runs it
forward with representative climate scenarios.

**Status: not runnable as is.** The scripts predate the current pipeline layout:
they start from `experiments/` (`cd ..`), call scripts that no longer exist
(`projections_parallel.py`, `./batchscripts/*.py`), read climate data from
`/home/vault/gwgi/gwgifu1h/`, and expect results in
`data/results/{alps_TI_projections,eALPS}/glaciers/<rgi_id>/`. Bring them up to
date before the next run.

## Run

Intended order (from `experiments/`, see Status):

1. `calibration.sh`: FROST calibration per glacier of `todo_codeskel.txt`,
   configs generated from `eALPS/config_TI_skel.yml`
2. `meta_myscript_projections_noCAL.sh`: member selection and projections
   (`projections.py`, `select_SMB_member.py`, `extract_CMIP_*.py`,
   `compile_output_ts.py`, `check_volume2020.py`)

## Inputs

- OGGM shop with TI climate (downloaded by the pipeline)
- CORDEX/CMIP climate per glacier (Johannes' vault folder, see Status)

## Outputs

- `data/results/alps_TI_projections/glaciers/<rgi_id>/`,
  `data/results/eALPS/glaciers/<rgi_id>/` (layout before the `glaciers/` level was dropped)

## Files

| File | Purpose |
|---|---|
| `eALPS/config_TI*.yml`, `params_inversion.yaml` | Pipeline and inversion configs (skeletons per glacier) |
| `todo_codeskel*.txt` | Glacier lists |
| `calibration.sh`, `meta_myscript_noCAL.sh`, `meta_myscript_projections_noCAL.sh`, `hpc_sbatch.sh` | Job scripts |
| `projections.py`, `select_SMB_member.py` | Member selection and forward runs |
| `extract_CMIP_climate_historical.py`, `extract_CMIP_representative.py` | Climate forcing |
| `compile_output_ts.py`, `check_volume2020.py` | Post-processing |
