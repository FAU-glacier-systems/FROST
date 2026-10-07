# Alps TI projections

Temperature-index (TI) SMB calibration with FROST and CORDEX projections to 2100
for the 380 glaciers in `glacier_selection.csv` (European Alps). Each glacier is
inverted (thickness fixed to Millan et al. 2022 shifted to 2000, sliding fitted
to velocity), calibrated with the OGGM TI model (prior from its OGGM
calibration) against Hugonnet dh/dt 2000-2020, and run to 2100 with the
posterior mean under every complete CORDEX run (RCP2.6/4.5/8.5). Current state
and next steps: `STATUS.md`.

## Run

All commands from the repository root; IGM runs only as Slurm jobs on Alex.

| Step | Command | Output |
|---|---|---|
| 0. CORDEX forcing (vault node) | `bash experiments/alps_TI_projections/copy_cordex.sh` | `data/raw/cordex_alps/<id>/CORDEX_merged.nc` |
| 1. Pipeline + projections | `sbatch --array=1-N experiments/alps_TI_projections/alps_TI_sbatch.sh <list>` | `data/results/alps_TI_projections/<rgi_id>/` |
| | (projections only: `projections_sbatch.sh <rgi_id>`) | `<rgi_id>/Projection/projections.{nc,png}` |
| 2. Forcing overview | `python experiments/alps_TI_projections/plot_cordex_overview.py` | `data/results/alps_TI_projections/cordex_overview.png` |

`<list>` has `<rgi_id> <name>` per line (`test_glaciers.txt`: the 10 largest
glaciers and Rhone). `run_projections.py --collect_only` writes
`projections.nc` from the finished runs of a job that hit its time limit.

Diagnostics (one GPU job each):

| Script | Question |
|---|---|
| `emulator_check_sbatch.sh` | Pretrained emulator vs direct solver on the inverted state |
| `inversion_variants_sbatch.sh`, `inversion_variants.py --collect` | Inversion strategies (`inversion_variants/*.yaml`) against GlaThiDa and velocity |
| `diagnose_projection_sbatch.sh`, `tau_cap_test_sbatch.sh` | Time-step collapse in projections; velocity cap and smoother tau_ref |

## Inputs

- OGGM shop (RGI7, Millan thickness and velocity, GSWP3-W5E5 climate,
  `mb_calib.json`), downloaded by the pipeline
- Hugonnet et al. (2021) dh/dt: `data/raw/hugonnet/11_rgi60_2000-01-01_2020-01-01`
- CORDEX per glacier (Johannes J. Fürst): `data/raw/cordex_alps/<last 5 digits of rgi_id>/CORDEX_merged.nc`

## Outputs

- `data/results/alps_TI_projections/<rgi_id>/`: inversion (`Preprocess/`),
  calibration (`calibration_results.json`, `Ensemble/`, `Monitor/`),
  projections (`Projection/projections.nc` with yearly volume and area per
  CORDEX run, `projections.png`, `forcing/`)
- `data/results/alps_TI_inversion_variants/<variant>/<rgi_id>/`, `tables/`:
  inversion comparison
- `data/results/alps_TI_projections_igm_guess/`: test glaciers with the
  former inversion (IGM initial guess), for comparison

## Files

| File | Purpose |
|---|---|
| `pipeline_config.yml`, `params_inversion.yaml` | Pipeline (TI, OGGM prior, velocity cap) and inversion config |
| `glacier_selection.csv`, `test_glaciers.txt`, `largest10.txt` | Glacier lists (`area_km2` of the selection is not in km²) |
| `cordex_forcing.py`, `run_projections.py`, `plot_projections.py` | Bias-corrected forcing, projection runs, plot |
| `alps_TI_sbatch.sh`, `projections_sbatch.sh`, `copy_cordex.sh` | Job and copy scripts |
| `plot_cordex_overview.py` | Maps and time series of the forcing |
| `emulator_check.py`, `inversion_variants.py`, `diagnose_projection.py` (+ `_sbatch.sh`) | Diagnostics, see above |
| `STATUS.md` | Current state, decisions, findings |

Legacy workflow with Johannes J. Fürst (not runnable with the current layout:
paths from `experiments/`, scripts and vault folders that no longer exist):
`calibration.sh`, `meta_myscript*.sh`, `hpc_sbatch.sh`, `projections.py`,
`select_SMB_member.py`, `extract_CMIP_*.py`, `compile_output_ts.py`,
`check_volume2020.py`, `eALPS/`, `todo_codeskel*.txt`.
