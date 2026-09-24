# Central Europe: regional ELA calibration

Calibrates ELA and mass-balance gradients for all 409 glaciers > 1 km² in RGI region 11
(Central Europe) over 2000–2019 and validates them against in-situ data and end-of-summer
snowlines. This is the workflow behind Herrmann et al., *Brief Communication: Inferring
Glacier Equilibrium Line Altitudes in Central Europe with FROST* (EGUsphere, 2026).
The exact submitted state is on the `hydra` branch.

All scripts are run from this folder unless noted.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 0. Select glaciers | `Select_RGI_ID.py` | `../../data/raw/central_europe/Split_Files/RGI_SELECT_PART_*.csv` (batches of 10) |
| 1. Run FROST (HPC) | `start_multiple.sh <split file>` with `pipeline_config.yaml`, `params_inversion.yaml` | `../../data/results/central_europe_submit/glaciers/<rgi_id>/` |
| 2. Reference data | `validation/` (see `validation/README.md`) | `validation/tables/combined_ela_gradients.csv` |
| 3. Collect | `collect_results.py` | `tables/aggregated_results.csv` |
| 4. Plot | `Scatterplot_map.py` | `plots/ALPS_*` maps (Fig. 1) |
| | `evaluation.py` | `plots/SLA_regional_run.png` (Fig. 2), `plots/GLAMOS_regional_run.png` (Fig. 3) |
| | `correlation_plots.py` | `plots/correlation.png` (Supplement S1) |
| Diagnostics | `inversion_check/visualise_inversion_results.py` (run from `inversion_check/`) | `inversion_check/plots/inversion_results.png` |

Run step 1 once per split file, e.g. `bash start_multiple.sh ../../data/raw/central_europe/Split_Files/RGI_SELECT_PART_1.csv`.
The results folder is named after `experiment_name` in `pipeline_config.yaml` (`central_europe_submit`).

## Plot options

Every plotting script (including those in `validation/` and `inversion_check/`) accepts:

- `--pdf`: save as PDF instead of PNG (the default),
- `--dark`: dark theme with transparent background, for slides.

The shared handling lives in `plot_style.py`. On the maps, the colorbars, legend and glacier labels
sit on the DEM and stay white in both themes.

`evaluation.py` imports `frost.visualization.utils`, so run it with the repository root on the
path: `PYTHONPATH=../.. python evaluation.py`.
`Scatterplot_map.py` needs `cartopy`, which is not in the `igm32-frost` environment.

## Tables

`tables/aggregated_results.csv` is the one table all plots read. `collect_results.py` builds it from:

- per-glacier calibration results (`calibration_results.json`, `observations.nc`),
- ensemble statistics, cached in `tables/ensemble_stats.csv`,
- inversion statistics (velocity and thickness vs observations), cached in `tables/inversion_results.csv`,
- end-of-summer snowlines (`../../data/raw/central_europe/Alps_Glacier_EoS_SLA_2000-2019_stats_v2.csv`),
- the GLAMOS + WGMS reference (`validation/tables/combined_ela_gradients.csv`).

The two caches are only rebuilt when `RECOMPUTE_ENSEMBLE` / `RECOMPUTE_INVERSION` are set to `True`
in `collect_results.py`. Reference values appear in the aggregated table with the suffix `_wgmsglamos`.
