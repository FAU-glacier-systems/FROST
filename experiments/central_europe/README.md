# Central Europe: regional ELA calibration

Calibrates ELA and mass-balance gradients for all 409 glaciers > 1 km² in RGI region 11
(Central Europe) over 2000–2019 and validates them against in-situ data and end-of-summer
snowlines. This is the workflow behind Herrmann et al., *Brief Communication: Inferring
Glacier Equilibrium Line Altitudes in Central Europe with FROST* (EGUsphere, 2026).
The exact submitted state is on the `hydra` branch.

## Run

All scripts are run from the repository root.

| Step | Command | Output |
|---|---|---|
| 0. Select glaciers | `python experiments/central_europe/Select_RGI_ID.py` | `data/raw/central_europe/Split_Files/RGI_SELECT_PART_*.csv` (batches of 10) |
| 1. Run FROST (HPC) | `bash experiments/central_europe/start_multiple.sh data/raw/central_europe/Split_Files/RGI_SELECT_PART_1.csv` | `data/results/central_europe_submit/<rgi_id>/` |
| 2. Reference data | see `experiments/validation/README.md` | `experiments/validation/tables/combined_ela_gradients.csv` |
| 3. Collect | `python experiments/central_europe/collect_results.py` | `tables/aggregated_results.csv` |
| 4. Plot | `python experiments/central_europe/Scatterplot_map.py` | `plots/ALPS_*` maps (Fig. 1) |
| | `python experiments/central_europe/evaluation.py` | `plots/SLA_regional_run.png` (Fig. 2), `plots/GLAMOS_regional_run.png` (Fig. 3) |
| | `python experiments/central_europe/correlation_plots.py` | `plots/correlation.png` (Supplement S1) |
| Diagnostics | `python experiments/central_europe/inversion_check/visualise_inversion_results.py` | `inversion_check/plots/inversion_results.png` |

Step 1 runs once per split file. The results folder is named after `experiment_name` in
`pipeline_config.yaml` (`central_europe_submit`).

Every plotting script accepts `--pdf` (PDF instead of PNG) and `--dark` (dark theme with
transparent background, for slides); the shared handling is `frost/visualization/plot_style.py`.
On the maps, the colorbars, legend and glacier labels sit on the DEM and stay white in both
themes. `Scatterplot_map.py` needs `cartopy`, which is not in the `igm32-frost` environment.

## Inputs

- `data/raw/RGI2000-v7.0-G-11_central_europe/` (RGI7 attributes)
- `data/raw/central_europe/Alps_Glacier_EoS_SLA_2000-2019_stats_v2.csv` (end-of-summer snowlines)
- `data/raw/visualization_context/` (DEM and country outlines for the maps)
- `experiments/validation/tables/combined_ela_gradients.csv` (GLAMOS + WGMS reference)

## Outputs

- `data/results/central_europe_submit/<rgi_id>/`: FROST results per glacier
- `tables/`, `plots/`, `inversion_check/plots/` in this folder

`tables/aggregated_results.csv` is the one table all plots read. `collect_results.py` builds it from:

- per-glacier calibration results (`calibration_results.json`, `observations.nc`),
- ensemble statistics, cached in `tables/ensemble_stats.csv`,
- inversion statistics (velocity and thickness vs observations), cached in `tables/inversion_results.csv`,
- the end-of-summer snowlines and the GLAMOS + WGMS reference (see Inputs).

The two caches are only rebuilt when `RECOMPUTE_ENSEMBLE` / `RECOMPUTE_INVERSION` are set to `True`
in `collect_results.py`. Reference values appear in the aggregated table with the suffix `_wgmsglamos`.

## Files

| File | Purpose |
|---|---|
| `pipeline_config.yaml`, `params_inversion.yaml` | FROST setup of the regional run |
| `Select_RGI_ID.py`, `start_multiple.sh` | Glacier selection and job submission |
| `collect_results.py` | Aggregated results table |
| `Scatterplot_map.py`, `evaluation.py`, `correlation_plots.py` | Paper figures |
| `inversion_check/` | Inversion diagnostics |
