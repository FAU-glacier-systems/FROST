# Validation against in-situ mass balance (GLAMOS, WGMS)

Everything that compares FROST with glaciological measurements. The main product
is the reference table of ELA, ablation and accumulation gradient and annual mass
balance per glacier that `central_europe/collect_results.py` merges into the
calibration results and `central_europe/evaluation.py` plots against. The other
scripts look at single glaciers and at the WGMS data themselves.

## Run

All scripts are run from the repository root.

Reference table, in this order:

```bash
python experiments/validation/extract_glamos_ela_gradients.py  # -> tables/GLAMOS_analysis_results.csv, plots/all_gradients.png
python experiments/validation/process_wgms.py                  # -> tables/wgms_ELA_gradients.csv, plots/glacier_smb/<glacier>_fit.png
python experiments/validation/merge_glamos_wgms.py             # -> tables/combined_ela_gradients.csv
```

These three accept `--dark` (dark theme, transparent background) and `--pdf` (PDF instead of PNG).

Further comparisons:

```bash
python experiments/validation/compare_single_glacier.py -p experiments/validation/params/rhone.json
python experiments/validation/compare_ti_model_to_glamos.py
python experiments/validation/plot_mb_per_year.py
python experiments/validation/plot_stake_positions.py
python experiments/validation/analysis_horizontal_spread_glamos.py
```

## Inputs

- `data/raw/glamos/` (GLAMOS mass balance, `GLAMOS_RGI.csv` lookup)
- `data/raw/DOI-WGMS-FoG-2025-02b/` (WGMS Fluctuations of Glaciers)
- `data/raw/RGI2000-v7.0-G-11_central_europe/` (RGI7 attributes)
- `tables/RGI6-7.csv`: hand-made lookup from WGMS glacier names to RGI7 IDs and GLAMOS names
- `compare_single_glacier.py`: a FROST result, set in `params/<glacier>.json`
  (paths relative to the repository root)
- `compare_ti_model_to_glamos.py`: specific mass balance of the temperature-index
  model runs, `tables/ti_model/MB_compare_output_v06.csv`

## Outputs

All in this folder:

- `tables/`: reference tables (`combined_ela_gradients.csv` is the one used downstream)
- `plots/`: reference and comparison plots; `plots/glacier_smb/` profile fits per
  WGMS glacier; `plots/wgms/` WGMS data plots; `plots/rhone/` single-glacier comparison

## Files

| File | Purpose |
|---|---|
| `extract_glamos_ela_gradients.py` | ELA and gradients from GLAMOS |
| `process_wgms.py` | ELA and gradients from WGMS |
| `merge_glamos_wgms.py` | Combined reference table, GLAMOS vs WGMS comparison |
| `compare_single_glacier.py`, `params/` | Specific mass balance of one FROST run vs GLAMOS and geodetic |
| `compare_ti_model_to_glamos.py` | Temperature-index model mass balance vs the reference |
| `plot_mb_per_year.py` | WGMS mass-balance profiles per glacier and year |
| `plot_stake_positions.py`, `analysis_horizontal_spread_glamos.py` | WGMS stake data on Aletsch |

## Combined table

`tables/combined_ela_gradients.csv` has one row per glacier. Where both sources cover a glacier,
GLAMOS takes precedence; the `*_source` columns record which source each value came from.

## Note: how the ELAs and gradients are derived

The two sources are not derived identically.

**GLAMOS** (`extract_glamos_ela_gradients.py`)
- ELA: the ELA reported by GLAMOS for each year, averaged over 2000–2019.
- Gradients: fitted per year through that year's ELA, then averaged.

**WGMS** (`process_wgms.py`) produces two different ELAs per glacier, both kept in
`tables/wgms_ELA_gradients.csv`:
- `ela_mean`: the ELA reported by WGMS (`mass_balance.csv`), averaged over 2000–2019.
  **This is the one used in the combined table.**
- `ELA_m`: the zero crossing of the mean mass-balance profile over all years (`mass_balance_band.csv`).
  If the profile never crosses zero, a linear fit is used instead (the script prints which glaciers).
- Gradients: fitted once to the mean profile, through `ELA_m` (not through the reported `ela_mean`).
  The accumulation gradient is only fitted when more than 3 bands lie above `ELA_m`.

So for WGMS glaciers, the ELA in the combined table and the ELA the gradients were fitted
around can differ. `plots/wgms_ela_reported_vs_fitted.png` compares the two.

`compare_single_glacier.py` computes the modelled mass balance on the surface of the
inversion (start of the period); it used the surface at the end of a separate final run
before, which FROST no longer writes.
