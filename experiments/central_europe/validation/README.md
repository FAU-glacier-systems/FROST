# Validation reference: GLAMOS + WGMS

Builds the in-situ reference table (ELA, ablation and accumulation gradient, annual mass
balance) that `../collect_results.py` merges into the calibration results and
`../evaluation.py` plots against.

Run from this folder, in this order:

```bash
python extract_glamos_ela_gradients.py   # -> tables/GLAMOS_analysis_results.csv, plots/all_gradients.png
python process_wgms.py                   # -> tables/wgms_ELA_gradients.csv, plots/glacier_smb/<glacier>_fit.png
python merge_glamos_wgms.py              # -> tables/combined_ela_gradients.csv
```

All three accept `--dark` (dark theme, transparent background) and `--pdf` (PDF instead of PNG).

Inputs are read from `../../../data/raw/glamos/` and `../../../data/raw/DOI-WGMS-FoG-2025-02b/`.
`tables/RGI6-7.csv` is a hand-made lookup from WGMS glacier names to RGI7 IDs and GLAMOS names.

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
