# Third-party data in this repository

Only these small reference files are part of the repository; everything else
under `data/` is downloaded by the pipeline and not tracked. They are used to
evaluate FROST, never to calibrate it.

| File | Source | Terms |
|---|---|---|
| `glamos/massbalance_observation.csv`, `glamos/massbalance_observation_elevationbins.csv` | GLAMOS, Swiss Glacier Mass Balance, release 2023 | CC BY 4.0 (GLAMOS publication and data policy); cite as below |
| `glamos/FoG_MB_473.csv` | WGMS Fluctuations of Glaciers database (Rhonegletscher, minimal fields) | Open access with correct citation; cite as below and/or the original sources in its INVESTIGATOR, SPONS_AGENCY and REFERENCE columns |
| `RGI2000-v7.0-G-11_central_europe/..._attributes.csv` | Randolph Glacier Inventory 7.0, region 11 attributes | CC BY 4.0; cite as below |
| `glamos/GLAMOS_RGI.csv` | FROST: lookup of GLAMOS names to RGI 7.0 IDs | GPL-3.0, as the code |

The files are unchanged copies, except that the GLAMOS and WGMS files keep
their own header with the citation.

## Citations

- GLAMOS (2023). Swiss Glacier Mass Balance, release 2023, Glacier Monitoring
  Switzerland, doi:10.18750/massbalance.2023.r2023.
  Data source: Glacier Monitoring Switzerland (GLAMOS).
- WGMS (2024). Fluctuations of Glaciers Database. World Glacier Monitoring
  Service (WGMS), Zurich, Switzerland. doi:10.5904/wgms-fog-2024-01.
- RGI 7.0 Consortium (2023). Randolph Glacier Inventory - A Dataset of Global
  Glacier Outlines, Version 7.0. NSIDC, doi:10.5067/f6jmovy5navz.

The WGMS database used in the evaluation scripts (`DOI-WGMS-FoG-2025-02b`) is
not part of the repository; download it from https://wgms.ch/data_databaseversions/.
