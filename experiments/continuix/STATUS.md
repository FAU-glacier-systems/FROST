# ContinuIX: status (2026-09-28)

FROST contribution to ContinuIX WP2/WP3
(https://github.com/ContinuIX/ContinuIX-1), SMB-gradient approach.
Submission deadline: **1 October 2026**.

## Setup

- Data: Zenodo 10.5281/zenodo.21401808, unzipped in `data/raw/continuix`
  (EXP01-EXP20).
- Glaciers: all single glaciers (G02-G06, S01, S02). The ice cap G01 is
  skipped for now.
- Pipeline per experiment and glacier (`run_continuix.py`):
  1. `prepare`: ContinuIX netCDF -> `input.nc` on a 50 m grid (coarser
     ContinuIX grids kept; block average; zeros inside the ice are gaps).
  2. `inversion`: provided THK fixed, sliding `tau_ref` inverted from the
     surface velocities (`params_inversion_tau.yaml`, log10 transform,
     lam 1e11 from the L-curve in `tau_sweep.py` on G03/G05).
  3. `calibrate`: EnKF on ELA, ablation and accumulation gradient against
     the ContinuIX DHDT over its timestamp period (synthetics: 2000-2010).
  4. `posterior`: forward runs of the final ensemble.
  5. `submit`: `EXP##_G##_method01.nc` on the original grid with SMB,
     UNCT_SMB (ensemble spread), FDIV, UNCT_FDIV, DENSITY (910 kg m-3).
- Run: `sbatch --array=1-N experiments/continuix/run_continuix.sh <tasks>`
  (`tasks_exp01.txt`: EXP01; `tasks_all.txt`: 123 tasks, EXP01, EXP03-20).
  About 10-15 min per glacier on one a40.

Why the provided THK: with the SIA thickness the THK experiments
(EXP03-07, EXP17-19) would all be identical to EXP01. With the default
sliding, the provided THK gives velocities off by factor 0.8-4
(`thk_forward.py`); inverting tau_ref fixes that (G03 speed RMS 43 -> 6.6
m/yr, G05 50 -> 12 m/yr).

## EXP01 results

Glacier means in m/yr ice equivalent.

| Glacier | Period | Obs. dh/dt | Model dh/dt | SMB | FDIV | ELA (m) |
|---|---|---|---|---|---|---|
| G02 | 2006-2013 | +1.56 | +0.82 | +0.83 | 0.01 | 5132 |
| G03 | 2012-2021 | -0.85 | -2.03 | -1.80 | 0.19 | 3066 |
| G04 | 2021-2025 | -1.27 | -0.82 | -0.89 | 0.00 | 4378 |
| G05 | 2018-2024 | -1.94 | -2.06 | -2.16 | 0.01 | 3113 |
| G06 | 2006-2017 | -0.94 | -0.97 | -0.99 | 0.03 | 3190 |
| S01 | synthetic | 0.00 | +0.12 | -1.65 | -0.02 | 2230 |
| S02 | synthetic | -0.42 | -0.56 | -0.34 | 0.16 | 2838 |

G04 needed `min_velocity_p99: 1` (slow glacier, 99th percentile 8.7 m/yr).

## Open

1. **G03**: calibrated model thins at -2.03 m/yr vs -0.85 observed, with
   a narrow ensemble. Compare per elevation band whether the SMB shape
   (too little accumulation high up) or the ice flow is the cause.
   Suspect: the observation error in
   `observation_provider.compute_bin_variance` adds the elevation spread
   within each 50 m band (~14 m, 1 sigma), as large as the whole signal
   of a 9-year period.
2. **S01**: model dh/dt fits, but the SMB written on the 1 m grid averages
   -1.65 m/yr. The SMB is evaluated on thin margin cells that the 50 m
   model barely resolves, so SMB and the interpolated FDIV don't match
   there.
3. **G02, G04**: reach about half (G02 thickening) and two thirds (G04)
   of the observed dh/dt.
4. Run EXP03-20 (`tasks_all.txt`) once 1-2 are settled.
5. Submission documents: `README_<GROUP>_method01.txt` (method,
   pre-processing, uncertainty), `log_<GROUP>.txt` (compute times from
   `data/results/continuix/*/*/timings.json`), filled
   `SUBMISSION_CHECKLIST.txt`.
6. Needed from Oskar: group shorthand, contributors with ORCID, whether
   to do the optional EXP02 (raw GeoTIFFs) and G01 (ice cap).

## Notes

- The Monitor plots on compute nodes have no text (font missing there).
- Reading S01 (5000x1001 at 1 m) takes about 6 min on the login node,
  seconds on the compute nodes.
