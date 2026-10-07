# ContinuIX: status (2026-10-07, EXP02 done)

FROST contribution to ContinuIX WP2/WP3
(https://github.com/ContinuIX/ContinuIX-1), SMB-gradient approach.
Submission deadline: **end of October 2026** (extended from 1 October);
goal: finish in the week of 5 October.

## Setup

- Data: Zenodo 10.5281/zenodo.21401808, unzipped in `data/raw/continuix`
  (EXP01-EXP20).
- Glaciers: G01 (ice cap, 100 m grid), G02-G06, S01 (25 m), S02.
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
  (`tasks_exp01.txt`: EXP01; `tasks_exp03-15.txt`: the mandatory glaciers
  G01, G05, S01, S02 of EXP03-15; `tasks_smoke.txt`: one glacier per
  EXP03-20; `tasks_all.txt`: old list incl. optional glaciers, no G01). About 10-15 min per glacier on one a40, G01
  about 37 min.

Why the provided THK: with the SIA thickness the THK experiments
(EXP03-07, EXP17-19) would all be identical to EXP01. With the default
sliding, the provided THK gives velocities off by factor 0.8-4
(`thk_forward.py`); inverting tau_ref fixes that (G03 speed RMS 43 -> 6.6
m/yr, G05 50 -> 12 m/yr).

## EXP01 rerun with ES-MDA (2026-10-05)

Array job 4460440, current pipeline (ES-MDA, band-mean dh/dt, IGM 3.2.0,
lam 1e11 fixed). The 28 Sep EnKF results are kept in
`data/results/continuix/EXP01_enkf_0928` (and `submission/EXP01_enkf_0928`).
The inversion output is byte-identical to the 28 Sep run.

| Glacier | Obs. dh/dt | Model dh/dt (spread) | chi2/n | Parameters |
|---|---|---|---|---|
| G02 | +1.56 | +0.31 (0.06) | 65 | ELA 5180, abl 11.6, acc 2.0 |
| G03 | -0.86 | -2.53 (0.01) | 1197 | ELA 3471, abl 3.9, acc 32: broken |
| G04 | -1.28 | -1.30 (0.12) | 11 | ELA 4489, abl 5.7, acc 0.4: good fit |
| G05 | -1.95 | -0.50 (0.04) | 279 | ELA 2683, abl 17.1, acc 1.8 |
| G06 | -0.94 | -0.83 (0.07) | 93 | ELA 3161, abl 7.0, acc 3.2 |
| S01 | 0.00 | -0.27 (0.04) | 20 | ELA 2264 (true 2350), abl 12.1, acc 4.4 |
| S02 | -0.42 | +56 | 2e8 | ELA 13757, negative gradients: diverged |

Worse than the EnKF run except G04; all ensembles collapse (posterior
std 0.3-5 % of the prior), so UNCT_SMB would be meaningless. Causes:

1. The band covariance in `observation_provider.py` holds only the dh/dt
   measurement error (G03: 0.07 m/yr per pixel from the file attribute,
   others default 0.2), averaged over each band to a few cm/yr. No model
   error term, so ES-MDA overfits a 3-parameter SMB.
2. S02 `UNCT_DHDT` goes down to 1.4e-5 m/yr (197 pixels): near-infinite
   weight on one band.
3. No bounds on the SMB parameters (negative gradients, ELA above the
   summit).

ContinuIX gives no per-pixel dh/dt uncertainty for G01-G06, only an
`uncertainty` attribute (G03 0.07 m/yr; G02 "+- 2 m"; others unknown or
guessed). Only S01 (0.2) and S02 have a `UNCT_DHDT` field.

Fix (2026-10-06): `model_error: 0.5` (m/yr) in `config.yml`, added as
model_error^2 I to the band covariance (`ObservationProvider`), so every
band now has about 0.5 m/yr (S02 up to 1.1). Not done yet: dhdt_err floor
(irrelevant now) and bounds on the SMB parameters. The 5 Oct ES-MDA
results are kept in `EXP01_esmda_1005` (and `submission/`). Rerun EXP01
(calibrate, submit) and evaluate with `evaluate_exp01.py`.

## EXP01 with model_error 0.5 (2026-10-06)

Kept in `EXP01_sigma05_1006` (and `submission/`). Posterior forward runs:

| Glacier | Obs. dh/dt | Model dh/dt (spread) | chi2/n | Band RMS | post/prior std |
|---|---|---|---|---|---|
| G02 | +1.56 | +1.46 (0.19) | 1.6 | 0.65 | 14-47 % |
| G03 | -0.86 | -1.54 (0.10) | 6.5 | 1.39 | 3-27 % |
| G04 | -1.28 | -1.18 (0.19) | 0.15 | 0.22 | 18-41 % |
| G05 | -1.95 | -2.20 (0.07) | 8.6 | 1.03 | 3-11 % |
| G06 | -0.94 | -1.18 (0.16) | 2.3 | 0.67 | 6-46 % |
| S01 | 0.00 | -0.05 (0.13) | 0.69 | 0.39 | 8-33 % |
| S02 | -0.42 | -0.45 (0.16) | 0.55 | 0.28 | 9-14 % |

S01 parameters: ELA 2260 +- 41 (true 2350), abl 13.4 +- 2.6 (10), acc
6.1 +- 1.1 (10, capped at 5 m/yr). G03, G05, G06: the upper half thins
too fast (-0.4 to -1.3 m/yr band misfit), the lower half fits.

Inversion check (same day): the misfit is not limited by lam. tau_ref is
smooth (0.05-0.09 decades at ~100 m), never at its bounds. G06 is within
its velocity noise (UNCT_VX ~14 m/yr); S01 lost its thin margins at 50 m
(18 % of cells < 10 m ice, 54 % of the misfit) and its ICEMASK covers 1 km
of ice-free domain past the terminus (also the cause of the -1.65 m/yr
SMB in the submission). The inversion weights all velocities with std 1
m/yr; IGM 3.2 field_inversion only takes a scalar std.

## EXP02 raw data (2026-10-07, done)

Decided to include EXP02 (G01-G06, S02; there is no S01).
`frost/preprocess/continuix_raw.py` (step `raw` of `run_continuix.py`)
turns the GeoTIFFs and shapefiles into `EXP02_<G>.nc` on the EXP01 grid of
the glacier (`exp02_dir`); the submission is on that grid too. Raw files
are in the EXP01 CRS, nothing is reprojected.
- DHDT: full period; G01 2013-2020 and 2020-2023 weighted 7 : 2.92;
  median resampling where the raw grid is finer (2-4 m outliers).
  Glacier means match EXP01 (G02 +1.62/+1.56, G03 -0.88/-0.86,
  G04 -1.30/-1.28).
- DEM: G02 DEM_2013 (DEM_2006 is 0 off the glacier), else the start DEM.
- VX/VY: mean of the years; G06 V-FLAGGED 0 (unreliable) removed. G04's
  V-FLAGGED is 1 exactly where the gappy VX is already missing (opposite
  convention), not used.
- THK: raster for G01 (2013; tagged "Geoid height" but is thickness,
  r 0.999 with EXP01), G04, G06. GPR points only for G02, G03, G05 (101k
  points), S02 (62 cells on the trunk): points averaged per cell
  (THKOBS); THK = IGM's start thickness (`initial_thickness`: SIA from
  velocity and slope, blended with a distance shape) times the
  interpolated ratio THKOBS / start thickness. Then all glaciers as EXP01
  (THK fixed, tau_ref inverted). Against the EXP01 THK: G02 r 0.69 (51 vs
  35 m), G03 0.86 (145/135), G05 0.94 (156/146), S02 0.90 (169/179).
  Tried and dropped: (1) joint inversion of thk and tau_ref with a
  thkobs misfit from the SIA start hardly moved (G03 at the GPR cells
  82 -> 94 m in 500 iterations, obs 285 m); (2) sqrt(distance to the
  outline) as the shape: S02 2.6 times too thick, ice flow 35 times too
  fast (mean 1716 m/yr), calibration diverged (ELA 14700 m).
- ICEMASK: outline at the start of the dh/dt period, used as is
  (attribute `icemask_source`).
Final jobs: 4485043 (G01, G06), 4485208 (G04), 4486238 (G02, G03, G05,
S02 with the IGM-shape THK). In `GROUP_FAU2/` (67 files, 21.9 GPU hours).

| Glacier | Obs. dh/dt | Model (EXP01) | Band RMS (EXP01) | SMB (EXP01) | ELA (EXP01) |
|---|---|---|---|---|---|
| G01 | -0.76 | -0.92 (-0.75) | 0.43 (0.29) | -1.05 (-0.87) | 1378 (1334) |
| G02 | +1.61 | +1.36 (+1.46) | 0.69 (0.65) | +1.41 (+1.46) | 5111 (5083) |
| G03 | -0.87 | -2.32 (-1.55) | 2.94 (1.38) | -2.03 (-1.26) | 3204 (3116) |
| G04 | -1.29 | -1.16 (-1.18) | 0.24 (0.22) | -1.21 (-1.21) | 4463 (4465) |
| G05 | -1.94 | -2.00 (-2.00) | 1.12 (1.04) | -1.99 (-2.08) | 3359 (3123) |
| G06 | -0.95 | -1.25 (-1.19) | 0.64 (0.67) | -1.26 (-1.22) | 3223 (3215) |
| S02 | -0.42 | -0.57 (-0.45) | 0.86 (0.28) | -0.39 (-0.23) | 3249 (2829) |

Velocity fit as EXP01 except G04 (r 0.40 vs 0.76) and G06 (0.40 vs
0.62): gappy raw velocities. G03 fits worse than in EXP01 (noted in the
README).

G03 analysis (2026-10-07): within a 100 m band the observed dh/dt varies
by only 0.2-0.5 m/yr, so the misfit is along the elevation profile, not
between basins. The SMB the data need (obs dh/dt + model FDIV) is flat at
~-4 m/yr from 2300 to 2600 m and then rises by ~11 m/yr per km to +1.9 at
3100 m: no ELA with two gradients fits both, the upper half thins too
fast. Above 3100 m the model slid 33 % too fast (lam 1e11). Test with
tau_ref lam 1e10 for G03 (EXP01, EXP02): upper velocities 1.33 -> 1.09,
r 0.97 -> 0.99, but the dh/dt fit did not improve (EXP01 band RMS 1.38 ->
1.41, mean -1.55 -> -1.63; EXP02 2.94 -> 2.52, mean -2.32 -> -2.45): the
worst bands (2800-3100 m) already had the right velocities, their FDIV did
not change. Reverted to lam 1e11 (same for all glaciers). Remaining
option: a more flexible SMB profile (more parameters, all glaciers).

## EXP03-15, mandatory glaciers (2026-10-06)

All 52 tasks (G01, G05, S01, S02) in `data/results/continuix/EXP03..15`
and `submission/`. Every file covers exactly the ice mask.

Ice mask (`continuix.experiment_icemask`): ICEMASK cells with thickness,
velocity or dh/dt data (ContinuIX fills gaps with 0, so no single field
marks ice-free cells); experiments on the EXP01 grid (EXP03-13) use the
EXP01 mask of the glacier, as they perturb the data, not the extent.
EXP05 adds thickness noise of +-16 % of the mean also past S01's
terminus, which put the ice-free end back into the mask (SMB -1.34).
Rerun with the EXP01 mask (job 4475605): EXP05 S01 SMB -0.18, EXP08 G05
and G01 unchanged (-2.09, -0.86).

Glacier-mean SMB minus EXP01 (m/yr):

| | G01 | G05 | S01 | S02 |
|---|---|---|---|---|
| EXP01 | -0.87 | -2.08 | -0.26 | -0.23 |
| EXP03 THK +-30 % local | +0.04 | +0.04 | +0.28 | -0.16 |
| EXP04 THK +-30 %, 2x corr. | +0.02 | 0.00 | +0.22 | +0.06 |
| EXP05 THK +-16 % of mean | +0.02 | -0.01 | +0.08 | -0.06 |
| EXP06 THK x 1.3 | +0.01 | -0.19 | +0.07 | -0.04 |
| EXP07 THK x 0.7 | +0.04 | +0.07 | +0.07 | -0.05 |
| EXP08-12 VEL random | <= 0.01 | <= 0.02 | <= 0.03 | <= 0.04 |
| EXP13 VEL bias ramp | +0.01 | -0.01 | -0.03 | +0.03 |
| EXP14 RES 3x coarser | -0.02 | -0.04 | 0.00 | -0.02 (300 m) |
| EXP15 RES 100 m | -0.04 | -0.01 (100 m) | -0.35 (100 m) | 0.00 |

dh/dt fits within +-0.13 m/yr (G05 EXP06 -0.24), chi2/n 0.2-2.3. The
thickness experiments move the parameters (G05 EXP03/04 ELA +190/+170 m,
S02 EXP03 +250 m, G01 EXP06 acc +5.3), not the glacier-mean SMB, which
the dh/dt pins down (glacier-mean FDIV ~ 0). Velocity noise is smoothed
out by the tau_ref regularisation. S01 at 100 m (EXP15) has 219 cells,
its margins average poorly (-0.35). S01 EXP03/04 (+0.28/+0.22) come from
a lower ablation gradient, same mask as EXP01.

GLOB (EXP16-20) set aside; smoke test (one glacier each) ran: Hugonnet
dh/dt 2000-2020 with UNCT_DHDT 1.5-3.7 m/yr per pixel barely constrains
the SMB (posterior 60-100 % of the prior, UNCT_SMB 0.8-1.8 m/yr). EXP16
G02 had a member with ELA above the summit (ice gone after 20 years):
legitimate, no parameter bounds (they would tie FROST to this SMB model).

EXP02 (raw GeoTIFFs, shapefiles, GPR-only thickness for G02, G03, G05)
is optional by the participation rules (README "Who can participate");
needs its own preprocessing and a thickness inversion, ~1-2 days.
Open question to the organisers: WP3 asks for "at least 4 real-world
cases", the experiment table names 2 real + 2 synthetic.

## EXP01 final (2026-10-06)

`data/results/continuix/EXP01` (and `submission/EXP01`): full rerun of all
glaciers with the new mask and the DEM gap filling in `write_submission`
(G02, G04, G06 had NaN SMB under DEM gaps). Every submission file has no
NaN inside the ice mask and no values outside it.

| Glacier | Obs. dh/dt | Model dh/dt | chi2/n | SMB (UNCT) |
|---|---|---|---|---|
| G01 | -0.71 | -0.75 | 0.67 | -0.87 (0.23) |
| G02 | +1.56 | +1.46 | 1.58 | +1.46 (0.25) |
| G03 | -0.86 | -1.55 | 0.99 | -1.26 (0.41) |
| G04 | -1.28 | -1.18 | 0.15 | -1.21 (0.28) |
| G05 | -1.95 | -2.00 | 1.06 | -2.08 (0.39) |
| G06 | -0.94 | -1.19 | 2.30 | -1.22 (0.21) |
| S01 | 0.00 | -0.11 | 1.45 | -0.26 (0.30) |
| S02 | -0.42 | -0.45 | 0.55 | -0.23 (0.26) |

Only G03 moved against the run below (-1.41 -> -1.55, within its spread
0.31): its input now uses the new mask too. The run below is kept in
`EXP01_accepted_1006`.

G01 (Langjokull, 832 km2, 11 basins), 100 m, 3-parameter ELA SMB:
inversion 5 min, calibration 30 min (17 min setup, mostly the pixel-pair
band covariance of 83k pixels; then ~2 min per iteration). ELA 1334 +- 31,
abl 9.2 +- 1.0, acc 8.3 +- 1.0. Velocity -10 %, RMS 7 m/yr, r 0.96. The
elevation profile fits within ~0.1 m/yr from 600 to 1400 m; at the dome
the model thickens at 1500 m (+0.55) and thins at 1700 m (-0.80).
Per basin (model - obs.): west and north-west too positive (8: +1.09,
11: +0.37, 9: +0.22), south and east too negative (10: -0.50, 7: -0.47,
5: -0.44). One ELA cannot hold this; an ELA with a horizontal trend
(ELA0 + a x + b y, 5 parameters) would. Basin-mean dh/dt is close to the
basin-mean SMB (the flux divergence only moves ice across the divides),
so this is SMB, not flow.

## EXP01 accepted (2026-10-06)

`data/results/continuix/EXP01` (and `submission/EXP01`). On top of the
run above (config `glaciers:` overrides, `continuix._icemask`):
- ice mask = ICEMASK cells with THK, velocity or dh/dt data (removes 33 %
  of S01, <= 0.3 % elsewhere), in input and submission;
- S01 on a 25 m grid (all steps);
- G03 model_error 1.3, G05 1.5 (0.5 * sqrt(chi2/n); calibrate, submit).

| Glacier | Obs. dh/dt | Model dh/dt (spread) | chi2/n | post/prior std |
|---|---|---|---|---|
| G03 | -0.86 | -1.41 (0.27) | 0.96 | 11-52 % |
| G05 | -1.95 | -2.00 (0.19) | 1.06 | 8-29 % |
| S01 | 0.00 | -0.11 (0.17) | 1.45 | 9-26 % |

G02, G04, G06, S02 as in the table above. chi2/n now 0.15-2.3 overall.
G03 upper half still thins ~1.2 m/yr too fast (structural).
S01: submission SMB mean -0.27 m/yr (was -2.34 on the full ICEMASK);
ELA 2290 +- 40 (true 2350), abl 12.1 +- 1.7 (10), acc 6.4 +- 0.9. The
velocity fit did not improve at 25 m (-29 %, r 0.64). tau_ref is about as
flat as at 50 m (p5-p95 0.15 vs 0.23 decades). lam is the same for all
glaciers (1e11) and does not depend on the grid: IGM's squared_laplacian
uses the Laplacian in physical units (/dx^2), averaged over the ice area,
so the same lam means the same smoothness in metres on any grid.

Decision (2026-10-06): velocity misfit std 1 m/yr and lam 1e11 for all
glaciers; the ContinuIX velocity uncertainty is not used. The provided
"uncertainties" of G01, G02, G05, G06 are standard deviations over the
period (G05 median 44 m/yr at 27 m/yr speed), not errors of the mean
field. Tested on G02-G06 with the median as std (inversion only): it
acts as a per-glacier lam, smoothing tau_ref and worsening the velocity
fit on four of five glaciers (G06 r 0.62 -> 0.23, G05 RMS 12 -> 17 m/yr);
only G04 (std 0.3) improved slightly.

Later: light smoothing of THK for the flux divergence.

Inversion (velocity fit, unchanged): good on S02, G03, G05; OK G02; weak
G04 (speeds near noise); poor G06 (-31 % speed, r 0.61) and S01 (-29 %,
cost only to 0.93). tau_ref never at its bounds; lowering lam would not
help (see the inversion check below).
Decision so far: keep the provided THK (needed for the THK experiments),
invert tau_ref only.

S01 truth (ContinuIX-1/Synthetic_Glacier/domain_config.py): ELA 2350 m,
10 m/yr per km both sides, accumulation capped at 5 m/yr.

## EXP01 results (EnKF, 2026-09-28)

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

All mandatory experiments and EXP02 are done. Known limitations (in the
README): G03 upper half thins ~1.2 m/yr too fast (worse in EXP02); G01
per-basin pattern (one ELA); S01 velocities -29 %; EXP02 thickness away
from the GPR profiles is an estimate.

1. Submission package (2026-10-07): `package_submission.py` writes
   `data/results/continuix/GROUP_FAU2/` (67 result files of
   `tasks_exp01.txt`, `tasks_exp02.txt`, `tasks_exp03-15.txt` with the
   `description` attribute, `log_GROUP_FAU2.txt` from `timings.json`,
   README, filled checklist, ContinuIX instruction files; 21.9 GPU
   hours). README and checklist are in `experiments/continuix/submission/`.
   Group FAU2, contributor Oskar Herrmann (ORCID 0000-0002-0319-9065), JRG
   Glacier Systems & Natural Hazards, Institute of Geography, FAU;
   Johannes Fürst acknowledged as FROST co-developer.
2. Decided (2026-10-07): group GROUP_FAU2; Johannes is acknowledged, not
   a contributor. Then upload `GROUP_FAU2/` to the ContinuIX SharePoint.
3. GLOB EXP16-20 for the 6 real glaciers (mandatory by the experiment
   table): `tasks_glob.txt` (30 runs; the glob_* files already hold the
   protocol's combination, unchanged fields are the site data). 20-year
   forward runs: G01 ~70 min, submit with --time=03:00:00. Then add
   `tasks_glob.txt` to `TASK_LISTS` and describe GLOB in the README.
4. Optional, if time: ELA with a horizontal trend for G01 (5 parameters).

## Notes

- The Monitor plots on compute nodes have no text (font missing there).
- Reading S01 (5000x1001 at 1 m) takes about 6 min on the login node,
  seconds on the compute nodes.
