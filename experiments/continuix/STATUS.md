# ContinuIX: status (2026-10-08, evening: repackage, then upload)

FROST contribution to ContinuIX WP2/WP3
(https://github.com/ContinuIX/ContinuIX-1), SMB-gradient approach.
Submission deadline: **end of October 2026**.

## Open (start here)

State at the end of 2026-10-08:
- Final grids: 50 m for G02-G06 and S01 (S01 rerun today, was 25 m);
  100 m for G01 (tested at 50 m: worse fit, calibrations 4 h to > 1 day)
  and S02 (its data grid). Reason: IGM's emulator dahunet_mini was
  pretrained on 50-250 m grids (S. Rosier, pers. comm.), so 50 m is the
  finest grid in its range; every finer grid tested worse. lam 1e11 kept.
- `data/results/continuix/submission/`: 149 files on these grids.
  GROUP_FAU2 and check_GROUP_FAU2.txt are still from 7 Oct (old S01).
- Code (not committed): FFT band covariance, in-process forward workers
  (`forward_workers`), lam_test.py, inversion_test.py, bench_forward.py,
  report_resolution.py, make_summary.py refactor, config.yml (G01 100 m,
  S01 50 m), method README (50 m / G01 100 m, resolution bullet).
- Report: `summary/ContinuIX_FROST_report_2026-10-08.pdf` (4 pages:
  submission summary + model-grid section); pages 1-2 need the new check
  report.

Next (in this order):
1. `sbatch experiments/continuix/package_and_check.sh`.
2. Method README: add "pretrained on 50-250 m grids (S. Rosier, pers.
   comm.)" to the resolution bullet; update the S01-dependent numbers from
   the new check report (perturbation bullet: share of runs within 0.07
   m/yr, S01 EXP03/04, EXP14/15, "S01 at 100 m has only 219 ice cells";
   S01 velocity 29 % -> 28 %); copy the README into GROUP_FAU2.
3. `python experiments/continuix/report_resolution.py` (final report).
4. Commit (git status -sb first: shared checkout), upload GROUP_FAU2.
5. Delete the backups after the upload: `backup_grid_1008/` (S01 25 m),
   `backup_g01_50m_1008/` (G01 50 m), and the test folders `res25/`,
   `res25_inproc/`, `res50/`, `res_native/`, `lam_test/`,
   `inversion_test/`, `bench/` (keep their numbers in this file).

Log of 8 Oct (details):

Every run is done, packaged and checked. GLOB accepted as documented
(decided 2026-10-08). Upload on hold: resolution rerun first.

Resolution problem (found 2026-10-08): the model grid is
max(50 m, data spacing), but the data are finer (G04 2 m, G05/G06 10 m,
G03 20 m, G02 25 m, S01 1 m -> 25 m, G01 25 m -> 100 m). So EXP14 (data
3x coarser) runs on the EXP01 model grid for G01, G04, G05, G06, S01 and
tests only input averaging, not resolution (SMB change <= 0.10 m/yr);
WP3 asks whether resolution matters. Decision: rerun everything at a
finer grid, step by step.
1. Now: 25 m test of EXP01 G02-G06 (`config_res25.yml`, results in
   `data/results/continuix/res25/`, submission untouched):
   `sbatch --array=1-5 --time=06:00:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_res25.txt --config experiments/continuix/config_res25.yml`
   Compare with EXP01 at 50 m: band RMS, chi2/n, glacier-mean SMB,
   velocity fit, run time (sets the cost of step 2).
   Efficiency (2026-10-08, after job 4501711 started, so not in it):
   the band covariance is now an FFT convolution
   (`ObservationProvider.band_covariance`), identical to the pixel-pair
   sum to 3e-16, G01 setup 17 min -> 0.8 s (pixel pairs would have taken
   ~3 days for G01 at 25 m). The forward step now logs
   `--- forward ... s` per round. Benchmark (`bench_forward.py`, G02 at
   25 m, job 4501821): one `igm_run` 18.9 s whether 0 or 7 years
   (TF import alone 8.9 s), 36 at once 55-58 s = the whole round; in
   process after the first call 2.6 s per run. So the forward runs are
   start-up, not simulation, on small glaciers. Change: the ensemble now
   keeps a spawn pool of `forward_workers` processes (default: job cores,
   16) that call IGM's main() in process (`igm_wrapper.run_igm`,
   `in_process=True`); `igm_run` subprocess stays the default for other
   callers (Alps TI). To verify (same results, round time) on copies of
   the 25 m G02/G06 inversions:
   `sbatch --array=1-2 --time=02:00:00 experiments/continuix/run_continuix.sh data/results/continuix/res25_inproc/tasks.txt --config data/results/continuix/res25_inproc/config.yml --steps calibrate,submit`
   Result (job 4501850): final ensembles bit-identical to the subprocess
   runs; rounds G02 55-67 -> 9-10 s, G06 78 -> 27 s (first round +27-40 s
   warm-up); calibrate G02 481 -> 126 s, G06 703 -> 272 s. Adopted.
   G05 at 25 m (job 4501711_4) failed: 36 igm_run at once ran the A40 out
   of memory (16 members RESOURCE_EXHAUSTED), then a missing output.nc
   stopped the calibration. Rerun with the 16 in-process workers:
   `sbatch --array=4 --time=04:00:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_res25.txt --config experiments/continuix/config_res25.yml --steps calibrate,submit`
   25 m test result (EXP01, 8 Oct): no gain. G02, G04 calibrate the
   same; G03 chi2/n 1.0 -> 4.3 (abl 5.7 -> 9.8), G06 model dh/dt -1.18 ->
   -1.35 (obs -0.96). Velocity fit worse at 25 m (G04 r 0.76 -> 0.59 even
   on 2x2 block means, G06 0.62 -> 0.37). Possible inversion causes:
   nbitmax 500 reached with the cost still falling (G03, G05, G06), lam
   1e11 keeps tau_ref as smooth as at 50 m (IGM normalises both terms by
   area), pretrained emulator never retrained (retrain_iter 0).
   Inversion test (`inversion_test.py`, 15 tasks: G04, G05, G06 x
   nbitmax 2000 / + lam 3e10 / + lam 1e10 / retrain 2x at 25 m / retrain
   2x at 50 m), results in `data/results/continuix/inversion_test/`:
   `sbatch --array=1-15 experiments/continuix/inversion_test.sh` (job
   4502373, done 2026-10-08).
   
   Result (velocity r on ice, 2x2 block means for 25 m; full table:
   `python experiments/continuix/inversion_test.py --summary`):

   | | 50 m now | 25 m now | 25 m nbitmax 2000 | + lam 3e10 | + lam 1e10 | 25 m retrain | 50 m retrain |
   |---|---|---|---|---|---|---|---|
   | G04 | 0.76 | 0.59 | 0.59 (converged at 470 already) | 0.69 | 0.78 | 0.61 | 0.67 |
   | G05 | 0.96 | 0.95 | 0.95 | 0.96 | 0.96 | 0.93 | 0.93 |
   | G06 | 0.62 | 0.36 | 0.70 | 0.76 | 0.80 | 0.54 | 0.39 |

   - G06: most of the 25 m loss was non-convergence (500 -> 1686
     iterations: r 0.36 -> 0.70); G06 at 50 m also stopped at 500 with
     the cost still falling. Lower lam adds 0.70 -> 0.80.
   - G04: converged; lower lam alone brings it back (0.59 -> 0.78).
   - Emulator retraining is worse everywhere, at 25 and 50 m: dropped.
   - Not yet shown: (1) whether lam 1e10 also helps at 50 m (lam is
     grid-independent, so it may be "1e10 is better", not "25 m needs
     1e10"; no 50 m lam 1e10 run); (2) whether the better velocity fit
     improves dh/dt: lower lam always fits velocities better (less
     smoothing), and on G03 at 50 m lam 1e10 improved velocities but not
     dh/dt. Next: full runs (inversion + calibrate, nbitmax 2000) of
     G04, G05, G06 with lam 1e10 at both 25 and 50 m, compared with the
     current 50 m and 25 m (lam 1e11) on band RMS, chi2/n, glacier mean.
     The inversion step must be rerun: lam is an inversion setting.
   G05 at 25 m (rerun, job 4502387, in-process workers, no OOM): chi2/n
   1.06 -> 1.42, model dh/dt -1.95 -> -2.00 (obs -1.94), ELA 3123 ->
   3244; calibrate 1510 s (rounds ~205 s). So at lam 1e11 no glacier
   gains from 25 m.
   lam test (`lam_test.py`, set up 8 Oct): full EXP01 runs (inversion
   nbitmax 2000, calibrate, submit) of G04, G05, G06 at 25 and 50 m with
   lam 3e10, 1e10, 3e9 (18 tasks), each scored on posterior band RMS,
   chi2/n, glacier mean and velocity fit (`band_score.json`; the lam 3e10
   tasks also score the lam 1e11 reference of their grid). One lam for
   all glaciers; if the best is at 3e9 (range edge), suspect noise
   fitting and stay near the L-curve bend.
   `sbatch --array=1-18 experiments/continuix/lam_test.sh`, then
   `python experiments/continuix/lam_test.py --summary`.
   lam test result (16 of 18; G05 50 m lam 1e10/3e9 hit 30 min): keep
   lam 1e11. Lower lam fits velocities better but dh/dt worse: G06 band
   RMS 25 m 0.68 -> 1.51 (chi2/n 1.7 -> 8.0), 50 m 0.78 -> 1.44; G05
   25 m 1.80 -> 1.96; G04 only gains at 25 m (0.30 -> 0.20 = its 50 m
   value). Resolution at lam 1e11: 25 m no better (G05 1.80 vs 1.55, G04
   0.30 vs 0.20, G06 draw), G02 same, G03 worse. Decision (8 Oct): no
   25 m rerun; 50 m is the method grid; the 25 m test is a WP3 finding
   for the README.
   Next: G01 (100 m) and S01 (25 m) at 50 m, so all glaciers share the
   grid and EXP14/15 test it (`config_res50.yml`, results in `res50/`;
   G01 with 8 forward workers for GPU memory):
   `sbatch --array=1 --time=04:00:00 experiments/continuix/res50_test.sh` (G01),
   `sbatch --array=2 --time=00:45:00 experiments/continuix/res50_test.sh` (S01),
   then `python experiments/continuix/lam_test.py --summary`-style scores in
   `res50/EXP01/<G>/band_score.json` vs `EXP01/<G>/band_score.json`.
   Switch S01 if it recovers its truth (ELA 2350, gradients 10, acc cap
   5) about as well as at 25 m (ELA 2290 +- 40, abl 12.1, acc 6.4);
   switch G01 if band RMS and velocity fit are no worse and all its runs
   (EXP01-20) fit in the time left. S02 stays at its 100 m data grid.
   Decision (8 Oct): grid rule stays data-driven (model grid = max(50 m,
   data grid)) in every experiment, so FROST uses the same EXP14/EXP15
   input as the other methods; no experiment-specific grids (rejected:
   EXP14 at 3x the model grid, S02 at 50 m). README must then state:
   EXP15 coarsens the model grid for all but S02 (EXP15 data = EXP01
   data); EXP14 coarsens it for G01 (at 50 m), G02, G03, S02, while for
   G04, G05, G06, S01 the 3x data are still finer than 50 m, so EXP14
   tests input detail below the grid (<= 0.10 m/yr); 25 m test: no gain,
   so 50 m is the method grid.
   Curiosity (not for the submission): EXP01 of G03 (20 m), G04 (2 m),
   G05 (10 m), G06 (10 m) on the data grid (`config_native.yml`, results
   in `res_native/`; forward workers G04 4, G05 3, G06 8 for GPU memory;
   commands in `native_test.sh`). G01 (25 m, 1.3M cells) and S01 (1 m, 2M
   cells) not feasible. Caveat: the emulator sees ~13 cells, far less than
   an ice thickness at 2-10 m.
   Results (8 Oct, `band_score.json` in each run dir; band RMS =
   posterior band-mean dh/dt vs observed):
   - S01 50 m (res50): band RMS 0.64 -> 0.50, chi2/n 1.45 -> 0.89, ELA
     2271 +- 38 vs 2289 +- 40 at 25 m (truth 2350), abl 13.2 vs 12.1
     (10), acc 6.1 vs 6.4: as close to the truth, better fit -> switch
     S01 to 50 m (rerun EXP01, EXP03-15).
   - G01 50 m (res50, 333k cells, 8 workers, 50 min): band RMS 0.41 ->
     0.57, area-weighted scatter 0.29 -> 0.44, chi2/n 0.67 -> 1.31, mean
     unchanged (-0.75 / -0.76, obs -0.71), velocity r 0.96 -> 0.95 ->
     keep G01 at 100 m.
   - Native grids: G03 20 m band RMS 2.32 vs 1.30 at 50 m (mean closer,
     -1.36 vs -1.55, obs -0.88, but band scatter 2.16 vs 1.19); G06 10 m
     band RMS 0.60 vs 0.78 and mean -1.10 vs -1.19 (obs -0.94), but the
     inversion failed (velocity r -0.08 vs 0.62), so the dh/dt fit comes
     from SMB compensating a wrong flow; G04 2 m and G05 10 m ran out of
     GPU memory (forward workers / inversion).
   Summary for the README: no grid finer than the method grid improved
   dh/dt and velocities together; G01 fits best at 100 m, S01 at 50 m.
   Decision (8 Oct, revised): G01 also at 50 m. At 100 m neither EXP14
   (75 m data) nor EXP15 (100 m) changes G01's grid, so one of the two
   real mandatory RES glaciers would show no resolution response; at 50 m
   both do (50 -> 75, 50 -> 100 m). Cost: band RMS 0.41 -> 0.57 (chi2/n
   0.67 -> 1.31), glacier mean unchanged; ~15-20 GPU hours.
   Final grids: 50 m G01-G06, S01; S02 at its 100 m data grid
   (`config.yml`: G01/S01 overrides removed, G01 forward_workers 8).
   Old runs (G01 100 m, S01 25 m: 34 run dirs and submission files) moved
   to `data/results/continuix/backup_grid_1008/` (59 GB; delete after the
   upload). Reruns (`tasks_grid50.txt`: 1-14 S01, 15-29 G01 EXP01-15,
   30-33 G01 EXP16-18/20, 34 G01 EXP19):
   `sbatch --array=1-14 --time=00:30:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_grid50.txt`
   `sbatch --array=15-29 --time=01:30:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_grid50.txt`
   `sbatch --array=30-33 --time=03:00:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_grid50.txt`
   `sbatch --array=34 --time=06:00:00 experiments/continuix/run_continuix.sh experiments/continuix/tasks_grid50.txt`
   S01 reruns done (job 4503082, 14 files, model grid 50 m, EXP15 100 m).
   Method README (`submission/README_GROUP_FAU2_method01.txt`) updated
   for 50 m (description, step 2, new "Resolution (EXP14, EXP15)"
   bullet). After the G01 reruns and package_and_check.sh, recheck these
   numbers from `check_GROUP_FAU2.txt` (they come from G01 100 m / S01
   25 m): the [TBD after check] in the resolution bullet (max glacier-mean
   SMB change in EXP14 for G04, G05, G06, S01); the perturbation bullet
   (76 % of 104 runs <= 0.07 m/yr; EXP03/04 S01 +0.28/+0.22; EXP14/15
   changes, "S01 at 100 m has only 219 ice cells ... -0.34"); S01
   velocity "about 29 % lower"; EXP19 G01 "-2.9 m/yr against -0.9";
   FDIV extremes "EXP04 G01 up to several hundred m/yr". Then
   make_summary.py, commit, upload.
   Reverted (8 Oct, evening): G01 back to 100 m. At 50 m six of its 20
   calibrations (EXP04, EXP05 thickness noise; EXP16, 17, 19, 20 GLOB)
   ran 1.5-3 h and were still in their first rounds (prior members with
   tiny time steps); estimated 4 h to > 24 h each, which is not
   reasonable for one calibration. The 20 G01 100 m runs are back in
   EXP*/G01 and submission/; the 50 m G01 runs (14 complete, 6 partial)
   are in `backup_g01_50m_1008/`; `backup_grid_1008/` now holds only the
   S01 25 m runs. Final grids: 50 m G02-G06, S01; 100 m G01 (fit and
   cost), S02 (data). submission/ has 149 files on these grids.
   `config.yml`, method README (description, step 2, resolution bullet)
   and the report updated accordingly; summary_data.json S01 entry from
   the new run (band-mean model dh/dt -0.097; old file kept as
   summary_data_1007.json).
   Next:
   1. `sbatch experiments/continuix/package_and_check.sh` (rebuilds
      GROUP_FAU2 and check_GROUP_FAU2.txt).
   2. Update the S01-dependent README numbers from the check report
      (perturbation bullet: share of runs within 0.07 m/yr, S01 EXP03/04,
      EXP14/15, "S01 at 100 m has only 219 ice cells"; S01 velocity 29 % ->
      28 %), copy the README into GROUP_FAU2.
   3. `python experiments/continuix/report_resolution.py`: 6-page report
      `summary/ContinuIX_FROST_report_2026-10-08.pdf` (submission summary
      from make_summary.py + model-grid section).
   4. Commit, upload.
2. Then: all experiments at 25 m (G01 at 25-50 m, depends on cost),
   repackage, recheck, update README (step 2 of the pre-processing, EXP14
   and EXP15 notes), redraw the summary.
3. Upload `data/results/continuix/GROUP_FAU2/` to the ContinuIX
   SharePoint.
4. Optional: ELA with a horizontal trend for G01 (5 parameters); a more
   flexible SMB profile for G03. Not planned for this submission.

If anything is rerun: `sbatch experiments/continuix/package_and_check.sh`
(compute node) rebuilds `GROUP_FAU2/` and `check_GROUP_FAU2.txt`, then
`python experiments/continuix/make_summary.py` (login node is fine)
redraws `summary/ContinuIX_FROST_summary.pdf`. Anything that reads many
result files runs as a job, not on the login node.

## Upload folder

`data/results/continuix/GROUP_FAU2/`: 149 result files, README, log
(42.9 GPU hours), filled checklist. Group FAU2; contributor Oskar
Herrmann (ORCID 0000-0002-0319-9065); Johannes Fürst acknowledged as FROST
co-developer, not a contributor.

| Experiments | Glaciers | Files |
|---|---|---|
| EXP01 | G01-G06, S01, S02 | 8 |
| EXP02 (raw data) | G01-G06, S02 (no S01) | 7 |
| EXP03-15 | G01-G06, S01, S02 | 104 |
| EXP16-20 (GLOB) | G01-G06 | 30 |

All files pass the format checks. The 34 findings in
`check_GROUP_FAU2.txt` are known and in the README: G03 budget (~-0.6
m/yr in EXP01-15, -1.4 in EXP02), S01 at 100 m (EXP15), GLOB weak
constraint (12 of 30 runs), local FDIV extremes in thick ice.

## Method

Data: Zenodo 10.5281/zenodo.21401808 in `data/raw/continuix`. Pipeline
per experiment and glacier (`run_continuix.py`, one a40 job each, 10-15
min, G01 ~37 min):
1. `raw` (EXP02 only, `frost/preprocess/continuix_raw.py`): GeoTIFFs and
   shapefiles -> `EXP02_input/` on the EXP01 grid.
2. `prepare`: ContinuIX netCDF -> 50 m grid (S01 25 m, G01 100 m; coarser
   grids kept; block average).
3. `inversion`: provided THK fixed, sliding `tau_ref` inverted from the
   surface velocities (`params_inversion_tau.yaml`, lam 1e11).
4. `calibrate`: ES-MDA (36 members, 6 iterations) on ELA, ablation and
   accumulation gradient against dh/dt in 50 m bands.
5. `submit`: `EXP##_G##_method01.nc` on the original grid: SMB, UNCT_SMB,
   FDIV, UNCT_FDIV, DENSITY (910 kg m-3).

Decisions and why:
- Provided THK, not inverted: otherwise the THK experiments equal EXP01.
  With default sliding the velocities were off by 0.8-4x; inverting
  tau_ref fixes it (G03 speed RMS 43 -> 6.6 m/yr).
- lam 1e11 for all glaciers (L-curve on G03/G05); grid-independent since
  IGM's Laplacian is in physical units. Lower lam on G03 (1e10) fitted
  velocities better but not dh/dt; reverted.
- Velocity misfit std 1 m/yr: the ContinuIX velocity "uncertainties" are
  temporal standard deviations; using them acted as a per-glacier lam and
  worsened four of five glaciers.
- `model_error: 0.5` m/yr added to the band covariance (G03 1.3, G05 1.5
  = 0.5 sqrt(chi2/n)). Without it ES-MDA collapsed the ensembles and S02
  diverged.
- No SMB parameter bounds (would tie FROST to this SMB model).
- Ice mask: ICEMASK cells with THK, velocity or dh/dt data (ContinuIX
  fills gaps with 0); EXP03-13 use the EXP01 mask of the glacier.
- DEM gaps filled in `write_submission`.
- EXP02 thickness: rasters for G01, G04, G06; GPR points only for G02,
  G03, G05, S02: IGM's start thickness scaled by the interpolated ratio
  to the per-cell GPR mean. Joint thk + tau_ref inversion and a
  sqrt(distance) shape were tried and dropped.
- GLOB: Hugonnet UNCT_DHDT with its long-range correlation constrains the
  glacier mean only to ~0.9 m/yr, so 12 of 30 runs miss the observed
  mean by > 0.5 m/yr (G02 EXP17-20, G05 EXP17/19/20, G06 EXP17/20, G01
  EXP19). Accepted and documented rather than tightening the constraint.

## Results

EXP01 (glacier means, m/yr ice eq.):

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

S01 truth (ELA 2350, gradients 10 m/yr per km, acc capped at 5 m/yr):
recovered ELA 2290 +- 40, abl 12.1 +- 1.7, acc 6.4 +- 0.9.

Known weaknesses:
- G03: the SMB the data need is flat at ~-4 m/yr from 2300 to 2600 m,
  then rises ~11 m/yr per km; no single-ELA profile fits, the upper half
  thins ~1.2 m/yr too fast (structural).
- G01: per-basin misfit up to +-1 m/yr (west too positive, south/east
  too negative); one ELA cannot hold it.
- Velocity fit poor on G06 (-31 %, r 0.61) and S01 (-29 %); G04 speeds
  near noise. EXP02 G04/G06 worse (gappy raw velocities).

EXP02 vs EXP01 (model dh/dt): G01 -0.92 (-0.75), G02 +1.36 (+1.46), G03
-2.32 (-1.55), G04 -1.16 (-1.18), G05 -2.00 (-2.00), G06 -1.25 (-1.19),
S02 -0.57 (-0.45).

EXP03-15 (glacier-mean SMB minus EXP01, mandatory glaciers): thickness
experiments move the parameters, not the mean SMB (|diff| <= 0.28 m/yr;
dh/dt pins it down); velocity experiments <= 0.04; EXP15 S01 at 100 m
-0.35 (219 cells, margins average poorly).

## Notes

- The Monitor plots on compute nodes have no text (font missing there).
- Reading S01 (5000x1001 at 1 m) takes ~6 min on the login node.
- Deleted 2026-10-08: superseded EXP01 runs (EnKF 28 Sep, ES-MDA 5 Oct,
  sigma 0.5 and accepted 6 Oct), `vel_std_test/`, `tau_sweep/`,
  `thk_forward/` outputs. Their numbers are in the git history of this
  file.
