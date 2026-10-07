# alps_TI_projections: status

Goal: TI calibration with FROST and CORDEX projections to 2100 for the 380
glaciers in `glacier_selection.csv`. State of 2026-10-07.

## Running

- `tau_cap_test_sbatch.sh` (array 1-11 over `test_glaciers.txt`): inversions
  `slide_only_tau1e11` and `slide_only_tau1e12` (tau_ref smoothed 10x/100x,
  lower bound 0.05), and for Fiescher BE (01581) and Gorner (01225) the
  RCP8.5 diagnostic to 2035 with the 1000 m/yr velocity cap on the current
  inversion and both smoother ones. Logs: `logs/tau_cap_<job>_*`.

## Next

1. Collect: `python experiments/alps_TI_projections/inversion_variants.py --collect`
   (`tables/inversion_variants.tsv`), and read
   `data/results/alps_TI_projections/diagnostics/{new_cap,tau1e11_cap,tau1e12_cap}/<rgi_id>/yearly.tsv`.
   Accept the cap if dt_cfl stays near 0.05-0.1 yr. Take the strongest
   tau_ref smoothing that keeps the velocity RMS near 6 m/yr.
2. Copy the chosen variant into `params_inversion.yaml` and rerun the 11 test
   glaciers: `sbatch --array=1-11 experiments/alps_TI_projections/alps_TI_sbatch.sh experiments/alps_TI_projections/test_glaciers.txt`.
   Raise the time limit if the projections are still slow.
3. Compare with the old inversion (`data/results/alps_TI_projections_igm_guess/`):
   band dh/dt fit (upper bands thickened before), posterior misfit,
   2100 volume (Oberaletsch/Corbassiere grew under RCP2.6 before). Do this
   analysis in a job, it reads many ensemble files.
4. Then all 380 glaciers (CORDEX files are in `data/raw/cordex_alps/`).
5. Commit: nothing of this is committed yet.

## Decisions

- SMB: OGGM TI model (melt_f, prcp_fac, temp_bias), prior mean from each
  glacier's OGGM `mb_calib.json` (`prior_from_oggm`), W5E5 baseline.
- Projections: posterior mean x all complete CORDEX runs (20 RCP2.6, 24
  RCP4.5, 59 RCP8.5), bias-corrected to W5E5 per month over 2000-2019,
  2000-2100 from the observed 2000 surface (`run_projections.py`).
- Ice flow: pretrained `dahunet_mini` only, frozen in inversion and forward
  runs (no fine-tuning, `nbit_init: 0`).
- Inversion: thickness fixed to Millan et al. (2022) shifted from 2017 to
  2000 with the Hugonnet dh/dt (`frost: thk_start: product_2000`); only
  tau_ref (log10) fitted to velocity. No Farinotti consensus.
- GlaThiDa only for validation, never in the fit; surveys are moved to 2000
  with their date and dh/dt (`thickness_validation.py`).
- Velocity cap `max_velbar: 1000` m/yr in this experiment's forward runs;
  off by default in the shared code (ContinuIX).
- All IGM runs as Slurm jobs on Alex, never on the login node.

## Findings

- Old inversion (IGM initial guess, velocity only) was 30-80 % too thin on
  most glaciers and missed the velocities by 3-12x even with the exact
  solver: the thin start, not the network. Calibrations then could not thin
  the glaciers enough.
- Pretrained network vs direct solver on the same state: 9-46 % too fast
  (median), small compared with that gap (`emulator_check.py`).
- Millan-2000 + tau_ref inversion (`inversion_variants.py`): velocity median
  0.71-1.08 of observed (was 0.13-0.93), velocity RMS 5.9 m/yr (was 10.9),
  thickness RMS vs GlaThiDa-2000 87 m (was 107). Too thick where Millan is
  (Aletsch, Pasterze, Gepatsch). `joint` (thk + tau_ref) = `slide_only`.
- With that inversion, single cells at retreating thick ice fronts reached
  1-3 km/yr in the projections and forced time steps of ~0.005 yr; Gorner,
  Fiescher BE and Corbassiere hit the job limit (partial `projections.nc`
  with 29/60/59 runs). Hence the cap and the smoother tau_ref.
