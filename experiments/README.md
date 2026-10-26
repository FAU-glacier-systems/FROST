# Experiments: where things stand (2026-10-26)

Start here after a break. Each folder has its own STATUS.md (or README.md)
with the details, commands and job numbers.

## Reference setup

`central_europe_combined_retrain`: the ELA calibration of the 25 GLAMOS /
WGMS glaciers with IGM 3.2.1 and ES-MDA. Report:
`central_europe_combined_retrain/sanity_report/report_dark.pdf`.

"Combined" means each glacier takes the calibration that suits it, chosen by
the ITS_LIVE speed on the ice (99th percentile, 10 m/yr):

| | Glaciers | Run (data in `data/results/`) |
|---|---|---|
| With ice flow, p99 >= 10 m/yr | 9 (Aletsch, Pasterze, Rhone, ...) | `central_europe_retrain` |
| Without, p99 < 10 m/yr: dh/dt = SMB | 16 | `central_europe_noflow` |

`data/results/central_europe_combined_retrain/` only holds links to these
two runs.

Settings (also in `test_default`, the default for new experiments):
- Download: IceBoost v2.0 thickness, ITS_LIVE 2014-2022 velocity, Hugonnet
  dh/dt 2000-2020, grids no finer than 50 m.
- Inversion: IceBoost geometry fixed (shifted to 2000 with the dh/dt), only
  tau_ref fitted, smoothness 1e10, on the geometry of the velocity year
  (2018); velocities ignored where p99 < 3 m/yr.
- Forward runs: emulator retrained on the glacier (300 iterations at the
  start, then 20 every step, or 5 every 5 on fast glaciers with short time
  steps), speed cap 1000 m/yr.
- ES-MDA: 32 members (128 without flow), 4 iterations, 50 m bands with the
  band covariance, model error 0.5 m/yr.

Against the submitted Brief Communication run (`central_europe`): SMB
profile MAE 0.86 m w.e./yr (1.13), 90 % of the bins within 2 sigma (72 %),
ablation gradient MAE 2.6 (3.6), accumulation 0.9 (1.7); ELA about as good
(r 0.62, MAE 139 m vs 0.68, 120 m) with honest spreads.

## Open

1. Aletsch: ELA still too high (3355 vs 3034 m), ablation gradient too low.
2. Selection rule for ice flow (p99 10 m/yr, or flux divergence against the
   band error); no-flow has no ice export, so its upper areas are biased.
3. Then all 409 glaciers of `central_europe`.
4. TI instead of the ELA model: `central_europe_TI` (first look, not
   finished; next step in its STATUS.md).
5. `alps_TI_projections`: next steps in its STATUS.md (branch
   `alps-ti-projections`).

## All experiments

| Folder | What it showed | Data |
|---|---|---|
| `central_europe_combined_retrain` | reference, see above | links |
| `central_europe_retrain` | IGM part of the reference | kept |
| `central_europe_noflow` | no-flow part of the reference; SMB profiles of slow glaciers as good as with IGM | kept |
| `central_europe_TI`, `central_europe_TI_meltout` | TI SMB model; interannual variability right, mean profile as ELA; leaving out melted-out bands hardly changes anything | TI kept, meltout ensembles deleted |
| `central_europe_combined`, `central_europe_esmda_velyear` | the same with the frozen emulator: start-up speed shock (Pasterze 2000/2018 6.5x), Jamtal unstable | ensembles deleted |
| `central_europe_esmda` | first IGM 3.2 + ES-MDA redo; tongues of large glaciers starve when tau_ref is fitted in 2000 | ensembles deleted |
| `emulator_test` | retraining the emulator in the forward runs follows the physics solver (speeds within ~10 %) | kept |
| `central_europe_pretrain`, `central_europe_combined_pretrain` | fine-tuning the emulator before the inversion: no gain, dropped | ensembles deleted |
| `central_europe_tau_default` | uniform tau_ref instead of the inversion: too crude, shock back | ensembles deleted |
| `inversion_setup`, `tau_ref_sweep` | IceBoost + tau_ref-only inversion chosen | kept |
| `thickness_esmda` | thickness product and velocity source vs calibrated SMB | ensembles deleted |
| `esmda_vs_enkf` | ES-MDA chosen over the iterated EnKF | ensembles deleted |
| `test_default` | the default setup on one glacier, end-to-end test | ensembles deleted |
| `central_europe` | the submitted Brief Communication (exact code on branch `hydra`) | `central_europe_submit` kept |
| `continuix` | ContinuIX submission v1 (`GROUP_FAU2.zip`) | kept until accepted |
| `alps_TI_projections` | TI + CORDEX projections, in progress | kept |
| `validation` | GLAMOS / WGMS reference tables | - |
| `TwinExperiment`, `Plots` | old FROST layout, not runnable | - |

"Ensembles deleted" (2026-10-26): `Ensemble/` (the members' IGM runs, ~95 %
of a run) removed to save space; `calibration_results.json`, `Monitor/`,
`observations.nc` and `Preprocess/` (inputs and inversion) are kept, so the
numbers can be checked and the inversions reused. The reports were made
before.
