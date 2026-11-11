# Experiments

Each folder holds the configuration, job scripts and evaluation scripts of
one experiment, and a README.md where the experiment is meant to be rerun.
Results are written to `data/results/<experiment>/` (not part of the
repository).

## Reference setup (FROST v2.0)

`central_europe_combined_retrain`: the ELA calibration of the 25 glaciers
with GLAMOS / WGMS mass-balance measurements, with IGM 3.2.1 and ES-MDA.
Each glacier takes the calibration that suits its flow speed (99th percentile
of the ITS_LIVE speed on the ice):

| | Glaciers | Run |
|---|---|---|
| With ice flow, p99 >= 5 m/yr | 17 (Aletsch, Pasterze, Rhone, ...) | `central_europe_retrain` |
| Without ice flow, p99 < 5 m/yr: dh/dt = SMB | 8 | `central_europe_noflow` |

Settings (also in `test_default`, the template for new experiments):
- Download: IceBoost v2.0 thickness, ITS_LIVE 2014-2022 velocity, Hugonnet
  et al. (2021) dh/dt 2000-2020, grids no finer than 50 m.
- Inversion: IceBoost geometry fixed (moved to 2000 with the dh/dt); only the
  sliding field tau_ref is fitted, smoothness 1e10, on the geometry of the
  velocity year (2018); velocities are ignored where their p99 on the ice is
  below 3 m/yr.
- Forward runs: ice-flow emulator retrained on the glacier (300 iterations at
  the start, then 20 every time step, or 5 every 5 steps on fast glaciers
  with short time steps), speed cap 1000 m/yr.
- ES-MDA: 32 members (128 without ice flow), 4 iterations, 50 m elevation
  bands with the band error covariance, model error 0.5 m/yr.

Evaluation against GLAMOS / WGMS, compared with FROST v1 (`central_europe`,
the run of the Brief Communication):

| | v2.0 | v1 |
|---|---|---|
| SMB profile RMSE (m w.e./yr) | 0.90 | 1.13 |
| Profile bins within 2 sigma | 88 % | 72 % |
| ELA r / MAE (16 glaciers with an uncapped reference ELA) | 0.80 / 113 m | 0.68 / 120 m |
| Ablation gradient MAE (m w.e./yr per km) | 2.6 | 3.6 |
| Accumulation gradient MAE (m w.e./yr per km) | 0.9 | 1.7 |

## Open questions

1. Grosser Aletschgletscher: the ELA is too high (3355 vs 3034 m) and the
   ablation gradient too low. The modelled ice flux into the tongue is
   smaller than dh/dt and the in-situ mass balance require
   (`central_europe_combined_retrain/flux_gate.py`, `inversion_thickness`).
2. Selection of the glaciers without ice flow: without ice export their upper
   areas are biased; the threshold is 5 m/yr since `central_europe_TI_slowflow`
   showed that ice flow improves dh/dt on 8 of 10 slower glaciers.
3. The temperature-index SMB model (`central_europe_TI*`), needed for climate
   projections (`alps_TI_projections`).
4. All 409 glaciers of the region.

## All experiments

| Folder | Result |
|---|---|
| `central_europe_combined_retrain` | reference, see above |
| `central_europe_retrain` | ice-flow part of the reference |
| `central_europe_noflow` | no-flow part of the reference; on slow glaciers the SMB profiles are as good as with IGM |
| `central_europe_TI`, `central_europe_TI_meltout` | temperature-index SMB: interannual variability reproduced, mean profile as with the ELA model; leaving out melted-out bands changes little |
| `central_europe_TI_slowflow`, `central_europe_TI_combined` | TI with ice flow on glaciers of 5-10 m/yr; combined TI run |
| `central_europe_TI_snow`, `central_europe_TI_snow_lapse`, `central_europe_TI_flow_snow_lapse` | snow/ice melt factors and a calibrated lapse rate: profile shape closer to GLAMOS, but no better overall |
| `central_europe_combined`, `central_europe_esmda_velyear` | the same with the frozen emulator: start-up speed shock |
| `central_europe_esmda` | first IGM 3.2 + ES-MDA run; tongues of large glaciers starve when tau_ref is fitted on the 2000 geometry |
| `emulator_test` | retraining the emulator in the forward runs follows the physical solver (speeds within ~10 %) |
| `central_europe_pretrain`, `central_europe_combined_pretrain` | fine-tuning the emulator before the inversion: no gain |
| `central_europe_tau_default` | uniform tau_ref instead of the inversion: too crude |
| `inversion_setup`, `tau_ref_sweep` | choice of IceBoost and the tau_ref-only inversion |
| `inversion_thickness` | joint thickness and sliding inversion (in progress) |
| `thickness_esmda` | influence of the thickness product and velocity source on the calibrated SMB |
| `esmda_vs_enkf` | ES-MDA chosen over the iterated EnKF |
| `test_default` | the default setup on one glacier, end-to-end test |
| `central_europe` | FROST v1, the Brief Communication run |
| `continuix` | contribution to the ContinuIX model intercomparison |
| `alps_TI_projections` | TI calibration and CORDEX projections for 380 Alpine glaciers (in progress) |
| `validation` | GLAMOS / WGMS reference tables |
| `TwinExperiment`, `Plots` | earlier FROST layout, not runnable |
