# igm330_TI: TI calibration on IGM 3.3.0

Temperature-index SMB calibration (OGGM monthly TI model: melt factor,
precipitation factor, temperature bias; prior mean from OGGM's calibration
of each glacier) of the 25 GLAMOS / WGMS glaciers, as `central_europe_TI_combined`
but with the current setup:

| | Glaciers | Config |
|---|---|---|
| With ice flow, ITS_LIVE p99 >= 5 m/yr | 17 | `pipeline_config.yaml`: IGM 3.3.0, live IGM model, credit retraining |
| Without ice flow, p99 < 5 m/yr: dh/dt = SMB | 8 | `pipeline_config_noflow.yaml` |

Inputs (download, inversion, emulator) from `central_europe_retrain` and
`central_europe_noflow`, as `igm330_live`; ELA reference: `igm330_live_combined`.
The live IGM model with credit retraining gives the same forward runs as
fresh IGM runs with the TI SMB too (Rhone, 8 members, bit-identical, 7 s
instead of 28 s per member; 2026-11-20).

## Steps (repository root, Alex)

1. `bash experiments/igm330_TI/copy_inputs.sh`
2. `sbatch --array=1-17 experiments/igm330_TI/calibration_sbatch.sh` and
   `sbatch experiments/igm330_TI/noflow_sbatch.sh`
3. Report: `sbatch experiments/igm330_TI/report_sbatch.sh`
   → `experiments/igm330_TI/ti_report/ti_report.pdf` (not in the repository; `--png DPI` also
   writes every page as PNG)

`ti_report.py` adapts the sanity report to the TI model: TI parameters
against the prior and the earlier TI calibration, ES-MDA iterations and
parameter trade-offs, glacier-wide balance per year, winter and summer
balance, band balance by elevation, TI flags; then the sanity report pages
(thickness, velocity, dh/dt, SMB profiles) and a TI page per glacier.

## Velocity against the ELA calibration (2026-11-20)

Forward speed in the velocity year (median over the 17 glaciers with ice
flow, against ITS_LIVE): ELA 1.09 (igm_default) and 1.04 (credit), TI 1.17
(credit); within +-33 %: 13, 13 and 9 of 17. The retraining rule is not the
cause: speeds of 2000 are identical in all three, ELA with credit is as ELA
with igm_default, and on IGM 3.2.1 (same rule for both) TI was faster too.
TI thins some glaciers less than observed and than ELA (Vadret Pers -0.74
against -1.09 m/yr observed, ELA -0.97; Hintereisferner -0.95 against
-1.21), so they stay thicker and faster (TI/ELA speed against TI-ELA dh/dt:
r 0.72); the TI band dh/dt misfit is higher than ELA's on almost all
glaciers. The retraining rule shows on slow glaciers: igm_default (5
iterations every 5 steps) lowers the speed in steps, credit smoothly.
