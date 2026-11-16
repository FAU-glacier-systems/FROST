# igm330_retrain: emulator retraining on IGM 3.3.0

FROST moves from IGM 3.2.1 plus a local option (`iceflow.unified.retrain_interval`,
20 retraining iterations every year of model time) to plain IGM 3.3.0. This
experiment chooses the retraining rule of the forward runs from IGM's own
options (`frost/glacier_model/igm_wrapper.py`, `RETRAIN_RULES`):

| Rule | After 300 start-up iterations |
|---|---|
| `igm_default` | 5 iterations every 5 time steps |
| `credit` | 5 iterations once the accumulated, normalised change of the inputs reaches kappa = 1e-3 (IGM 3.3.0 `adaptive_time`) |

Reference: `central_europe_retrain_t1` (IGM 3.2.1 + `retrain_interval`).

## Steps (repository root)

1. Forward test against the solver (Aletsch, Rhone, Corbassière; Alex):
   `sbatch --array=1-3 experiments/igm330_retrain/forward_test_sbatch.sh`
2. Calibrations of the 17 glaciers with ice flow (Alex):
   `bash experiments/igm330_retrain/copy_inputs.sh`, then
   `sbatch --array=1-17 experiments/igm330_retrain/calibration_sbatch.sh igm330_default` and
   `sbatch --array=1-17 experiments/igm330_retrain/calibration_sbatch.sh igm330_credit`
3. GLAMOS/WGMS evaluation (Fritz): `sbatch experiments/igm330_retrain/evaluate_sbatch.sh`
4. Report: `python experiments/igm330_retrain/report.py --default_job <id> --credit_job <id>`
   → `experiments/igm330_retrain/igm330_retrain_report.pdf`
