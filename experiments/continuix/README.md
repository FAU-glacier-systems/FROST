# ContinuIX: SMB-gradient contribution

FROST's contribution to ContinuIX WP2/WP3
(https://github.com/ContinuIX/ContinuIX-1): for each experiment (EXP01-EXP20) and
single glacier, calibrate the ELA mass-balance model against the provided dh/dt and
write the submission file `EXP##_G##_method01.nc`. The current state, results and
open items are in `STATUS.md`.

## Run

All commands from the repository root.

```bash
# one experiment and glacier
python experiments/continuix/run_continuix.py --exp EXP01 --glacier S01
# on Alex, one job per line of a task list ("EXP01 S01")
sbatch --array=1-$(wc -l < experiments/continuix/tasks_exp01.txt) \
    experiments/continuix/run_continuix.sh experiments/continuix/tasks_exp01.txt
```

`--steps prepare,inversion,calibrate,submit` runs a subset of the steps.

## Inputs

- `data/raw/continuix/` (Zenodo 10.5281/zenodo.21401808, unzipped)

## Outputs

- `data/results/continuix/EXP##/<glacier>/`: FROST results per glacier
- `data/results/continuix/submission/EXP##/`: submission files
- `data/results/continuix/EXP02_input/`: EXP02 raw data on the EXP01 grid (step `raw`)
- `data/results/continuix/GROUP_FAU2/`: upload folder (`package_and_check.sh`)
- `data/results/continuix/check_GROUP_FAU2.txt`: format and budget check of every file
- `data/results/continuix/summary/`: two-page PDF summary (`make_summary.py`)

## Files

| File | Purpose |
|---|---|
| `STATUS.md` | State of the submission, EXP01 results, open items |
| `config.yml` | Data and results folders, resolution, EnKF settings |
| `params_inversion_tau.yaml` | Inversion of the sliding `tau_ref` with the provided thickness (used) |
| `params_inversion.yaml` | Thickness inversion from an SIA start (alternative) |
| `run_continuix.py`, `run_continuix.sh` | Pipeline for one experiment and glacier; Slurm job |
| `package_submission.py`, `submission/` | Upload folder `GROUP_FAU2/`: result files, log, README, checklist |
| `check_submission.py`, `package_and_check.sh` | Format and budget check of the upload folder; Slurm job that packages and checks |
| `make_summary.py` | PDF summary from the check report |
| `report_resolution.py` | Report: submission summary + model-grid section with the tests of 8 Oct |
| `tasks_*.txt` | Task lists for job arrays (`exp01`, `exp02`, `exp03-15`, `exp03-15_optional`, `glob`, `res25`) |
| `inversion_test.py`, `.sh`, `inversion_test_tasks.txt` | Inversion settings (nbitmax, lam, emulator retraining) at 25 vs 50 m |
| `lam_test.py`, `.sh`, `lam_test_tasks.txt` | lam and grid test: full EXP01 runs scored on the posterior dh/dt bands |
| `bench_forward.py`, `.sh` | Timing of the ES-MDA forward runs (start-up vs simulation) |
| `tasks_grid50.txt` | Reruns of G01 and S01 on the 50 m grid (8 Oct) |
| `config_res50.yml`, `res50_test.sh`, `tasks_res50.txt` | G01 and S01 on the 50 m grid of the other glaciers |
| `config_native.yml`, `native_test.sh`, `tasks_native.txt` | Curiosity: EXP01 of G03-G06 on the ContinuIX data grid (2-20 m) |
| `config_res25.yml` | Resolution test: as `config.yml` on a 25 m grid, results in `res25/` |
| `thk_forward.py`, `.sh` | Forward run with the provided thickness vs observed velocities (diagnostic, outputs deleted) |
| `tau_sweep.py`, `.sh` | L-curve of the `tau_ref` regularisation on G03, G05 (diagnostic, outputs deleted) |
