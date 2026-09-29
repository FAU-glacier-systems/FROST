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
- `data/results/continuix/thk_forward/`, `tau_sweep/`: diagnostics

## Files

| File | Purpose |
|---|---|
| `STATUS.md` | State of the submission, EXP01 results, open items |
| `config.yml` | Data and results folders, resolution, EnKF settings |
| `params_inversion_tau.yaml` | Inversion of the sliding `tau_ref` with the provided thickness (used) |
| `params_inversion.yaml` | Thickness inversion from an SIA start (alternative) |
| `run_continuix.py`, `run_continuix.sh` | Pipeline for one experiment and glacier; Slurm job |
| `tasks_*.txt` | Task lists for job arrays (`test`, `exp01`, `all`) |
| `thk_forward.py`, `.sh` | Forward run with the provided thickness vs observed velocities |
| `tau_sweep.py`, `.sh` | L-curve of the `tau_ref` regularisation (G03, G05) |
