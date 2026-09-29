# tau_ref sweep: sliding in the thickness inversion

Which fixed sliding parameter `tau_ref` gives the best thickness in the
`test_default` inversion? The inversion is run on Rhone and Aletsch for a range of
`tau_ref` values (basal shear stress in MPa that gives `u_ref` = 100 m/yr sliding;
the igm-examples default is 0.213) and scored against the GlaThiDa thickness
observations, which the inversion does not use, and the surface velocities.

## Run

All commands from the repository root.

```bash
# inversions, one GPU job per glacier
sbatch experiments/tau_ref_sweep/tau_ref_sweep.sh --glaciers RGI2000-v7.0-G-11-01706
sbatch experiments/tau_ref_sweep/tau_ref_sweep.sh --glaciers RGI2000-v7.0-G-11-02596
# comparison
python experiments/tau_ref_sweep/tau_ref_sweep.py --plot
```

`--tau_refs` runs other values than the default list.

## Inputs

- `data/results/test_default/<rgi_id>/Preprocess/data/input.nc` (run `test_default` first)
- `experiments/test_default/params_inversion.yaml`, with `tau_ref` replaced

## Outputs

- `data/results/tau_ref_sweep/<rgi_id>/tau_ref_<value>/`: inversion per value,
  with `Preprocess/outputs/thickness_validation.{json,png}`
- `tables/tau_ref_sweep.tsv`: thickness, GlaThiDa bias and RMS, speed ratio and
  RMS, sliding share per glacier and value
- `plots/tau_ref_sweep.png`: the same against `tau_ref`

## Files

| File | Purpose |
|---|---|
| `tau_ref_sweep.py` | Runs the inversions (`--run`) and compares them (`--plot`) |
| `tau_ref_sweep.sh` | Slurm job for the inversions |
