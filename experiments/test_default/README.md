# test_default: reference run on one glacier

The default FROST setup, used for development and the end-to-end test: download
OGGM-shop data, invert the ice thickness from surface velocities, calibrate the
ELA mass-balance model with ES-MDA against Hugonnet et al. (2021) elevation
change. The inversion is set up exactly as the igm-examples Aletsch tutorial
(Part D, `params_D_real.yaml`): SIA start, Huber velocity misfit,
squared-Laplacian regularisation of the bed.

## Run

All commands from the repository root.

```bash
# Rhone (the configured glacier)
python frost_pipeline.py --config experiments/test_default/pipeline_config.yml
# Aletsch, with the same setup
python frost_pipeline.py --config experiments/test_default/pipeline_config.yml \
    --rgi_id RGI2000-v7.0-G-11-02596
# the inversion on the igm-examples' own Aletsch input.nc, for comparison
python experiments/test_default/reference_inversion.py
# on Alex: sbatch hpc_sbatch.sh experiments/test_default/pipeline_config.yml <rgi_id>
```

## Inputs

- OGGM shop (downloaded by the pipeline)
- `data/raw/hugonnet/11_rgi60_2000-01-01_2020-01-01` (Hugonnet et al. 2021 dh/dt)
- `../igm-examples/aletsch/data/input.nc` (only `reference_inversion.py`)

## Outputs

- `data/results/test_default/<rgi_id>/`: `calibration_results.json`,
  `observations.nc`, `Preprocess/` (input and inversion), `Ensemble/`, `Monitor/`
- `data/results/test_default/igm_examples_aletsch/`: the reference inversion

## Files

| File | Purpose |
|---|---|
| `pipeline_config.yml` | Glacier, pipeline steps, EnKF settings and SMB prior |
| `params_inversion.yaml` | IGM `field_inversion` setup |
| `reference_inversion.py` | Same inversion on the igm-examples input |
