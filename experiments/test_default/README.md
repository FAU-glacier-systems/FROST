# test_default: reference run on one glacier

The default FROST setup, used for development and the end-to-end test: download
OGGM-shop data with the IceBoost v2.0 thickness, invert for sliding, calibrate
the ELA mass-balance model with ES-MDA against Hugonnet et al. (2021) elevation
change. The inversion keeps the geometry fixed: IceBoost, shifted from 2012
(its TanDEM-X surface) to 2000 with the Hugonnet dh/dt; only tau_ref is fitted
to the surface velocity (Huber misfit, squared-Laplacian smoothness 1e10) on
the geometry of the ITS_LIVE year (2018). Chosen in `experiments/inversion_setup`
(IceBoost against date-corrected GlaThiDa: bias +8 m, RMS 46 m; Millan et al.
2022: +69 m, 115 m). The forward runs retrain the iceflow emulator on the
glacier (`emulator_retrain`). These are the settings of the reference run
`central_europe_combined_retrain` (see `experiments/README.md`); glaciers that
barely move (ITS_LIVE p99 on the ice < 10 m/yr) take `EnKF: ice_flow: none`
instead.

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
| `params_inversion.yaml` | IGM `field_inversion` setup (fixed IceBoost geometry, tau_ref fitted) |
| `params_inversion_igm_example.yaml` | The igm-examples Aletsch Part D inversion (thickness from velocity), the default until 2026-10-22 |
| `reference_inversion.py` | The Part D inversion on the igm-examples input |
