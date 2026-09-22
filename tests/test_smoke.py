"""
Fast smoke tests for the FROST pipeline.

These checks are deliberately cheap: no network access, no glacier data
download, no IGM inversion or EnKF ensemble runs, and no `igm`/TensorFlow
install at all. They only verify that the package's own modules import
cleanly and that every experiment config in the repo is valid, structured
YAML. That makes them safe and fast to run on every commit (e.g. in CI),
unlike a real end-to-end run of an experiment such as `test_default`, which
downloads real glacier data and takes tens of minutes.

Note: frost_pipeline.py itself (and frost_calibration.py, which it imports)
pull in `igm`, which drags in full GPU-enabled TensorFlow and a set of CUDA
wheels that are never used here (the pipeline forces CPU-only execution
anyway). Exercising that CLI entrypoint is intentionally left out of this
fast tier to avoid that cost on every commit; every frost.* module it
depends on is still checked individually below.
"""
import glob
import os

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FROST_MODULES = [
    "frost",
    "frost.calibration.ensemble_kalman_filter",
    "frost.calibration.observation_provider",
    "frost.glacier_model.igm_wrapper",
    "frost.preprocess.download_data",
    "frost.preprocess.igm_inversion",
    "frost.preprocess.create_observation",
    "frost.preprocess.synth_observation",
    "frost.visualization.monitor",
    # frost.preprocess.insert_thk is intentionally excluded: it is a one-off
    # script hardcoded to a specific glacier (RGI2000-v7.0-G-17-14719) and
    # reads a local CSV at import time, so it isn't meant to be importable
    # as a library module (nothing else in the codebase imports it either).
]


@pytest.mark.parametrize("module_name", FROST_MODULES)
def test_module_imports(module_name):
    """Every core module should import without error or missing dependency."""
    __import__(module_name)


def _experiment_config_paths():
    patterns = [
        "experiments/*/pipeline_config.yaml",
        "experiments/*/pipeline_config.yml",
        "experiments/*/params_inversion.yaml",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(REPO_ROOT, pattern)))
    assert paths, "No experiment config files were found - did the repo layout change?"
    return sorted(paths)


@pytest.mark.parametrize("config_path", _experiment_config_paths())
def test_experiment_config_is_valid_yaml(config_path):
    """Every tracked experiment config must at least parse as a YAML mapping."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{config_path} did not parse to a mapping"


def test_default_pipeline_config_schema():
    """The reference test_default config should have the fields the pipeline expects."""
    config_path = os.path.join(
        REPO_ROOT, "experiments", "test_default", "pipeline_config.yml"
    )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for key in ("experiment_name", "rgi_id", "smb_model", "pipeline_steps", "download", "EnKF"):
        assert key in cfg, f"pipeline_config.yml is missing required key '{key}'"

    for step in ("download", "inversion", "calibrate"):
        assert isinstance(cfg["pipeline_steps"][step], bool)

    for key in ("ensemble_size", "iterations", "seed"):
        assert isinstance(cfg["EnKF"][key], int)
