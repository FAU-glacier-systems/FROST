"""
End-to-end run of the `test_default` experiment on the Rhone glacier
(RGI2000-v7.0-G-11-01706): download, IGM inversion and EnKF calibration.

Unlike tests/test_smoke.py this needs network access, the full igm install
and tens of minutes of compute, so it is skipped unless FROST_E2E=1 is set:

    FROST_E2E=1 python -m pytest tests/test_end_to_end.py -v -s

or as a GPU job on Alex: sbatch tests/run_end_to_end.sh

Results go to data/results/test_default/glaciers/RGI2000-v7.0-G-11-01706,
the same place `python frost_pipeline.py` writes them.
"""
import json
import os
import subprocess
import sys
import time

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join("experiments", "test_default", "pipeline_config.yml")
RGI_ID = "RGI2000-v7.0-G-11-01706"  # Rhone

pytestmark = pytest.mark.skipif(
    os.environ.get("FROST_E2E") != "1",
    reason="slow end-to-end run; set FROST_E2E=1 to enable",
)


def _is_fresh(path, since):
    return os.path.exists(path) and os.path.getmtime(path) >= since


def test_test_default_rhone():
    with open(os.path.join(REPO_ROOT, CONFIG_PATH)) as f:
        cfg = yaml.safe_load(f)
    rgi_id_dir = os.path.join(REPO_ROOT, "data", "results",
                              cfg["experiment_name"], "glaciers", RGI_ID)
    preprocess_outputs = os.path.join(rgi_id_dir, "Preprocess", "outputs")
    outputs_before = set(os.listdir(preprocess_outputs)) \
        if os.path.isdir(preprocess_outputs) else set()

    start = time.time()
    # Run as a subprocess: the pipeline chdirs around and hydra keeps
    # global state, neither of which should leak into the test process.
    result = subprocess.run(
        [sys.executable, "frost_pipeline.py",
         "--config", CONFIG_PATH, "--rgi_id", RGI_ID],
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, "frost_pipeline.py failed"

    # Download
    assert _is_fresh(os.path.join(rgi_id_dir, "Preprocess", "data",
                                  "input.nc"), start)

    # Inversion: fixed hydra run dir, no new outputs/<date>/<time> folders
    assert _is_fresh(os.path.join(preprocess_outputs, "igm", "inversion",
                                  "optimize.nc"), start)
    assert _is_fresh(os.path.join(preprocess_outputs, "output.nc"), start)
    new_entries = set(os.listdir(preprocess_outputs)) - outputs_before
    assert new_entries <= {"igm", "output.nc", "iceflow-model"}, \
        f"unexpected new entries in Preprocess/outputs: {new_entries}"

    # Calibration
    assert _is_fresh(os.path.join(rgi_id_dir, "observations.nc"), start)
    results_path = os.path.join(rgi_id_dir, "calibration_results.json")
    assert _is_fresh(results_path, start)
    with open(results_path) as f:
        results = json.load(f)
    ensemble_size = cfg["EnKF"]["ensemble_size"]
    assert len(results["final_ensemble"]) == ensemble_size
    assert len(results["final_mean"]) == 3  # ela, abl_grad, acc_grad
    assert all(v == v for v in results["final_mean"]), "NaN in final_mean"
    assert os.path.isdir(os.path.join(rgi_id_dir, "Ensemble", "Member_0",
                                      "outputs", "igm", "forward"))
