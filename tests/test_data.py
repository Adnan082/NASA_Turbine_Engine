"""
Data integrity and v1 metric-regression checks. Run on a clean clone after
`python scripts/prepare_data.py` — no GPU or API key required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "agents"))
sys.path.append(str(ROOT / "scripts"))

MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
MODEL_DIR = ROOT / "models"

EXPECTED_TEST_COUNTS = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}

pytestmark = pytest.mark.skipif(
    not (MODEL_READY_DIR / "X_test.npy").exists(),
    reason="DATA/model_ready is missing — run `python scripts/prepare_data.py` first",
)


@pytest.fixture(scope="module")
def test_arrays():
    return {
        "X_test": np.load(MODEL_READY_DIR / "X_test.npy"),
        "y_test": np.load(MODEL_READY_DIR / "y_test.npy"),
        "subset_test": np.load(MODEL_READY_DIR / "subset_test.npy"),
    }


def test_shapes(test_arrays):
    assert test_arrays["X_test"].shape == (707, 50, 16)
    assert test_arrays["y_test"].shape == (707,)


def test_engine_counts_per_subset(test_arrays):
    subset_test = test_arrays["subset_test"]
    for name, expected_n in EXPECTED_TEST_COUNTS.items():
        assert int((subset_test == name).sum()) == expected_n


def test_no_nans(test_arrays):
    assert not np.isnan(test_arrays["X_test"]).any()
    assert not np.isnan(test_arrays["y_test"]).any()


def test_rul_in_valid_range(test_arrays):
    y_test = test_arrays["y_test"]
    assert y_test.min() >= 0
    assert y_test.max() <= 125


def test_s6_scaled_range_in_fd004(test_arrays):
    """s6 legitimately reaches about -0.6 in FD004 — the per-condition MinMax
    scaler is fit on train, so test values can sit outside [0, 1]."""
    X_test, subset_test = test_arrays["X_test"], test_arrays["subset_test"]

    import pandas as pd
    sensor_cols = [c for c in pd.read_parquet(ROOT / "DATA" / "pre_processed_data" / "train_FD001.parquet").columns if c.startswith("s")]
    s6_idx = sensor_cols.index("s6")

    fd004_s6 = X_test[subset_test == "FD004"][:, :, s6_idx]
    assert fd004_s6.min() < -0.4, f"expected s6 to dip below -0.4 in FD004, got {fd004_s6.min()}"


def test_prepare_data_matches_committed_arrays(tmp_path, monkeypatch):
    """scripts/prepare_data.py, run fresh, must reproduce DATA/model_ready
    exactly (np.allclose) — this is what makes the pipeline reproducible
    from a clean clone instead of relying on a gitignored cache."""
    import importlib

    prepare_data = importlib.import_module("prepare_data")

    monkeypatch.setattr(prepare_data, "MODEL_READY_DIR", tmp_path)
    prepare_data.main()

    for name in ["X_train", "y_train", "cond_train", "X_test", "y_test", "cond_test"]:
        committed = np.load(MODEL_READY_DIR / f"{name}.npy")
        fresh = np.load(tmp_path / f"{name}.npy")
        assert fresh.shape == committed.shape, f"{name} shape mismatch"
        assert np.allclose(fresh, committed), f"{name} values differ from the committed model_ready cache"


def test_v1_metrics_regression():
    """Pins the CV-quoted v1 numbers to ±0.01. If this fails, the v1 model
    checkpoint or the evaluation code has changed — investigate before
    touching anything else."""
    if not (MODEL_DIR / "agent2_rul_predictor.pt").exists():
        pytest.skip("v1 model checkpoint not found")

    from agent2_rul import RULAgent
    from evaluate_rul import compute_metrics

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")
    agent = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")

    metrics = compute_metrics(X_test, y_test, agent)

    assert metrics["pooled"]["mae"] == pytest.approx(12.20, abs=0.01)
    assert metrics["pooled"]["rmse"] == pytest.approx(17.58, abs=0.01)


def test_v1_conformal_coverage_at_least_90_percent():
    """The conformal interval targets 90% coverage (alpha=0.10); the
    non-calibration engines should meet or exceed that, matching the
    92.9% observed for v1."""
    if not (MODEL_DIR / "agent2_rul_predictor.pt").exists():
        pytest.skip("v1 model checkpoint not found")

    from agent2_rul import RULAgent
    from evaluate_rul import compute_metrics

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")
    agent = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")

    metrics = compute_metrics(X_test, y_test, agent)
    assert metrics["conformal"]["coverage_application"] >= 0.90
