"""
Reproduce the v1 headline numbers for Agent 2 (RUL prediction) and its
split-conformal calibration, exactly as originally computed.

This does not fix the known calibration-slice issue (see agents/mapie.py
and FIX_REPORT.md) — it reproduces v1 as-is so the numbers quoted for v1
stay verifiable from a script. The corrected calibration lives in
scripts/evaluate_v2.py.

Usage:
    python scripts/evaluate_rul.py

Requires DATA/model_ready/*.npy (see scripts/prepare_data.py) and the
committed v1 checkpoint models/agent2_rul_predictor.pt.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "agents"))

from agent2_rul import RULAgent  # noqa: E402

MODEL_DIR = ROOT / "models"
MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
REPORTS_DIR = ROOT / "reports" / "v1"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
TEST_COUNTS = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}


def require_model_ready():
    required = ["X_test.npy", "y_test.npy", "cond_test.npy"]
    missing = [f for f in required if not (MODEL_READY_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {MODEL_READY_DIR}. "
            "Run `python scripts/prepare_data.py` first."
        )


def nasa_score(y_true, y_pred):
    """Standard C-MAPSS asymmetric scoring function (late predictions cost more)."""
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(s))


def subset_slices():
    slices = {}
    start = 0
    for name in SUBSETS:
        n = TEST_COUNTS[name]
        slices[name] = slice(start, start + n)
        start += n
    return slices


def compute_metrics(X_test, y_test, agent):
    """Pure computation, shared by main() and the pytest regression test —
    one code path for the numbers everyone relies on."""
    results = agent.predict(X_test)
    preds = np.array([r["predicted_RUL"] for r in results], dtype="float32")

    mae = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))

    slices = subset_slices()
    per_subset = {}
    for name, sl in slices.items():
        yt, yp = y_test[sl], preds[sl]
        per_subset[name] = {
            "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "mae": float(np.mean(np.abs(yp - yt))),
            "nasa_score": nasa_score(yt, yp),
            "n": int(sl.stop - sl.start),
        }

    # v1 conformal calibration: last 20% of X_test (see agents/mapie.py).
    # File order (FD001, FD002, FD003, FD004) means this slice is FD004 only.
    split = int(len(X_test) * 0.8)
    X_cal, y_cal = X_test[split:], y_test[split:]
    X_app, y_app = X_test[:split], y_test[:split]

    cal_results = agent.predict(X_cal)
    cal_preds = np.array([r["predicted_RUL"] for r in cal_results])
    residuals = np.abs(cal_preds - y_cal)
    alpha = 0.10
    quantile = float(np.quantile(residuals, 1 - alpha))

    app_results = agent.predict(X_app)
    app_preds = np.array([r["predicted_RUL"] for r in app_results])
    lower = np.clip(app_preds - quantile, 0, 125)
    upper = np.clip(app_preds + quantile, 0, 125)
    covered = (y_app >= lower) & (y_app <= upper)
    coverage = float(covered.mean())

    # Coverage by sub-dataset, restricted to the non-calibration (application) rows.
    coverage_by_subset = {}
    for name, sl in slices.items():
        lo, hi = sl.start, min(sl.stop, split)
        if lo >= hi:
            coverage_by_subset[name] = None
            continue
        coverage_by_subset[name] = {
            "n": int(hi - lo),
            "coverage": float(covered[lo:hi].mean()),
        }

    return {
        "pooled": {"mae": mae, "rmse": rmse, "n": len(y_test)},
        "per_subset": per_subset,
        "conformal": {
            "alpha": alpha,
            "calibration_n": len(y_cal),
            "application_n": len(y_app),
            "quantile": quantile,
            "coverage_application": coverage,
            "coverage_by_subset": coverage_by_subset,
        },
    }


def main():
    require_model_ready()

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")

    agent = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")
    metrics = compute_metrics(X_test, y_test, agent)

    mae = metrics["pooled"]["mae"]
    rmse = metrics["pooled"]["rmse"]
    per_subset = metrics["per_subset"]
    quantile = metrics["conformal"]["quantile"]
    coverage = metrics["conformal"]["coverage_application"]
    coverage_by_subset = metrics["conformal"]["coverage_by_subset"]
    alpha = metrics["conformal"]["alpha"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Pooled  — MAE {mae:.2f}  RMSE {rmse:.2f}  (n={metrics['pooled']['n']})")
    for name in SUBSETS:
        m = per_subset[name]
        print(f"{name:6} — RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  "
              f"NASA score {m['nasa_score']:.0f}  (n={m['n']})")

    print(f"\nConformal — alpha={alpha}, calibration n={metrics['conformal']['calibration_n']} "
          f"(last 20% of X_test), quantile=±{quantile:.1f}")
    print(f"Coverage on {metrics['conformal']['application_n']} non-calibration engines: {coverage * 100:.1f}%")
    for name in SUBSETS:
        c = coverage_by_subset[name]
        if c is None:
            print(f"  {name:6} — no non-calibration engines in this slice")
        else:
            print(f"  {name:6} — coverage {c['coverage'] * 100:.1f}%  (n={c['n']})")

    print(f"\nSaved metrics to {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
