"""
Calibrate and evaluate the v2 anomaly detector (see FIX_REPORT.md, Phase 4).

The model (models/v2/agent1_autoencoder.pt, from scripts/train_anomaly.py on
early-life-only windows) already carries in-sample thresholds computed on its
own training data — this script replaces them with thresholds chosen for a
5% false-alarm rate on the held-out early-life windows from the calibration
engines (DATA/model_ready/X_early_cal.npy), then evaluates once on the 707
test engines.

Regimes are unique across sub-datasets ("FD002_r3", not just "3") — see
scripts/prepare_anomaly_v2_data.py.

Usage:
    python scripts/evaluate_anomaly_v2.py

Requires models/v2/agent1_autoencoder.pt and
DATA/model_ready/{X,regime}_early_cal.npy. Writes reports/v2/anomaly_metrics.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT / "agents"))

from agent1_anomaly import AnomalyAgent  # noqa: E402

MODEL_DIR = ROOT / "models"
MODEL_READY_DIR = ROOT / "DATA" / "model_ready"
REPORTS_DIR = ROOT / "reports" / "v2"

TARGET_FALSE_ALARM_RATE = 0.05
# Non-overlapping bands (RUL is capped at 125, so the top band is finite in practice).
RUL_BANDS = [("<=30", 0, 30), ("31-60", 31, 60), ("61-100", 61, 100), (">100", 101, np.inf)]


def require_inputs():
    v2_model = MODEL_DIR / "v2" / "agent1_autoencoder.pt"
    required = [
        v2_model, MODEL_READY_DIR / "X_early_cal.npy", MODEL_READY_DIR / "regime_early_cal.npy",
        MODEL_READY_DIR / "X_test.npy", MODEL_READY_DIR / "subset_test.npy", MODEL_READY_DIR / "cond_test.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}. Run scripts/prepare_anomaly_v2_data.py and "
            "scripts/train_anomaly.py (writing to models/v2/) first."
        )
    return v2_model


def composite_test_regimes():
    subset_test = np.load(MODEL_READY_DIR / "subset_test.npy")
    cond_test = np.load(MODEL_READY_DIR / "cond_test.npy")
    return np.array([f"{s}_r{int(c)}" for s, c in zip(subset_test, cond_test)])


def calibrate_thresholds(agent, X_early_cal, regime_early_cal, target_rate):
    errors = agent.get_reconstruction_errors(X_early_cal)
    thresholds = {}
    for regime in np.unique(regime_early_cal):
        mask = regime_early_cal == regime
        thresholds[regime] = float(np.percentile(errors[mask], (1 - target_rate) * 100))
    return thresholds


def main():
    v2_model_path = require_inputs()

    agent = AnomalyAgent(model_path=v2_model_path)

    X_early_cal = np.load(MODEL_READY_DIR / "X_early_cal.npy").astype("float32")
    regime_early_cal = np.load(MODEL_READY_DIR / "regime_early_cal.npy")

    print(f"Calibrating thresholds for a {TARGET_FALSE_ALARM_RATE:.0%} false-alarm rate "
          f"on {len(X_early_cal)} held-out early-life calibration windows...")
    thresholds = calibrate_thresholds(agent, X_early_cal, regime_early_cal, TARGET_FALSE_ALARM_RATE)
    agent.thresholds = thresholds
    for regime, t in sorted(thresholds.items()):
        print(f"  {regime}: {t:.6f}")

    # Sanity check: the false-alarm rate actually achieved on the calibration
    # windows used to set the threshold (in-sample by construction).
    cal_errors = agent.get_reconstruction_errors(X_early_cal)
    cal_flagged = np.array([
        cal_errors[i] > thresholds[regime_early_cal[i]] for i in range(len(cal_errors))
    ])
    achieved_far = float(cal_flagged.mean())

    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")
    regime_test = composite_test_regimes()

    results = agent.predict(X_test, regime_test)
    flagged = np.array([r["status"] != "NORMAL" for r in results])

    low_rul = y_test <= 30
    tp = int((flagged & low_rul).sum())
    fp = int((flagged & ~low_rul).sum())
    fn = int((~flagged & low_rul).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    by_band = {}
    for name, lo, hi in RUL_BANDS:
        mask = (y_test >= lo) & (y_test <= hi) if hi != np.inf else (y_test >= lo)
        by_band[name] = {"n": int(mask.sum()), "flag_rate": float(flagged[mask].mean()) if mask.sum() else None}

    metrics = {
        "target_false_alarm_rate": TARGET_FALSE_ALARM_RATE,
        "achieved_false_alarm_rate_on_calibration_windows": achieved_far,
        "n_calibration_windows": len(X_early_cal),
        "thresholds_by_regime": thresholds,
        "test_flag_rate_by_rul_band": by_band,
        "rul_le_30_precision": precision,
        "rul_le_30_recall": recall,
        "n_test_engines": len(y_test),
        "n_flagged": int(flagged.sum()),
        "v1_comparison": {
            "note": "v1 flagged 39.6% of RUL<=30 engines and 19.5% of RUL>=100 engines "
                    "(75th-percentile in-sample threshold, all-life training) — see README Known limitations.",
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "anomaly_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAchieved false-alarm rate on calibration windows: {achieved_far * 100:.1f}% "
          f"(target {TARGET_FALSE_ALARM_RATE * 100:.0f}%)")
    print(f"\nTest set — {flagged.sum()} / {len(y_test)} flagged ({flagged.mean() * 100:.1f}%)")
    for name, _, _ in RUL_BANDS:
        b = by_band[name]
        print(f"  RUL {name:6} — flag rate {b['flag_rate'] * 100:.1f}%  (n={b['n']})" if b["flag_rate"] is not None
              else f"  RUL {name:6} — n=0")
    print(f"\nRUL<=30 — precision {precision:.2f}  recall {recall:.2f}" if precision is not None
          else "\nRUL<=30 — no engines flagged")

    print(f"\nSaved metrics to {REPORTS_DIR / 'anomaly_metrics.json'}")


if __name__ == "__main__":
    main()
