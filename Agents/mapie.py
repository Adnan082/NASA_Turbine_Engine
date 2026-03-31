"""
Run after main.py to compute conformal prediction intervals.
Enriches live_results.json with rul_lower, rul_upper, rul_ci per engine.

Usage:
    python Agents/mapie.py
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Agent_2_rul import RULAgent

MODEL_DIR       = Path(__file__).parent.parent / "models"
MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"
RESULTS_FILE    = Path(__file__).parent.parent / "dashboard" / "live_results.json"


def main():
    if not RESULTS_FILE.exists():
        print("No live_results.json found. Run main.py first.")
        return

    print("Loading data and model...")
    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    y_test = np.load(MODEL_READY_DIR / "y_test.npy").astype("float32")

    agent = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")

    # Split: last 20% = calibration, first 80% = application
    split = int(len(X_test) * 0.8)
    X_cal, y_cal = X_test[split:], y_test[split:]

    print(f"Calibration set: {len(X_cal)} engines")
    print(f"Full test set:   {len(X_test)} engines")

    # Step 3 — compute conformal quantile from calibration set
    print("Running predictions on calibration set...")
    cal_results = agent.predict(X_cal)
    cal_preds   = np.array([r["predicted_RUL"] for r in cal_results])

    residuals = np.abs(cal_preds - y_cal)
    alpha     = 0.10  # 90% coverage
    quantile  = np.quantile(residuals, 1 - alpha)

    print(f"Conformal quantile (90% coverage): ±{quantile:.1f} cycles")
    print(f"Calibration MAE: {residuals.mean():.2f} cycles")

    # Step 4 — apply intervals to all engines and enrich live_results.json
    print("Running predictions on full test set...")
    all_results = agent.predict(X_test)

    print("Enriching live_results.json with confidence intervals...")
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    for record in results.values():
        engine_id = record.get("engine_id")
        if engine_id is None:
            continue
        engine_id = int(engine_id)
        if engine_id >= len(all_results):
            continue

        rul = all_results[engine_id]["predicted_RUL"]
        lower = round(max(0.0, rul - quantile), 1)
        upper = round(min(125.0, rul + quantile), 1)

        record["rul_lower"] = lower
        record["rul_upper"] = upper
        record["rul_ci"]    = round(quantile, 1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f)

    print(f"Done. All engines enriched with ±{quantile:.1f} cycle confidence intervals.")


if __name__ == "__main__":
    main()
