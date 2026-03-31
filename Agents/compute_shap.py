"""
Run after main.py to compute per-engine SHAP values and enrich live_results.json
with top contributing sensors for each engine.

Usage:
    python compute_shap.py
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Agent_2_rul import RULAgent, FEATURE_NAMES

MODEL_DIR       = Path(__file__).parent.parent / "models"
MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"
RESULTS_FILE    = Path(__file__).parent.parent / "dashboard" / "live_results.json"


def main():
    if not RESULTS_FILE.exists():
        print("No live_results.json found. Run main.py first.")
        return

    print("Loading data and model...")
    X_test = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    agent  = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")

    print(f"Computing SHAP values for {len(X_test)} engines (takes 5-10 mins on CPU)...")
    mean_shap, feature_names, global_importance = agent.explain(X_test, background_size=100)

    print("Enriching results with top sensors per engine...")
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    for key, record in results.items():
        engine_id = record.get("engine_id")
        if engine_id is None:
            continue
        engine_id = int(engine_id)
        if engine_id >= len(mean_shap):
            continue

        shap_row = mean_shap[engine_id].flatten().tolist()

        # top 3 sensors by importance
        ranked = sorted(zip(feature_names, shap_row), key=lambda x: x[1], reverse=True)
        top3   = [{"sensor": name, "importance": round(score, 5)} for name, score in ranked[:3]]

        record["top_sensors"] = top3
        record["sensor_importance"] = {name: round(score, 5) for name, score in zip(feature_names, shap_row)}

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f)

    print("Done. live_results.json enriched with SHAP sensor data.")

    # print global ranking
    ranked_global = sorted(zip(feature_names, global_importance.tolist()), key=lambda x: x[1], reverse=True)
    print("\nGlobal sensor importance:")
    for name, score in ranked_global:
        bar = "█" * int(score * 500)
        print(f"  {name:10} {score:.5f}  {bar}")


if __name__ == "__main__":
    main()
