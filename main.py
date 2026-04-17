import asyncio
import numpy as np
import sys
import json
import mlflow
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "event_bus"))
sys.path.append(str(Path(__file__).parent / "pipeline"))
sys.path.append(str(Path(__file__).parent / "agents"))

from bus import bus  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402

RESULTS_FILE = Path(__file__).parent / "dashboard" / "live_results.json"


def save_results_to_file():
    results = bus.get_all_results()
    serializable = {}
    for k, v in results.items():
        row = {key: (float(val) if isinstance(val, (int, float)) else val)
               for key, val in v.items()
               if key != "sensor_window"}
        serializable[str(k)] = row

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "status"  : "running",
            "total"   : len(serializable),
            "results" : serializable
        }, f)


async def simulate_stream(X_test: np.ndarray, cond_test: np.ndarray, delay: float = 0.1):
    print(f"\nStarting TurbineAgent pipeline...")
    print(f"Simulating {len(X_test)} engines at {delay}s interval\n")
    print("=" * 75)

    # reset results file
    with open(RESULTS_FILE, "w") as f:
        json.dump({"status": "running", "total": 0, "results": {}}, f)

    for engine_id in range(len(X_test)):
        await bus.publish("new_data", {
            "engine_id"    : engine_id,
            "sensor_window": X_test[engine_id],
            "condition"    : float(cond_test[engine_id])
        })
        await asyncio.sleep(delay)

        # write results after every engine
        save_results_to_file()

    # wait for all events to finish processing
    await asyncio.sleep(1.0)

    # mark as complete
    results = bus.get_all_results()
    serializable = {}
    for k, v in results.items():
        row = {key: (float(val) if isinstance(val, (int, float)) else val)
               for key, val in v.items()
               if key != "sensor_window"}
        serializable[str(k)] = row

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "status"  : "complete",
            "total"   : len(serializable),
            "results" : serializable
        }, f)

    print("\nResults saved to dashboard/live_results.json")


async def main():
    MODEL_READY_DIR = Path(__file__).parent / "DATA" / "model_ready"

    # load test data
    print("Loading test data...")
    X_test    = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    cond_test = np.load(MODEL_READY_DIR / "cond_test.npy")
    print(f"Loaded {len(X_test)} engines\n")

    # initialize orchestrator — subscribes all agents to event bus
    orchestrator = Orchestrator()

    # simulate streaming data
    await simulate_stream(X_test, cond_test, delay=0.05)

    # print final summary
    results = bus.get_all_results()
    print("\n" + "=" * 75)
    print("PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 75)

    priority_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for r in results.values():
        priority_counts[r["priority"]] += 1

    print(f"\nTotal engines processed: {len(results)}")
    for priority, count in priority_counts.items():
        print(f"  {priority:10} : {count}")

    # show urgent engines
    urgent = [r for r in results.values() if r["priority"] in ["CRITICAL", "HIGH"]]
    urgent = sorted(urgent, key=lambda x: x["predicted_RUL"])

    print(f"\nUrgent engines ({len(urgent)}):")
    for r in urgent[:10]:
        print(f"  Engine {r['engine_id']:3} | RUL: {r['predicted_RUL']:6.1f} | "
              f"Severity: {r['severity']:10} | Decision: {r['decision']}")

    # ── MLflow tracking ───────────────────────────────────────
    mlflow.set_experiment("TurbineAgent Fleet Monitor")
    with mlflow.start_run():
        ruls = [r["predicted_RUL"] for r in results.values()]
        mlflow.log_metric("total_engines",    len(results))
        mlflow.log_metric("critical_engines", priority_counts["CRITICAL"])
        mlflow.log_metric("high_engines",     priority_counts["HIGH"])
        mlflow.log_metric("healthy_engines",  priority_counts["NONE"])
        mlflow.log_metric("avg_predicted_RUL", round(float(np.mean(ruls)), 2))
        mlflow.log_metric("min_predicted_RUL", round(float(np.min(ruls)), 2))
        mlflow.log_metric("agent2_mae",  12.20)
        mlflow.log_metric("agent2_rmse", 17.58)
        mlflow.log_param("model", "CNN-BiLSTM")
        mlflow.log_param("dataset", "NASA C-MAPSS FD001-FD004")
        mlflow.log_param("rul_cap", 125)
        print("\nMLflow run logged.")

    # ── Post-processing: SHAP + MAPIE ─────────────────────────
    sys.path.append(str(Path(__file__).parent))
    from agents.mapie import main as run_mapie
    from agents.compute_shap import main as run_shap

    print("\n" + "=" * 75)
    print("Running MAPIE confidence intervals...")
    run_mapie()

    print("\n" + "=" * 75)
    print("Running SHAP sensor importance (this takes a few minutes)...")
    run_shap()


if __name__ == "__main__":
    asyncio.run(main())
