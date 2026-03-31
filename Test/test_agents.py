import sys
import json
import pytest
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "Agents"))

from Agent_4_descion import DecisionAgent


def test_decision_critical():
    """Engine with low RUL and CRITICAL severity should get CRITICAL priority."""
    agent = DecisionAgent()
    a1 = [{"engine_index": 0, "status": "CRITICAL", "anomaly_score": 0.9, "reconstruction_error": 0.9}]
    a2 = [{"engine_index": 0, "predicted_RUL": 5.0,   "confidence": 12.0}]
    result = agent.predict(a1, a2)
    assert result[0]["priority"] == "CRITICAL"


def test_decision_healthy():
    """Engine with high RUL and NORMAL severity should get NONE priority."""
    agent = DecisionAgent()
    a1 = [{"engine_index": 0, "status": "NORMAL", "anomaly_score": 0.01, "reconstruction_error": 0.01}]
    a2 = [{"engine_index": 0, "predicted_RUL": 120.0, "confidence": 12.0}]
    result = agent.predict(a1, a2)
    assert result[0]["priority"] == "NONE"


def test_rul_prediction_range():
    """All RUL predictions should be between 0 and 125 cycles."""
    from Agent_2_rul import RULAgent

    MODEL_DIR       = Path(__file__).parent.parent / "models"
    MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"

    X_test  = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    agent   = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")
    results = agent.predict(X_test[:10])

    for r in results:
        assert 0 <= r["predicted_RUL"] <= 125


def test_mapie_intervals_valid():
    """MAPIE lower bound must be <= predicted RUL <= upper bound."""
    RESULTS_FILE = Path(__file__).parent.parent / "dashboard" / "live_results.json"

    if not RESULTS_FILE.exists():
        pytest.skip("live_results.json not found — run main.py first")

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    for record in data.get("results", {}).values():
        if "rul_lower" in record:
            assert record["rul_lower"] <= record["predicted_RUL"] <= record["rul_upper"]
