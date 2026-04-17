import sys
import numpy as np
from pathlib import Path
from langchain_core.tools import Tool

sys.path.append(str(Path(__file__).parent.parent / "agents"))
sys.path.append(str(Path(__file__).parent.parent / "event_bus"))

from agent1_anomaly import AnomalyAgent
from agent2_rul import RULAgent
from agent4_decision import DecisionAgent
from agent5_orchestrator import OrchestratorAgent
from bus import bus
from events import AnomalyEvent, RULEvent, DecisionEvent, ReportEvent


MODEL_DIR = Path(__file__).parent.parent / "models"


# ── Load agents once ───────────────────────────────────────────
agent1 = AnomalyAgent(model_path=MODEL_DIR / "agent1_autoencoder.pt")
agent2 = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")
agent4 = DecisionAgent()
agent5 = OrchestratorAgent()


# ── Tool Functions ─────────────────────────────────────────────
async def run_anomaly_detection(data: dict) -> dict:
    engine_id     = data["engine_id"]
    sensor_window = data["sensor_window"]
    condition     = data["condition"]

    X = sensor_window[np.newaxis, ...]  # (1, 50, 16)
    C = np.array([condition])

    result = agent1.predict(X, C)[0]

    event = AnomalyEvent(
        engine_id            = engine_id,
        severity             = result["status"],
        reconstruction_error = result["reconstruction_error"],
        threshold            = result["threshold"]
    )

    await bus.publish("anomaly_detected", {
        "engine_id"            : engine_id,
        "severity"             : event.severity,
        "reconstruction_error" : event.reconstruction_error,
        "threshold"            : event.threshold,
        "sensor_window"        : sensor_window,
        "condition"            : condition
    })

    return {"severity": event.severity, "error": event.reconstruction_error}


async def run_rul_prediction(data: dict) -> dict:
    engine_id     = data["engine_id"]
    sensor_window = data["sensor_window"]

    X = sensor_window[np.newaxis, ...]  # (1, 50, 16)

    result = agent2.predict(X)[0]

    event = RULEvent(
        engine_id     = engine_id,
        predicted_RUL = result["predicted_RUL"],
        confidence    = result["confidence"]
    )

    await bus.publish("rul_predicted", {
        "engine_id"    : engine_id,
        "predicted_RUL": event.predicted_RUL,
        "confidence"   : event.confidence,
        "severity"     : data.get("severity", "NORMAL")
    })

    return {"predicted_RUL": event.predicted_RUL}


async def run_decision(data: dict) -> dict:
    engine_id     = data["engine_id"]
    severity      = data["severity"]
    predicted_RUL = data["predicted_RUL"]

    a1 = [{"engine_index": 0, "status": severity,
            "reconstruction_error": data.get("reconstruction_error", 0),
            "threshold": data.get("threshold", 0)}]
    a2 = [{"engine_index": 0, "predicted_RUL": predicted_RUL, "confidence": 0}]

    decision = agent4.predict(a1, a2)[0]

    event = DecisionEvent(
        engine_id     = engine_id,
        severity      = severity,
        predicted_RUL = predicted_RUL,
        decision      = decision["decision"],
        priority      = decision["priority"],
        action        = decision["action"]
    )

    await bus.publish("decision_made", {
        "engine_id"    : engine_id,
        "severity"     : event.severity,
        "predicted_RUL": event.predicted_RUL,
        "decision"     : event.decision,
        "priority"     : event.priority,
        "action"       : event.action
    })

    return {"decision": event.decision, "priority": event.priority}


async def run_report(data: dict) -> dict:
    engine_id = data["engine_id"]

    # only generate report for HIGH/CRITICAL engines to save API cost
    if data.get("priority") in ["CRITICAL", "HIGH"]:
        prompt   = f"Engine {engine_id}: RUL={data['predicted_RUL']:.1f} cycles, Severity={data['severity']}, Decision={data['decision']}. Give a 2 sentence maintenance recommendation."
        response = agent5.chat(prompt, [data])
    else:
        response = f"Engine {engine_id} — {data['decision']}. Action: {data['action']}"

    event = ReportEvent(
        engine_id = engine_id,
        report    = response,
        decision  = data["decision"],
        priority  = data["priority"]
    )

    bus.store_result(engine_id, {
        "engine_id"    : engine_id,
        "severity"     : data["severity"],
        "predicted_RUL": data["predicted_RUL"],
        "decision"     : data["decision"],
        "priority"     : data["priority"],
        "action"       : data["action"],
        "report"       : event.report
    })

    return {"report": event.report}


# ── LangChain Tools ────────────────────────────────────────────
anomaly_tool = Tool(
    name="AnomalyDetector",
    func=lambda x: x,
    coroutine=run_anomaly_detection,
    description="Detects anomalies in engine sensor data using LSTM Autoencoder"
)

rul_tool = Tool(
    name="RULPredictor",
    func=lambda x: x,
    coroutine=run_rul_prediction,
    description="Predicts Remaining Useful Life using CNN-BiLSTM"
)

decision_tool = Tool(
    name="DecisionEngine",
    func=lambda x: x,
    coroutine=run_decision,
    description="Makes maintenance decisions based on RUL and anomaly severity"
)

report_tool = Tool(
    name="ReportGenerator",
    func=lambda x: x,
    coroutine=run_report,
    description="Generates natural language maintenance report using Claude API"
)
