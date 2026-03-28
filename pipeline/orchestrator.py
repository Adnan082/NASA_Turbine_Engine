import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "event_bus"))

from bus import bus
from tools import (
    run_anomaly_detection,
    run_rul_prediction,
    run_decision,
    run_report
)


class Orchestrator:
    def __init__(self):
        # subscribe each handler to its event
        bus.subscribe("new_data",        self.handle_new_data)
        bus.subscribe("anomaly_detected", self.handle_anomaly)
        bus.subscribe("rul_predicted",    self.handle_rul)
        bus.subscribe("decision_made",    self.handle_decision)

        self._anomaly_cache = {}
        self._rul_cache     = {}

    async def handle_new_data(self, data: dict):
        engine_id = data["engine_id"]
        print(f"[Engine {engine_id:3}] Data received — running Agent 1 + Agent 2...")

        # run Agent 1 and Agent 2 in parallel
        await asyncio.gather(
            run_anomaly_detection(data),
            run_rul_prediction(data)
        )

    async def handle_anomaly(self, data: dict):
        engine_id = data["engine_id"]
        self._anomaly_cache[engine_id] = data

        # check if RUL result is already available
        if engine_id in self._rul_cache:
            await self._run_decision(engine_id)

    async def handle_rul(self, data: dict):
        engine_id = data["engine_id"]
        self._rul_cache[engine_id] = data

        # check if anomaly result is already available
        if engine_id in self._anomaly_cache:
            await self._run_decision(engine_id)

    async def _run_decision(self, engine_id: int):
        anomaly = self._anomaly_cache.pop(engine_id)
        rul     = self._rul_cache.pop(engine_id)

        merged = {
            "engine_id"            : engine_id,
            "severity"             : anomaly["severity"],
            "reconstruction_error" : anomaly["reconstruction_error"],
            "threshold"            : anomaly["threshold"],
            "predicted_RUL"        : rul["predicted_RUL"],
            "confidence"           : rul["confidence"],
            "sensor_window"        : anomaly.get("sensor_window"),
            "condition"            : anomaly.get("condition", 0)
        }

        await run_decision(merged)

    async def handle_decision(self, data: dict):
        engine_id = data["engine_id"]
        print(f"[Engine {engine_id:3}] {data['severity']:10} | "
              f"RUL: {data['predicted_RUL']:6.1f} | "
              f"Decision: {data['decision']:25} | "
              f"Priority: {data['priority']}")

        await run_report(data)
