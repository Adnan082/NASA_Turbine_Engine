from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class NewDataEvent:
    event: str = "new_data"
    engine_id: int = 0
    sensor_window: np.ndarray = field(default_factory=lambda: np.array([]))
    condition: float = 0.0


@dataclass
class AnomalyEvent:
    event: str = "anomaly_detected"
    engine_id: int = 0
    severity: str = "NORMAL"
    reconstruction_error: float = 0.0
    threshold: float = 0.0


@dataclass
class RULEvent:
    event: str = "rul_predicted"
    engine_id: int = 0
    predicted_RUL: float = 0.0
    confidence: float = 0.0


@dataclass
class DecisionEvent:
    event: str = "decision_made"
    engine_id: int = 0
    severity: str = "NORMAL"
    predicted_RUL: float = 0.0
    decision: str = "HEALTHY"
    priority: str = "NONE"
    action: str = "No action required"


@dataclass
class ReportEvent:
    event: str = "report_generated"
    engine_id: int = 0
    report: str = ""
    decision: str = "HEALTHY"
    priority: str = "NONE"
