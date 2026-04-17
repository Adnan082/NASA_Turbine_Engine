from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent))

from Agent_1_autoencoder import AnomalyAgent
from Agent_2_rul import RULAgent


class DecisionAgent:
    def __init__(self):
        # Decision rules based on RUL + anomaly severity
        self.rules = [
            # (max_rul, severity,      decision,              priority, action)
            (10,  "CRITICAL",   "GROUND ENGINE",           "CRITICAL", "Remove from service immediately"),
            (10,  "SUSPICIOUS", "URGENT MAINTENANCE",      "HIGH",     "Schedule inspection within 24 hours"),
            (20,  "CRITICAL",   "URGENT MAINTENANCE",      "HIGH",     "Schedule inspection within 48 hours"),
            (20,  "SUSPICIOUS", "PLAN MAINTENANCE",        "MEDIUM",   "Schedule maintenance within 1 week"),
            (50,  "CRITICAL",   "PLAN MAINTENANCE",        "MEDIUM",   "Schedule maintenance within 2 weeks"),
            (50,  "SUSPICIOUS", "MONITOR",                 "LOW",      "Increase monitoring frequency"),
            (100, "CRITICAL",   "MONITOR",                 "LOW",      "Watch closely next 10 cycles"),
            (100, "SUSPICIOUS", "MONITOR",                 "LOW",      "Check again in 20 cycles"),
            (125, "CRITICAL",   "INVESTIGATE",             "LOW",      "Inspect sensor readings"),
            (125, "SUSPICIOUS", "MONITOR",                 "LOW",      "Routine monitoring"),
            (125, "NORMAL",     "HEALTHY",                 "NONE",     "No action required"),
        ]

    def decide(self, rul, severity):
        for max_rul, sev, decision, priority, action in self.rules:
            if rul <= max_rul and severity == sev:
                return decision, priority, action

        return "HEALTHY", "NONE", "No action required"

    def predict(self, agent1_results, agent2_results):
        decisions = []

        for r1, r2 in zip(agent1_results, agent2_results):
            rul      = r2["predicted_RUL"]
            severity = r1["status"]

            decision, priority, action = self.decide(rul, severity)

            decisions.append({
                "engine_index"         : r1["engine_index"],
                "predicted_RUL"        : rul,
                "severity"             : severity,
                "decision"             : decision,
                "priority"             : priority,
                "action"               : action,
                "reconstruction_error" : r1["reconstruction_error"]
            })

        return decisions


if __name__ == "__main__":
    MODEL_DIR       = Path(__file__).parent.parent / "models"
    MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"

    # load data
    X_test    = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    cond_test = np.load(MODEL_READY_DIR / "cond_test.npy")

    # run Agent 1
    agent1         = AnomalyAgent(model_path=MODEL_DIR / "agent1_autoencoder.pt")
    agent1_results = agent1.predict(X_test, cond_test)

    # run Agent 2
    agent2         = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")
    agent2_results = agent2.predict(X_test)

    # run Agent 4
    agent4         = DecisionAgent()
    decisions      = agent4.predict(agent1_results, agent2_results)

    # summary
    priority_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for d in decisions:
        priority_counts[d["priority"]] += 1

    print(f"Total engines:  {len(decisions)}")
    print(f"\nPriority Summary:")
    for priority, count in priority_counts.items():
        print(f"  {priority:10} : {count}")

    # show top 5 urgent engines
    urgent = [d for d in decisions if d["priority"] in ["CRITICAL", "HIGH"]]
    urgent = sorted(urgent, key=lambda x: x["predicted_RUL"])

    print(f"\nTop urgent engines:")
    for d in urgent[:5]:
        print(f"  Engine {d['engine_index']:3} | RUL: {d['predicted_RUL']:6.1f} | "
              f"Severity: {d['severity']:10} | Decision: {d['decision']:25} | Action: {d['action']}")
