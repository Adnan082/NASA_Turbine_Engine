import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.append(str(Path(__file__).parent))

from agent1_anomaly import AnomalyAgent
from agent2_rul import RULAgent
from agent4_decision import DecisionAgent


class OrchestratorAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model  = "claude-haiku-4-5-20251001"  # fast + cheap for inference
        self.conversation_history = []

        self.system_prompt = """You are TurbineAgent, an expert AI assistant for aircraft turbofan engine health monitoring.
You analyze sensor data from NASA C-MAPSS turbofan engines and provide maintenance recommendations.

You have access to outputs from 3 specialized agents:
- Agent 1 (Anomaly Detector): Detects unusual sensor behavior using LSTM Autoencoder
- Agent 2 (RUL Predictor): Predicts Remaining Useful Life using CNN-BiLSTM (MAE: 12.2 cycles)
- Agent 4 (Decision Engine): Provides maintenance decisions based on RUL and anomaly severity

SHAP sensor importance data is available for each engine in the context under "top_sensors" and "sensor_importance".
When asked which sensors are causing degradation or failure, USE the SHAP data provided in the context.
Never say SHAP data is unavailable if it appears in the context — it will be listed as "Top contributing sensors (SHAP)".

Sensor names and their physical meaning:
- T24: Total temperature at LPC outlet
- T30: Total temperature at HPC outlet
- T50: Total temperature at LPT outlet (exhaust)
- P15: Total pressure at fan inlet
- P30: Total pressure at HPC outlet
- Nf: Physical fan speed
- Nc: Physical core speed
- Ps30: Static pressure at HPC outlet
- phi: Fuel flow to Ps30 ratio
- NRf: Corrected fan speed
- NRc: Corrected core speed
- BPR: Bypass ratio
- htBleed: Enthalpy bleed
- Nf_dmd: Demanded fan speed
- W31: HPT coolant bleed
- W32: LPT coolant bleed

Key facts:
- RUL is measured in flight cycles (1 cycle = 1 takeoff + cruise + landing)
- Severity levels: NORMAL, SUSPICIOUS, CRITICAL
- Priority levels: NONE, LOW, MEDIUM, HIGH, CRITICAL
- RUL cap is 125 cycles

Always be concise, professional, and focus on actionable insights."""

    def generate_fleet_report(self, decisions):
        priority_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
        for d in decisions:
            priority_counts[d["priority"]] += 1

        urgent = [d for d in decisions if d["priority"] in ["CRITICAL", "HIGH"]]
        urgent = sorted(urgent, key=lambda x: x["predicted_RUL"])[:5]

        urgent_str = "\n".join([
            f"  - Engine {d['engine_index']}: RUL={d['predicted_RUL']:.1f} cycles, "
            f"Severity={d['severity']}, Decision={d['decision']}"
            for d in urgent
        ])

        prompt = f"""Generate a concise fleet health report based on the following data:

Fleet Summary:
- Total engines monitored: {len(decisions)}
- CRITICAL priority: {priority_counts['CRITICAL']} engines
- HIGH priority:     {priority_counts['HIGH']} engines
- MEDIUM priority:   {priority_counts['MEDIUM']} engines
- LOW priority:      {priority_counts['LOW']} engines
- HEALTHY:           {priority_counts['NONE']} engines

Top urgent engines:
{urgent_str}

Write a professional 3-paragraph report covering:
1. Overall fleet health status
2. Engines requiring immediate attention
3. Recommended actions for maintenance team"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def chat(self, user_message, decisions):
        import re

        # fleet summary
        context = f"""Current fleet data ({len(decisions)} engines monitored):
- CRITICAL: {sum(1 for d in decisions if d.get('priority') == 'CRITICAL')}
- HIGH:     {sum(1 for d in decisions if d.get('priority') == 'HIGH')}
- MEDIUM:   {sum(1 for d in decisions if d.get('priority') == 'MEDIUM')}
- LOW:      {sum(1 for d in decisions if d.get('priority') == 'LOW')}
- HEALTHY:  {sum(1 for d in decisions if d.get('priority') == 'NONE')}"""

        # always include top 5 urgent engines
        urgent = sorted(
            [d for d in decisions if d.get('priority') in ['CRITICAL', 'HIGH']],
            key=lambda x: x.get('predicted_RUL', 999)
        )[:5]
        if urgent:
            context += "\n\nTop urgent engines:\n"
            for d in urgent:
                eid = d.get('engine_id', d.get('engine_index', '?'))
                context += (f"  Engine {eid}: RUL={d.get('predicted_RUL','N/A')} cycles, "
                            f"Severity={d.get('severity','N/A')}, Priority={d.get('priority','N/A')}, "
                            f"Action={d.get('action','N/A')}\n")

        # exact engine lookup — search all engines, not just urgent
        matched = None
        for d in decisions:
            raw = d.get("engine_id", d.get("engine_index", ""))
            eid_str = str(int(float(raw))) if raw != "" else ""
            if eid_str and re.search(rf"\b{re.escape(eid_str)}\b", user_message):
                matched = d
                break

        # fallback: also search by float string e.g. "5.0"
        if matched is None:
            for d in decisions:
                raw = d.get("engine_id", d.get("engine_index", ""))
                eid_float = str(float(raw)) if raw != "" else ""
                if eid_float and re.search(rf"\b{re.escape(eid_float)}\b", user_message):
                    matched = d
                    break

        if matched:
            eid = matched.get("engine_id", matched.get("engine_index", "unknown"))
            context += f"""
Engine {eid} full details:
- Predicted RUL: {matched.get('predicted_RUL', 'N/A')} cycles
- Anomaly severity: {matched.get('severity', 'N/A')}
- Decision: {matched.get('decision', 'N/A')}
- Priority: {matched.get('priority', 'N/A')}
- Action: {matched.get('action', 'N/A')}"""

            # include SHAP sensor data if available
            top_sensors = matched.get("top_sensors")
            if top_sensors:
                context += "\n- Top contributing sensors (SHAP):\n"
                for s in top_sensors:
                    context += f"    {s['sensor']}: importance={s['importance']}\n"

            sensor_imp = matched.get("sensor_importance")
            if sensor_imp:
                context += "- All sensor importances (SHAP):\n"
                ranked = sorted(sensor_imp.items(), key=lambda x: x[1], reverse=True)
                for name, score in ranked:
                    context += f"    {name}: {score}\n"

        context += f"\n\nUser question: {user_message}"

        self.conversation_history.append({"role": "user", "content": context})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=self.system_prompt,
            messages=self.conversation_history
        )

        reply = response.content[0].text
        self.conversation_history.append({"role": "assistant", "content": reply})

        return reply

    def reset_conversation(self):
        self.conversation_history = []


if __name__ == "__main__":
    MODEL_DIR       = Path(__file__).parent.parent / "models"
    MODEL_READY_DIR = Path(__file__).parent.parent / "DATA" / "model_ready"

    print("Loading data and models...")
    X_test    = np.load(MODEL_READY_DIR / "X_test.npy").astype("float32")
    cond_test = np.load(MODEL_READY_DIR / "cond_test.npy")

    # run all agents
    agent1    = AnomalyAgent(model_path=MODEL_DIR / "agent1_autoencoder.pt")
    agent2    = RULAgent(model_path=MODEL_DIR / "agent2_rul_predictor.pt")
    agent4    = DecisionAgent()

    a1_results = agent1.predict(X_test, cond_test)
    a2_results = agent2.predict(X_test)
    decisions  = agent4.predict(a1_results, a2_results)

    # initialize orchestrator
    agent5 = OrchestratorAgent()

    # generate fleet report
    print("\n" + "="*60)
    print("FLEET HEALTH REPORT")
    print("="*60)
    report = agent5.generate_fleet_report(decisions)
    print(report)

    # chat interface
    print("\n" + "="*60)
    print("TurbineAgent Chat Interface")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("="*60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            agent5.reset_conversation()
            print("Conversation reset.\n")
            continue
        elif not user_input:
            continue

        response = agent5.chat(user_input, decisions)
        print(f"\nTurbineAgent: {response}\n")
