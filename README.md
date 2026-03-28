# TurbineAgent — NASA C-MAPSS Fleet Health Monitor

> A production-grade multi-agent AI system for predictive maintenance of aircraft turbofan engines, built on the NASA C-MAPSS run-to-failure dataset.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=flat-square&logo=streamlit)
![Anthropic](https://img.shields.io/badge/Claude-Haiku-blueviolet?style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-Core-green?style=flat-square)

---

## Overview

TurbineAgent monitors a fleet of 707 turbofan engines in real time, running four specialised AI agents in a streaming event-driven pipeline. Each engine passes through anomaly detection, RUL prediction, and a decision engine before a Claude LLM generates a natural language maintenance report. Results are visualised on a live Streamlit dashboard.

**Key results:**
- RUL prediction MAE: **12.2 cycles** (CNN-BiLSTM)
- Anomaly detection: **26.9%** flagged, **39.6%** near-failure capture rate
- Fleet classification: 10 CRITICAL · 19 HIGH · 33 MEDIUM · 128 LOW · 517 HEALTHY

---

## Architecture

```
Raw Sensor Stream
       │
       ▼
  Event Bus (asyncio)
       │
   ┌───┴───┐
   │       │  (parallel)
   ▼       ▼
Agent 1  Agent 2
LSTM     CNN-BiLSTM
Auto-    RUL
encoder  Predictor
   │       │
   └───┬───┘
       ▼
    Agent 4
  Decision Engine
  (Rule-based)
       │
       ▼
    Agent 5
  Claude Haiku LLM
  (Fleet Orchestrator)
       │
       ▼
  Streamlit Dashboard
  (Live Streaming)
```

Agents 1 and 2 run **in parallel** via `asyncio.gather`. Results are merged in the Orchestrator before Agent 4 makes a maintenance decision. Agent 5 generates natural language reports only for HIGH/CRITICAL engines to minimise API cost.

---

## Agents

| Agent | Model | Task | Performance |
|-------|-------|------|-------------|
| **Agent 1** | LSTM Autoencoder | Anomaly detection via reconstruction error | Per-condition thresholds (75th percentile) |
| **Agent 2** | CNN-BiLSTM | Remaining Useful Life (RUL) prediction | MAE: 12.2 · RMSE: 17.6 cycles |
| **Agent 4** | Rule-based engine | Maintenance decision + priority classification | 5 priority levels |
| **Agent 5** | Claude Haiku (LLM) | Natural language fleet reporting + chat | Anthropic API |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch — LSTM, CNN, BiLSTM |
| Hyperparameter Tuning | Optuna |
| Agent Orchestration | asyncio event bus + LangChain Core tools |
| LLM | Anthropic Claude Haiku |
| Dashboard | Streamlit + Plotly |
| Data | NASA C-MAPSS FD001–FD004 (turbofan run-to-failure) |

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) contains run-to-failure simulations of turbofan engines across four sub-datasets (FD001–FD004) with varying operating conditions and fault modes.

| Split | Engines | Windows |
|-------|---------|---------|
| Train | 706 engines | 125,618 windows |
| Test  | 707 engines | 707 windows (last cycle) |

**Preprocessing:**
- Dropped 6 constant/low-variance sensors: `op3, s1, s5, s10, s16, s19`
- KMeans clustering for operating condition labels (FD002/FD004 only — 6 regimes)
- MinMaxScaler normalisation per condition
- Rolling mean smoothing on `s3`
- RUL capped at 125 cycles
- Sliding windows: 50 cycles, stride 1

---

## Project Structure

```
NASA_TURBOJET/
├── Agents/
│   ├── Agent_1_autoencoder.py   # LSTM Autoencoder — anomaly detection
│   ├── Agent_2_rul.py           # CNN-BiLSTM — RUL prediction
│   ├── Agent_4_descion.py       # Rule-based decision engine
│   └── Agent_5_interface.py     # Claude LLM orchestrator + chat
├── pipeline/
│   ├── orchestrator.py          # Event handler — coordinates all agents
│   └── tools.py                 # LangChain tool wrappers
├── event_bus/
│   ├── bus.py                   # asyncio publish/subscribe event bus
│   └── events.py                # Event dataclasses
├── dashboard/
│   ├── app.py                   # Streamlit dashboard (5 pages)
│   └── live_results.json        # Written by main.py, read by dashboard
├── models/
│   ├── agent1_autoencoder.pt    # Trained LSTM Autoencoder
│   └── agent2_rul_predictor.pt  # Trained CNN-BiLSTM
├── DATA/
│   ├── raw/                     # NASA C-MAPSS .txt files
│   ├── processed/               # Parquet files
│   └── model_ready/             # .npy arrays (X_train, X_test, etc.)
├── Notebook/
│   ├── EDA_Turbojet_engine.ipynb
│   └── Pre-processing.ipynb
├── Architecture/
│   ├── ARCHITECTURE.md
│   └── architecture.drawio
├── main.py                      # Entry point — runs streaming pipeline
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NASA_TURBOJET.git
cd NASA_TURBOJET

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Download the NASA C-MAPSS dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place the `.txt` files in `DATA/raw/`.

Run the preprocessing notebook: `Notebook/Pre-processing.ipynb`

---

## Running

**Option 1 — Full pipeline with live dashboard:**

```bash
# Terminal 1: run the streaming pipeline
python main.py

# Terminal 2: run the dashboard
python -m streamlit run dashboard/app.py
```

Open `http://localhost:8501` → select **Live Stream**.

**Option 2 — Dashboard only (view previous results):**

```bash
python -m streamlit run dashboard/app.py
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Live Stream** | Real-time engine feed, KPI cards, live charts — auto-refreshes every second |
| **Fleet Overview** | Priority/severity distribution, engines requiring attention |
| **Engine Detail** | Per-engine RUL gauge, maintenance directive, priority badge |
| **RUL Analysis** | RUL histogram, scatter plot, full fleet time series with thresholds |
| **AI Chat** | Conversational interface powered by Claude Haiku — ask about any engine or fleet status |

---

## Roadmap

- [x] LSTM Autoencoder — anomaly detection
- [x] CNN-BiLSTM — RUL prediction (MAE: 12.2 cycles)
- [x] Rule-based Decision Engine
- [x] Claude LLM Orchestrator + chat
- [x] asyncio event bus + LangChain tools
- [x] Live-streaming Streamlit dashboard
- [ ] SHAP explanations — sensor-level feature importance
- [ ] MAPIE conformal prediction — RUL confidence intervals
- [ ] MLflow experiment tracking
- [ ] FastAPI REST endpoints
- [ ] Docker containerisation
- [ ] pytest unit tests

---

## Model Details

**Agent 1 — LSTM Autoencoder**
- Architecture: LSTM encoder-decoder, hidden=64, layers=1, dropout=0.40
- Trained on normal engine windows only
- Threshold: 75th percentile of reconstruction error per operating condition
- Tuned with Optuna (15 trials)

**Agent 2 — CNN-BiLSTM**
- Architecture: Conv1D (filters=64, kernel=3) → BiLSTM (hidden=128, layers=2) → FC
- Dropout: 0.15 · Learning rate: 8.3×10⁻⁴
- Input: (batch, 50, 16) sensor windows
- RUL cap: 125 cycles
- Tuned with Optuna (15 trials)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
