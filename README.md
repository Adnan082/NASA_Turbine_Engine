# TurbineAgent — NASA C-MAPSS Fleet Health Monitor

> A production-grade multi-agent AI system for predictive maintenance of aircraft turbofan engines, built on the NASA C-MAPSS run-to-failure dataset.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=flat-square&logo=streamlit)
![Anthropic](https://img.shields.io/badge/Claude-Haiku-blueviolet?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.12%2B-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-29%2B-2496ED?style=flat-square&logo=docker)
![pytest](https://img.shields.io/badge/pytest-9.0%2B-green?style=flat-square&logo=pytest)

---

## Screenshots

| Fleet Overview | Engine Detail |
|---|---|
| ![Fleet](screenshots/fleet_overview.png) | ![Engine](screenshots/engine_detail.png) |

| RUL Analysis | AI Chat |
|---|---|
| ![RUL](screenshots/rul_analysis.png) | ![Chat](screenshots/ai_chat.png) |

---

## Overview

TurbineAgent monitors a fleet of 707 turbofan engines in real time, running four specialised AI agents in a streaming event-driven pipeline. Each engine passes through anomaly detection, RUL prediction, SHAP explanation, and a decision engine before a Claude LLM generates a natural language maintenance report. Results are visualised on a live Streamlit dashboard and exposed via a FastAPI REST API.

**Key results:**
- RUL prediction MAE: **12.2 cycles** (CNN-BiLSTM) — competitive with published SOTA
- Anomaly detection: **26.9%** flagged, **39.6%** near-failure capture rate
- Conformal prediction intervals: **RUL ± 33.3 cycles** at 90% coverage
- Fleet classification: 10 CRITICAL · 19 HIGH · 33 MEDIUM · 128 LOW · 517 HEALTHY
- Tested on **all 4 sub-datasets** (FD001–FD004) with per-condition normalisation

---

## How TurbineAgent Compares to Published Models

Most published work on NASA C-MAPSS only tests on FD001 (single condition, single fault). TurbineAgent tests on all four datasets with per-condition handling — a significantly harder and more realistic benchmark.

### RUL Prediction — MAE Comparison

```
Method                        FD001    FD002    FD003    FD004
────────────────────────────────────────────────────────────
LSTM (vanilla)                16.14    24.49    16.18    28.17
CNN                           18.45    30.29    19.82    29.16
BiLSTM                        15.20    22.10    14.90    25.40
Transformer                   13.90    21.50    13.40    22.80
─────────────────────────────────────────────────────────
TurbineAgent CNN-BiLSTM       12.20    —        —        —
(single combined test set)
────────────────────────────────────────────────────────────
```

> TurbineAgent MAE of **12.20** beats vanilla LSTM (16.14), CNN (18.45), and matches Transformer-level performance — without attention mechanisms.

### What Makes TurbineAgent Different

| Feature | Most Published Papers | TurbineAgent |
|---|---|---|
| Datasets tested | FD001 only | FD001 + FD002 + FD003 + FD004 |
| Operating conditions | 1 | 6 (KMeans clustered) |
| Fault modes | 1 | 2 |
| Explainability | None | SHAP GradientExplainer per engine |
| Uncertainty quantification | None | Conformal prediction (±33 cycles, 90%) |
| Deployment | Script | FastAPI + Docker + Streamlit |
| LLM integration | None | Claude Haiku — natural language reports + chat |
| Experiment tracking | None | MLflow |
| Testing | None | pytest unit tests |

---

## Architecture

```
Raw Sensor Stream  (707 engines × 50 cycles × 16 sensors)
        │
        ▼
  Event Bus (asyncio pub/sub)
        │
   ┌────┴────┐
   │         │  (parallel)
   ▼         ▼
Agent 1    Agent 2
LSTM       CNN-BiLSTM
Auto-      RUL Predictor
encoder    + SHAP Explain
   │         │
   └────┬────┘
        ▼
     Agent 4
   Decision Engine
   (Rule-based, 5 levels)
        │
        ▼
     Agent 5
   Claude Haiku LLM
   Fleet Orchestrator + Chat
        │
        ▼
  ┌─────┴──────┐
  │            │
  ▼            ▼
Streamlit   FastAPI
Dashboard   REST API
(5 pages)   (/fleet /engine /urgent /chat)
        │
        ▼
     MLflow
  Experiment Tracker
```

Agents 1 and 2 run **in parallel** via `asyncio.gather`. Results are merged before Agent 4 makes a maintenance decision. Agent 5 generates natural language reports only for HIGH/CRITICAL engines to minimise API cost.

---

## Agents

| Agent | Model | Task | Performance |
|-------|-------|------|-------------|
| **Agent 1** | LSTM Autoencoder | Anomaly detection via reconstruction error | Per-condition thresholds (75th percentile) |
| **Agent 2** | CNN-BiLSTM | RUL prediction + SHAP sensor importance | MAE: 12.2 · RMSE: 17.6 cycles |
| **Agent 4** | Rule-based engine | Maintenance decision + priority classification | 5 priority levels |
| **Agent 5** | Claude Haiku (LLM) | Natural language fleet reporting + interactive chat | Anthropic API |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch — LSTM, CNN, BiLSTM |
| Hyperparameter Tuning | Optuna (15 trials per model) |
| Explainability | SHAP GradientExplainer |
| Uncertainty | Split Conformal Prediction (custom, no library) |
| Agent Orchestration | asyncio event bus + LangChain Core tools |
| LLM | Anthropic Claude Haiku |
| Dashboard | Streamlit + Plotly |
| REST API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Testing | pytest |
| Containerisation | Docker + Docker Compose |
| Data | NASA C-MAPSS FD001–FD004 |

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) contains run-to-failure simulations of turbofan engines across four sub-datasets with varying operating conditions and fault modes.

| Sub-dataset | Train Engines | Test Engines | Conditions | Fault Modes |
|-------------|--------------|--------------|------------|-------------|
| FD001 | 100 | 100 | 1 | 1 |
| FD002 | 260 | 259 | 6 | 1 |
| FD003 | 100 | 100 | 1 | 2 |
| FD004 | 248 | 248 | 6 | 2 |
| **Total** | **708** | **707** | — | — |

**Preprocessing:**
- Dropped 6 constant/low-variance sensors: `op3, s1, s5, s10, s16, s19`
- KMeans clustering for operating condition labels (FD002/FD004 — 6 regimes)
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
│   ├── Agent_2_rul.py           # CNN-BiLSTM — RUL prediction + SHAP
│   ├── Agent_4_descion.py       # Rule-based decision engine
│   ├── Agent_5_interface.py     # Claude LLM orchestrator + chat
│   ├── compute_shap.py          # Post-processing: SHAP sensor importance
│   └── mapie.py                 # Post-processing: conformal prediction intervals
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
├── Test/
│   └── test_agents.py           # pytest unit tests (4 tests)
├── DATA/
│   ├── raw/                     # NASA C-MAPSS .txt files (not in repo)
│   ├── processed/               # Parquet files (not in repo)
│   └── model_ready/             # .npy arrays — X_train, X_test, y_train, y_test
├── Notebook/
│   ├── EDA_Turbojet_engine.ipynb
│   └── Pre-processing.ipynb
├── Architecture/
│   └── architecture.drawio
├── main.py                      # Entry point — runs full pipeline
├── api.py                       # FastAPI REST API
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Runs dashboard + API together
└── requirements.txt
```

---

## Setup

### Option 1 — Local

```bash
# 1. Clone
git clone https://github.com/Adnan082/NASA_Turbine_Engine.git
cd NASA_Turbine_Engine

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Download the NASA C-MAPSS dataset from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) and place `.txt` files in `DATA/raw/`. Then run `Notebook/Pre-processing.ipynb`.

### Option 2 — Docker

```bash
docker compose up
```

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000/docs`

### Option 3 — Deploy to Render (free)

1. Fork this repo
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your forked repo
4. Add environment variable: `ANTHROPIC_API_KEY=sk-ant-...`
5. Set start command: `uvicorn api:app --host 0.0.0.0 --port 8000`
6. Deploy → get a public URL for the REST API

For the dashboard, create a second Render service with start command:
`streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0`

---

## Running

```bash
# Run full pipeline (agents + SHAP + MAPIE + MLflow)
python main.py

# Launch dashboard
python -m streamlit run dashboard/app.py

# Launch REST API
uvicorn api:app --reload

# Run tests
python -m pytest Test/ -v

# View MLflow experiment runs
mlflow ui
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/fleet` | All 707 engines + priority summary |
| GET | `/engine/{id}` | Single engine full details |
| GET | `/urgent` | CRITICAL + HIGH engines sorted by RUL |
| POST | `/chat` | Claude AI chat — ask about any engine |

Interactive docs: `http://localhost:8000/docs`

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Live Stream** | Real-time engine feed, KPI cards, live charts |
| **Fleet Overview** | Priority/severity distribution, engines requiring attention |
| **Engine Detail** | RUL gauge, confidence interval (±33 cycles), SHAP sensors, maintenance directive |
| **RUL Analysis** | RUL histogram, scatter, full fleet time series with confidence band |
| **AI Chat** | Claude Haiku — ask about any engine or fleet status |

---

## Model Details

**Agent 1 — LSTM Autoencoder**
- Architecture: LSTM encoder-decoder, hidden=64, layers=1, dropout=0.40
- Trained on normal engine windows only (first 30% of each engine's life)
- Threshold: 75th percentile of reconstruction error per operating condition
- Tuned with Optuna (15 trials)

**Agent 2 — CNN-BiLSTM**
- Architecture: Conv1D (filters=64, kernel=3) → BiLSTM (hidden=128, layers=2) → FC
- Dropout: 0.15 · Learning rate: 8.3×10⁻⁴
- Input: (batch, 50, 16) sensor windows · RUL cap: 125 cycles
- Tuned with Optuna (15 trials)
- SHAP GradientExplainer — top 3 sensors per engine logged to `live_results.json`
- Conformal prediction — 90% coverage intervals via split conformal (no external library)

**Experiments tried and rejected:**
- Graph Attention Network (GAT) autoencoder — separation ratio 0.99x (no signal). C-MAPSS degradation is temporal, not spatial — LSTM stays as Agent 1.

---

## Roadmap

- [x] LSTM Autoencoder — anomaly detection
- [x] CNN-BiLSTM — RUL prediction (MAE: 12.2 cycles)
- [x] Rule-based Decision Engine
- [x] Claude LLM Orchestrator + chat
- [x] asyncio event bus + LangChain tools
- [x] Live-streaming Streamlit dashboard
- [x] SHAP explanations — sensor-level feature importance
- [x] Conformal prediction — RUL confidence intervals (±33 cycles, 90%)
- [x] MLflow experiment tracking
- [x] FastAPI REST endpoints
- [x] Docker containerisation
- [x] pytest unit tests
- [ ] Agent 3 — XGBoost emissions predictor
- [ ] Cloud deployment (AWS/GCP)
- [ ] CI/CD pipeline (GitHub Actions)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
