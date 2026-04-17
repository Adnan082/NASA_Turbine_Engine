"""
TurbineAgent REST API
Run with: uvicorn api:app --reload

Endpoints:
    GET  /fleet          - all engine results + summary
    GET  /engine/{id}    - single engine details
    GET  /urgent         - CRITICAL + HIGH engines only
    POST /chat           - Claude AI chat
"""
import json
import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.append(str(Path(__file__).parent / "agents"))

from agent5_orchestrator import OrchestratorAgent

RESULTS_FILE = Path(__file__).parent / "dashboard" / "live_results.json"

app = FastAPI(
    title="TurbineAgent API",
    description="NASA C-MAPSS turbofan engine health monitoring API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

agent5 = OrchestratorAgent()


def load_results():
    if not RESULTS_FILE.exists():
        return {}
    try:
        with open(RESULTS_FILE) as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {}


class ChatRequest(BaseModel):
    message: str


# ── GET /fleet ─────────────────────────────────────────────────
@app.get("/fleet")
def get_fleet():
    data    = load_results()
    results = data.get("results", {})

    if not results:
        raise HTTPException(status_code=503, detail="No results yet. Run main.py first.")

    engines = list(results.values())
    priority_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0}
    for e in engines:
        priority_counts[e["priority"]] += 1

    return {
        "status"         : data.get("status"),
        "total_engines"  : len(engines),
        "priority_counts": priority_counts,
        "engines"        : engines
    }


# ── GET /engine/{id} ───────────────────────────────────────────
@app.get("/engine/{engine_id}")
def get_engine(engine_id: int):
    data    = load_results()
    results = data.get("results", {})

    record = results.get(str(engine_id))
    if record is None:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found.")

    return record


# ── GET /urgent ────────────────────────────────────────────────
@app.get("/urgent")
def get_urgent():
    data    = load_results()
    results = data.get("results", {})

    urgent = [
        e for e in results.values()
        if e.get("priority") in ["CRITICAL", "HIGH"]
    ]
    urgent = sorted(urgent, key=lambda x: x.get("predicted_RUL", 999))

    return {
        "count"  : len(urgent),
        "engines": urgent
    }


# ── POST /chat ─────────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest):
    data      = load_results()
    decisions = list(data.get("results", {}).values())

    if not decisions:
        raise HTTPException(status_code=503, detail="No results yet. Run main.py first.")

    reply = agent5.chat(request.message, decisions)
    return {"response": reply}
