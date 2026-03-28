import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.append(str(Path(__file__).parent.parent / "Agents"))

from Agent_5_interface import OrchestratorAgent

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="TurbineAgent Dashboard",
    page_icon="✈️",
    layout="wide"
)

# ── Professional Industrial CSS ───────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Base ── */
html, body, [class*="css"], p, div, span, label {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background-color: #0d1117;
    background-image: none;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #010409 !important;
    border-right: 1px solid #21262d;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8b949e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 400;
    padding: 0.35rem 0;
    transition: color 0.15s ease;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #e6edf3 !important;
}

/* ── Page titles ── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.375rem !important;
    color: #e6edf3 !important;
    letter-spacing: -0.02em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.75rem;
    margin-bottom: 1.25rem !important;
}

h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #c9d1d9 !important;
    letter-spacing: -0.01em;
}

h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.25rem 0.875rem;
    transition: border-color 0.15s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #30363d;
}

[data-testid="stMetricLabel"] > div {
    color: #8b949e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stMetricValue"] > div {
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.75rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important;
    border-radius: 6px;
    overflow: hidden;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 6px;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem;
    border: none;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8125rem !important;
    font-weight: 500 !important;
    border-radius: 6px;
    padding: 0.375rem 1rem !important;
    transition: all 0.15s ease;
    letter-spacing: 0;
}

.stButton > button:hover {
    background-color: #30363d !important;
    border-color: #8b949e !important;
    color: #e6edf3 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    border-radius: 6px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px;
    margin-bottom: 0.5rem;
}

/* ── Divider ── */
hr {
    border-color: #21262d !important;
    margin: 1rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Section header ── */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.6875rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0 0 0.5rem 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1rem;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.6875rem;
    font-weight: 600;
    padding: 0.2rem 0.5rem;
    border-radius: 2rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
.badge-high     { background: rgba(227,146,74,0.15); color: #e3924a; border: 1px solid rgba(227,146,74,0.3); }
.badge-medium   { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
.badge-low      { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
.badge-none     { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }

/* ── Info card ── */
.info-card {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.info-card-label {
    font-size: 0.6875rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.info-card-value {
    font-size: 0.9375rem;
    font-weight: 500;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

RESULTS_FILE   = Path(__file__).parent / "live_results.json"
SEVERITY_COLOR = {"NORMAL": "#3fb950", "SUSPICIOUS": "#d29922", "CRITICAL": "#f85149"}
PRIORITY_COLOR = {"NONE": "#3fb950", "LOW": "#58a6ff", "MEDIUM": "#d29922", "HIGH": "#e3924a", "CRITICAL": "#f85149"}

# ── Load results from JSON ─────────────────────────────────────
def load_results():
    if not RESULTS_FILE.exists():
        return {"status": "waiting", "total": 0, "results": {}}
    try:
        with open(RESULTS_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return {"status": "waiting", "total": 0, "results": {}}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        return {"status": "waiting", "total": 0, "results": {}}

# ── Build DataFrame ───────────────────────────────────────────
def build_df(results: dict):
    if not results:
        return pd.DataFrame()
    rows = list(results.values())
    return pd.DataFrame(rows)

# ── Plotly dark theme ──────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161b22",
    font=dict(family="Inter", color="#8b949e", size=12),
    title_font=dict(family="Inter", color="#c9d1d9", size=13, weight=600),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", size=11)),
    xaxis=dict(gridcolor="#21262d", color="#8b949e", linecolor="#21262d", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262d", color="#8b949e", linecolor="#21262d", tickfont=dict(size=11)),
    margin=dict(t=40, b=30, l=10, r=10),
)

# ── Agent 5 (cached) ──────────────────────────────────────────
@st.cache_resource
def load_agent5():
    return OrchestratorAgent()

agent5 = load_agent5()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding: 1.25rem 0.5rem 1rem;">
    <div style="font-family:'Inter',sans-serif; font-weight:700; font-size:0.9375rem;
                color:#e6edf3; letter-spacing:-0.01em;">
        TurbineAgent
    </div>
    <div style="font-family:'Inter',sans-serif; font-size:0.6875rem; font-weight:400;
                color:#484f58; margin-top:2px; letter-spacing:0.01em;">
        NASA C-MAPSS Fleet Monitor
    </div>
    <div style="margin-top:1rem; height:1px; background:#21262d;"></div>
    <div style="font-family:'Inter',sans-serif; font-size:0.6875rem; font-weight:600;
                color:#484f58; text-transform:uppercase; letter-spacing:0.07em;
                margin-top:1rem; margin-bottom:0.5rem;">
        Navigation
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["Live Stream", "Fleet Overview", "Engine Detail", "RUL Analysis", "AI Chat"],
    label_visibility="collapsed"
)

st.sidebar.markdown("""
<div style="padding: 1rem 0.5rem; margin-top: 2rem; border-top: 1px solid #21262d;">
    <div style="font-family:'Inter',sans-serif; font-size:0.6875rem; color:#30363d; letter-spacing:0.02em;">
        Powered by Anthropic Claude Haiku
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 0 — Live Stream
# ══════════════════════════════════════════════════════════════
if page == "Live Stream":
    st.markdown("<h1>Live Engine Stream</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Real-time turbofan health monitoring</div>', unsafe_allow_html=True)

    placeholder = st.empty()

    while True:
        data    = load_results()
        results = data.get("results", {})
        status  = data.get("status", "waiting")
        total   = data.get("total", 0)

        with placeholder.container():
            if status == "waiting":
                st.info("STANDBY — Awaiting pipeline signal. Execute `python main.py` to initiate.")
            elif status == "running":
                st.success(f"PIPELINE ACTIVE — {total} engines processed")
            else:
                st.success(f"PIPELINE COMPLETE — {total} engines processed")

            if results:
                df = build_df(results)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("PROCESSED",  total)
                col2.metric("CRITICAL",   int((df["priority"] == "CRITICAL").sum()))
                col3.metric("HIGH",       int((df["priority"] == "HIGH").sum()))
                col4.metric("MEDIUM",     int((df["priority"] == "MEDIUM").sum()))
                col5.metric("HEALTHY",    int((df["priority"] == "NONE").sum()))

                st.markdown('<div class="section-header">Latest engines</div>', unsafe_allow_html=True)
                latest = df.tail(10)[["engine_id", "predicted_RUL", "severity", "decision", "priority"]].iloc[::-1]
                st.dataframe(latest, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    priority_counts = df["priority"].value_counts().reset_index()
                    priority_counts.columns = ["Priority", "Count"]
                    fig = px.pie(
                        priority_counts, values="Count", names="Priority",
                        title="PRIORITY DISTRIBUTION",
                        color="Priority", color_discrete_map=PRIORITY_COLOR,
                        hole=0.4
                    )
                    fig.update_traces(textfont=dict(family="Rajdhani"), textfont_color="white")
                    fig.update_layout(**PLOT_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.scatter(
                        df, x="engine_id", y="predicted_RUL",
                        color="severity", color_discrete_map=SEVERITY_COLOR,
                        title="RUL PER ENGINE",
                        labels={"engine_id": "Engine", "predicted_RUL": "RUL (cycles)"}
                    )
                    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
                    fig.update_layout(**PLOT_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                urgent = df[df["priority"].isin(["CRITICAL", "HIGH"])].sort_values("predicted_RUL")
                if not urgent.empty:
                    st.markdown(
                        f'<div class="section-header">Urgent engines — {len(urgent)} requiring immediate attention</div>',
                        unsafe_allow_html=True
                    )
                    st.dataframe(
                        urgent[["engine_id", "predicted_RUL", "severity", "decision", "priority", "action"]],
                        use_container_width=True
                    )

        if status == "complete":
            break

        time.sleep(1)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — Fleet Overview
# ══════════════════════════════════════════════════════════════
elif page == "Fleet Overview":
    st.markdown("<h1>Fleet Health Overview</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">NASA C-MAPSS turbofan fleet status</div>', unsafe_allow_html=True)

    data    = load_results()
    results = data.get("results", {})

    if not results:
        st.warning("No telemetry received. Execute `python main.py` to begin analysis.")
    else:
        df = build_df(results)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("TOTAL ENGINES",  len(df))
        col2.metric("HEALTHY",        int((df["priority"] == "NONE").sum()))
        col3.metric("MONITOR",        int((df["priority"] == "LOW").sum()))
        col4.metric("MAINTENANCE",    int((df["priority"].isin(["MEDIUM", "HIGH"])).sum()))
        col5.metric("CRITICAL",       int((df["priority"] == "CRITICAL").sum()))

        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            priority_counts = df["priority"].value_counts().reset_index()
            priority_counts.columns = ["Priority", "Count"]
            fig = px.pie(
                priority_counts, values="Count", names="Priority",
                title="FLEET PRIORITY DISTRIBUTION",
                color="Priority", color_discrete_map=PRIORITY_COLOR,
                hole=0.45
            )
            fig.update_traces(textfont=dict(family="Rajdhani"), textfont_color="white")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            severity_counts = df["severity"].value_counts().reset_index()
            severity_counts.columns = ["Severity", "Count"]
            fig = px.bar(
                severity_counts, x="Severity", y="Count",
                title="ANOMALY SEVERITY DISTRIBUTION",
                color="Severity", color_discrete_map=SEVERITY_COLOR
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Engines requiring attention</div>', unsafe_allow_html=True)
        urgent = df[df["priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])].sort_values("predicted_RUL")
        if not urgent.empty:
            st.dataframe(
                urgent[["engine_id", "predicted_RUL", "severity", "decision", "priority", "action"]],
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════
# PAGE 2 — Engine Detail
# ══════════════════════════════════════════════════════════════
elif page == "Engine Detail":
    st.markdown("<h1>Engine Detail View</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Individual engine diagnostics</div>', unsafe_allow_html=True)

    data    = load_results()
    results = data.get("results", {})

    if not results:
        st.warning("No telemetry received. Execute `python main.py` to begin analysis.")
    else:
        df        = build_df(results)
        engine_id = st.selectbox("SELECT ENGINE", df["engine_id"].tolist())
        engine    = df[df["engine_id"] == engine_id].iloc[0]

        priority = engine["priority"]
        color_map = {"CRITICAL": "#ff2244", "HIGH": "#ff6600", "MEDIUM": "#ffaa00", "LOW": "#00aaff", "NONE": "#00ff88"}
        eng_color = color_map.get(priority, "#00d4ff")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PREDICTED RUL",  f"{engine['predicted_RUL']:.1f} cycles")
        col2.metric("SEVERITY",       engine["severity"])
        col3.metric("PRIORITY",       engine["priority"])
        col4.metric("DECISION",       engine["decision"])

        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-header">Maintenance directive</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background: rgba(0,20,45,0.8); border: 1px solid rgba(0,212,255,0.2);
                        border-radius:8px; padding: 1.2rem; font-family:'Rajdhani',sans-serif;">
                <div style="color:#5a9ab5; font-size:0.7rem; letter-spacing:2px; margin-bottom:0.3rem;">DECISION</div>
                <div style="color:#00d4ff; font-size:1.1rem; margin-bottom:1rem;">{engine['decision']}</div>
                <div style="color:#5a9ab5; font-size:0.7rem; letter-spacing:2px; margin-bottom:0.3rem;">ACTION</div>
                <div style="color:#a0c4d8; font-size:1rem; margin-bottom:1rem;">{engine['action']}</div>
                <div style="color:#5a9ab5; font-size:0.7rem; letter-spacing:2px; margin-bottom:0.3rem;">PRIORITY</div>
                <div style="color:{eng_color}; font-size:1.2rem; font-weight:bold;
                            text-shadow: 0 0 10px {eng_color};">{priority}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=engine["predicted_RUL"],
                number={"font": {"family": "Orbitron", "color": "#00d4ff"}, "suffix": " cycles"},
                title={"text": "PREDICTED RUL", "font": {"family": "Orbitron", "color": "#00d4ff", "size": 12}},
                gauge={
                    "axis": {"range": [0, 125], "tickcolor": "#5a9ab5",
                             "tickfont": {"family": "Rajdhani", "color": "#5a9ab5"}},
                    "bar":  {"color": eng_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,212,255,0.2)",
                    "steps": [
                        {"range": [0, 20],   "color": "rgba(255,34,68,0.15)"},
                        {"range": [20, 50],  "color": "rgba(255,102,0,0.1)"},
                        {"range": [50, 125], "color": "rgba(0,255,136,0.08)"}
                    ],
                    "threshold": {
                        "line": {"color": "#ff2244", "width": 2},
                        "thickness": 0.8,
                        "value": 20
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Rajdhani", color="#a0c4d8"),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — RUL Analysis
# ══════════════════════════════════════════════════════════════
elif page == "RUL Analysis":
    st.markdown("<h1>RUL Prediction Analysis</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Remaining useful life — fleet analytics</div>', unsafe_allow_html=True)

    data    = load_results()
    results = data.get("results", {})

    if not results:
        st.warning("No telemetry received. Execute `python main.py` to begin analysis.")
    else:
        df = build_df(results)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x="predicted_RUL", nbins=30,
                title="RUL DISTRIBUTION",
                color_discrete_sequence=["#00d4ff"]
            )
            fig.update_traces(marker_line_color="rgba(0,212,255,0.3)", marker_line_width=1)
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                df, x="engine_id", y="predicted_RUL",
                color="severity", color_discrete_map=SEVERITY_COLOR,
                title="PREDICTED RUL PER ENGINE"
            )
            fig.update_traces(marker=dict(size=7))
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df["predicted_RUL"].values,
            name="Predicted RUL",
            line=dict(color="#00d4ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.05)"
        ))
        fig.add_hline(y=20, line_dash="dash", line_color="#ff2244",
                      annotation_text="CRITICAL THRESHOLD", annotation_font_color="#ff2244")
        fig.add_hline(y=50, line_dash="dash", line_color="#ffaa00",
                      annotation_text="MAINTENANCE THRESHOLD", annotation_font_color="#ffaa00")
        fig.update_layout(
            title="PREDICTED RUL — ALL ENGINES",
            xaxis_title="ENGINE INDEX",
            yaxis_title="RUL (CYCLES)",
            **PLOT_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — AI Chat
# ══════════════════════════════════════════════════════════════
elif page == "AI Chat":
    st.markdown("<h1>TurbineAgent AI</h1>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Claude-powered fleet intelligence interface</div>', unsafe_allow_html=True)

    data      = load_results()
    results   = data.get("results", {})
    decisions = list(results.values()) if results else []

    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found in .env file. Add it and restart the dashboard.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Query fleet status, engine health, or maintenance priorities..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing fleet telemetry..."):
                try:
                    response = agent5.chat(prompt, decisions)
                except Exception as e:
                    response = f"**Error:** {e}"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("CLEAR"):
            st.session_state.messages = []
            agent5.reset_conversation()
            st.rerun()
