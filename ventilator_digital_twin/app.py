from __future__ import annotations

import time
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scenario_engine import ScenarioControls, ScenarioEngine


def _init_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = ScenarioEngine(dt_s=0.2)
    if "run" not in st.session_state:
        st.session_state.run = True
    if "steps_per_refresh" not in st.session_state:
        st.session_state.steps_per_refresh = 5
    if "oxygen_supply_ok" not in st.session_state:
        st.session_state.oxygen_supply_ok = True


def _telemetry_gauges(latest: Dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("PIP (cmH2O)", f"{latest.get('pip_cmH2O', 0.0):.1f}")
    c1.metric("Plateau (cmH2O)", f"{latest.get('plateau_cmH2O', 0.0):.1f}")

    c2.metric("PEEP (cmH2O)", f"{latest.get('peep_cmH2O', 0.0):.1f}")
    c2.metric("Tidal Volume (L)", f"{latest.get('tidal_volume_L', 0.0):.2f}")

    c3.metric("Flow (L/s)", f"{latest.get('flow_Lps', 0.0):.2f}")
    c3.metric("RR (bpm)", f"{latest.get('rr_bpm', 0.0):.0f}")

    c4.metric("FiO2", f"{latest.get('fio2', 0.0):.2f}")
    c4.metric("Blower Speed (%)", f"{latest.get('blower_speed_pct', 0.0):.0f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("SpO2 (%)", f"{latest.get('spo2_pct', 0.0):.1f}")
    c6.metric("EtCO2 (mmHg)", f"{latest.get('etco2_mmHg', 0.0):.1f}")
    c7.metric("Leak (%)", f"{latest.get('leak_pct', 0.0):.1f}")
    c8.metric("Battery (%)", f"{latest.get('battery_pct', 0.0):.1f}")


def _alarm_risk_panel(latest: Dict[str, float]) -> None:
    st.subheader("Alarm & Risk Level")
    level = latest.get("severity_level", "Normal")
    score = float(latest.get("severity_score", 0.0))
    st.write(f"**Risk Level:** {level}")
    st.progress(min(1.0, max(0.0, score)))

    st.write(f"**Fault Class:** {latest.get('fault_class', 'normal')}")
    st.write(f"**Root Cause:** {latest.get('fault_root_cause', 'none')}")


def _sensor_health_panel(latest: Dict[str, float]) -> None:
    st.subheader("Sensor Health Indicator")
    drift = float(latest.get("sensor_drift_pct", 0.0)) if "sensor_drift_pct" in latest else None
    leak = float(latest.get("leak_pct", 0.0))

    health = 1.0
    health *= (1.0 - min(0.7, leak / 100.0))
    st.progress(max(0.0, min(1.0, health)))
    st.write(f"**Estimated sensor/circuit health:** {health:.2f}")


def _what_if_sliders() -> ScenarioControls:
    st.subheader("What-if Sliders")

    fio2 = st.slider("FiO2", min_value=0.21, max_value=1.0, value=0.5, step=0.01)
    inspiratory_flow = st.slider("Inspiratory flow (L/s)", min_value=0.05, max_value=1.5, value=0.5, step=0.05)
    airway_pressure = st.slider("Airway pressure (proxy target, cmH2O)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
    leak = st.slider("Leak (%)", min_value=0.0, max_value=60.0, value=0.0, step=1.0)
    compliance = st.slider("Lung compliance (L/cmH2O)", min_value=0.005, max_value=0.08, value=0.05, step=0.001)
    sensor_drift = st.slider("Sensor drift (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    rr = st.slider("Respiratory rate (bpm)", min_value=6.0, max_value=40.0, value=16.0, step=1.0)
    blower_speed = st.slider("Blower speed (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
    valve_delay = st.slider("Valve delay (s)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    controls = ScenarioControls(
        fio2=float(fio2),
        inspiratory_flow_Lps=float(inspiratory_flow),
        airway_pressure_target_cmH2O=float(airway_pressure),
        leak_pct=float(leak),
        compliance_L_per_cmH2O=float(compliance),
        sensor_drift_pct=float(sensor_drift),
        respiratory_rate_bpm=float(rr),
        blower_speed_pct=float(blower_speed),
        valve_delay_s=float(valve_delay),
    )
    return controls


def _ai_reasoning_panel(latest: Dict[str, float]) -> None:
    st.subheader("AI Reasoning Explanation")
    st.write(f"**Fault classification:** {latest.get('explanation_fault_classification', 'normal')}")
    st.write(f"**Root cause:** {latest.get('explanation_root_cause', 'none')}")
    st.write(f"**Severity:** {latest.get('explanation_severity', '')}")
    st.write(f"**Recommended action:** {latest.get('explanation_recommended_action', '')}")
    st.write(f"**Urgency:** {latest.get('explanation_urgency', '')}")
    st.write(f"**Active alarms:** {latest.get('explanation_active_alarms', 'None')}")


def _trend_graphs(history: List[Dict[str, float]]) -> None:
    st.subheader("Trend Graphs")
    if not history:
        st.info("No data yet.")
        return

    df = pd.DataFrame(history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["pip_cmH2O"], name="PIP (cmH2O)", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["plateau_cmH2O"], name="Plateau (cmH2O)", mode="lines"))
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["peep_cmH2O"], name="PEEP (cmH2O)", mode="lines"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["time_s"], y=df["tidal_volume_L"], name="VT (L)", mode="lines"))
    fig2.add_trace(go.Scatter(x=df["time_s"], y=df["flow_Lps"], name="Flow (L/s)", mode="lines"))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["time_s"], y=df["spo2_pct"], name="SpO2 (%)", mode="lines"))
    fig3.add_trace(go.Scatter(x=df["time_s"], y=df["etco2_mmHg"], name="EtCO2 (mmHg)", mode="lines"))
    fig3.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
    st.plotly_chart(fig3, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Ventilator Digital Twin (ARDS)", layout="wide")
    st.title("Ventilator Digital Twin Simulation Platform (Adult ARDS ICU)")

    _init_state()

    with st.sidebar:
        st.header("Simulation")
        st.session_state.run = st.toggle("Run", value=st.session_state.run)
        st.session_state.steps_per_refresh = st.slider("Steps per refresh", 1, 25, int(st.session_state.steps_per_refresh), 1)
        st.session_state.oxygen_supply_ok = st.toggle("Oxygen supply OK", value=st.session_state.oxygen_supply_ok)
        if st.button("Reset"):
            st.session_state.engine = ScenarioEngine(dt_s=0.2)

    col_left, col_right = st.columns([1.1, 0.9])

    with col_right:
        controls = _what_if_sliders()
        st.session_state.engine.set_controls(controls, oxygen_supply_ok=st.session_state.oxygen_supply_ok)

    if st.session_state.run:
        st.session_state.engine.step_n(st.session_state.steps_per_refresh)

    latest = st.session_state.engine.get_latest()
    history = st.session_state.engine.get_history()

    with col_left:
        st.subheader("Real-time Telemetry Gauges")
        _telemetry_gauges(latest)

        _alarm_risk_panel(latest)
        _ai_reasoning_panel(latest)

    with col_right:
        _sensor_health_panel(latest)

    _trend_graphs(history)

    # Soft real-time refresh loop
    if st.session_state.run:
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()
