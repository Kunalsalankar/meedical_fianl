from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import simpy

from fault_engine import FaultControls, FaultEngine, PatientRiskFactors
from lung_model import LungModel, LungParameters, make_ards_params
from thresholds import AlarmThresholds, DEFAULT_THRESHOLDS
from ventilator_model import VentilatorModel, VentilatorSettings
from utils import clamp


@dataclass
class ScenarioControls:
    fio2: float = 0.5
    inspiratory_flow_Lps: float = 0.5
    airway_pressure_target_cmH2O: float = 20.0  # used as soft influence
    leak_pct: float = 0.0
    compliance_L_per_cmH2O: float = 0.05
    sensor_drift_pct: float = 0.0
    respiratory_rate_bpm: float = 16.0
    blower_speed_pct: float = 55.0
    valve_delay_s: float = 0.0


class ScenarioEngine:
    """SimPy-driven digital twin wrapper.

    The environment advances in dt increments. On each step we:
    - Update model parameters from controls
    - Step ventilator+lung
    - Apply fault effects
    - Derive gas exchange signals
    - Compute alarms, temporal validation, reasoning, severity, explanation
    """

    def __init__(self, dt_s: float = 0.2, thresholds: AlarmThresholds = DEFAULT_THRESHOLDS):
        self._dt_s = float(max(0.05, dt_s))
        self._thr = thresholds

        # patient risk factors are used during gas exchange calculations and to
        # configure the lung model according to ARDS severity.  We create the
        # patient object early so that ``make_ards_params`` can consume the
        # severity when building the initial lung parameters.
        self._patient = PatientRiskFactors(ards_severity=0.6)

        # derive the initial lung parameters based on ARDS severity; these
        # values can later be modified interactively via the compliance slider.
        base_lung = make_ards_params(self._patient.ards_severity)
        self._lung = LungModel(params=base_lung, initial_volume_L=0.0)

        self._vent = VentilatorModel(
            lung=self._lung,
            settings=VentilatorSettings(
                inspiratory_flow_Lps=0.5,
                respiratory_rate_bpm=16.0,
                fio2=0.5,
                peep_cmH2O=8.0,
                blower_speed_pct=55.0,
            ),
        )
        self._fault_engine = FaultEngine(thresholds=self._thr, dt_s=self._dt_s, persist_seconds=5.0)

        self._env = simpy.Environment()
        self._telemetry_history: List[Dict[str, float]] = []
        self._latest: Dict[str, float] = {}
        self._battery_pct = 100.0
        self._filters: Dict[str, float] = {}
        self._control_info: Dict[str, float] = {}

        self._process = self._env.process(self._run())

        self._controls = ScenarioControls()
        self._oxygen_supply_ok = True

    @property
    def dt_s(self) -> float:
        return self._dt_s

    def set_controls(self, controls: ScenarioControls, oxygen_supply_ok: bool) -> None:
        self._controls = controls
        self._oxygen_supply_ok = bool(oxygen_supply_ok)

    def get_latest(self) -> Dict[str, float]:
        return dict(self._latest)

    def get_history(self) -> List[Dict[str, float]]:
        return list(self._telemetry_history)

    def step_n(self, n: int = 1) -> None:
        for _ in range(max(1, int(n))):
            self._env.run(until=self._env.now + self._dt_s)

    def _apply_controls(self) -> None:
        c = self._controls

        # Update compliance directly (slider).
        lp = self._lung.params
        self._lung.set_params(
            LungParameters(
                compliance_L_per_cmH2O=float(max(0.005, c.compliance_L_per_cmH2O)),
                resistance_cmH2O_per_Lps=float(max(1.0, lp.resistance_cmH2O_per_Lps)),
                peep_cmH2O=float(max(0.0, self._vent.settings.peep_cmH2O)),
            )
        )

        # Settings
        s = self._vent.settings

        # Pressure-target control proxy:
        # Compute an inspiratory flow command to track the selected airway pressure target.
        # Flow is limited by the inspiratory flow slider (treated as max flow capability).
        max_insp_flow = float(max(0.05, c.inspiratory_flow_Lps))
        target_p = float(max(0.0, c.airway_pressure_target_cmH2O))
        lp2 = self._lung.params
        v = float(self._lung.volume_L)
        comp = float(max(1e-6, lp2.compliance_L_per_cmH2O))
        res = float(max(1e-6, lp2.resistance_cmH2O_per_Lps))
        elastic_p = (v / comp) + float(s.peep_cmH2O)
        flow_needed = (target_p - elastic_p) / res
        insp_flow_cmd = float(clamp(flow_needed, 0.0, max_insp_flow))

        self._control_info = {
            "pressure_target_cmH2O": target_p,
            "insp_flow_max_Lps": max_insp_flow,
            "insp_flow_cmd_Lps": insp_flow_cmd,
        }
        self._vent.set_settings(
            VentilatorSettings(
                inspiratory_flow_Lps=insp_flow_cmd,
                respiratory_rate_bpm=float(clamp(c.respiratory_rate_bpm, 6.0, 40.0)),
                fio2=float(clamp(c.fio2, 0.21, 1.0)),
                peep_cmH2O=float(clamp(s.peep_cmH2O, 0.0, 20.0)),
                blower_speed_pct=float(clamp(c.blower_speed_pct, 0.0, 100.0)),
                ie_ratio_insp=s.ie_ratio_insp,
                ie_ratio_exp=s.ie_ratio_exp,
            )
        )

    def _run(self):
        while True:
            self._apply_controls()

            sim_time_s = float(self._env.now)

            # Core ventilator physics step
            base = self._vent.step(sim_time_s=sim_time_s, dt_s=self._dt_s)

            # Derived telemetry
            self._battery_pct = float(max(0.0, self._battery_pct - 0.001 * self._dt_s * (1.0 + base["blower_speed_pct"] / 100.0)))

            telemetry = dict(base)
            telemetry["battery_pct"] = self._battery_pct

            # Fault controls
            fctrl = FaultControls(
                leak_pct=float(clamp(self._controls.leak_pct, 0.0, 60.0)),
                sensor_drift_pct=float(clamp(self._controls.sensor_drift_pct, 0.0, 20.0)),
                valve_delay_s=float(max(0.0, self._controls.valve_delay_s)),
                oxygen_supply_ok=self._oxygen_supply_ok,
            )

            measured = self._fault_engine.apply_sensor_effects(telemetry, fctrl)
            measured = self._fault_engine.derive_gases(measured, fctrl, self._patient)

            # Low-pass filter select signals for display stability.
            pip_f = self._lpf("pip_cmH2O", float(measured.get("pip_cmH2O", 0.0)), tau_s=1.2)
            plateau_f = self._lpf("plateau_cmH2O", float(measured.get("plateau_cmH2O", 0.0)), tau_s=1.5)
            flow_f = self._lpf("flow_Lps", float(measured.get("flow_Lps", 0.0)), tau_s=0.8)
            insp_flow_f = self._lpf("insp_flow_Lps", float(measured.get("insp_flow_Lps", 0.0)), tau_s=0.8)
            etco2_f = self._lpf("etco2_mmHg", float(measured.get("etco2_mmHg", 40.0)), tau_s=3.0)
            measured["pip_cmH2O_filt"] = pip_f
            measured["plateau_cmH2O_filt"] = plateau_f
            measured["flow_Lps_filt"] = flow_f
            measured["insp_flow_Lps_filt"] = insp_flow_f
            measured["etco2_mmHg_filt"] = etco2_f

            alarms = self._fault_engine.compute_alarms(measured)

            # Add alarm booleans into history for temporal validation
            alarm_flags = {
                "alarm_pip_high": any(a.name == "PIP High" and a.active for a in alarms),
                "alarm_plateau_high": any(a.name == "Plateau High" and a.active for a in alarms),
                "alarm_spo2_low": any(a.name == "SpO2 Low" and a.active for a in alarms),
                "alarm_etco2_low": any(a.name == "EtCO2 Low" and a.active for a in alarms),
                "alarm_etco2_high": any(a.name == "EtCO2 High" and a.active for a in alarms),
                "alarm_leak_high": any(a.name == "Leak High" and a.active for a in alarms),
                "alarm_battery_low": any(a.name == "Battery Low" and a.active for a in alarms),
            }

            record = dict(measured)
            record.update(alarm_flags)
            record["time_s"] = sim_time_s
            self._telemetry_history.append(record)

            # Keep a bounded history (10 minutes)
            max_points = int(600.0 / self._dt_s)
            if len(self._telemetry_history) > max_points:
                self._telemetry_history = self._telemetry_history[-max_points:]

            persistent = self._fault_engine.temporal_validate(self._telemetry_history, alarms)
            fault = self._fault_engine.reason_fault(measured, persistent)
            score, level = self._fault_engine.severity_score(measured, alarms, self._patient)
            explanation = self._fault_engine.explain(measured, alarms, fault, score, level)

            self._latest = {
                **measured,
                **self._control_info,
                "severity_score": float(score),
                "severity_level": str(level),
                "fault_class": fault.get("fault_class", "normal"),
                "fault_root_cause": fault.get("root_cause", "none"),
                "explanation_fault_classification": explanation["fault_classification"],
                "explanation_root_cause": explanation["root_cause"],
                "explanation_severity": explanation["severity"],
                "explanation_recommended_action": explanation["recommended_action"],
                "explanation_urgency": explanation["urgency"],
                "explanation_active_alarms": explanation["active_alarms"],
            }

            yield self._env.timeout(self._dt_s)

    def _lpf(self, key: str, x: float, tau_s: float) -> float:
        """First-order low-pass filter: y += alpha*(x-y), alpha=dt/(tau+dt)."""
        tau_s = float(max(1e-3, tau_s))
        alpha = float(self._dt_s / (tau_s + self._dt_s))
        if key not in self._filters:
            self._filters[key] = float(x)
            return float(x)
        y = float(self._filters[key])
        y = y + alpha * (float(x) - y)
        self._filters[key] = y
        return y
