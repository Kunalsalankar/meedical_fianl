from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from thresholds import AlarmThresholds
from utils import Alarm, clamp, normalize_deviation, rolling_window_last


@dataclass
class FaultControls:
    leak_pct: float = 0.0
    sensor_drift_pct: float = 0.0
    valve_delay_s: float = 0.0
    oxygen_supply_ok: bool = True


@dataclass
class PatientRiskFactors:
    ards_severity: float = 0.6


class FaultEngine:
    def __init__(
        self,
        thresholds: AlarmThresholds,
        dt_s: float,
        persist_seconds: float = 5.0,
    ):
        self._thr = thresholds
        self._dt_s = float(max(1e-3, dt_s))
        self._persist_seconds = float(max(1.0, persist_seconds))

    def apply_sensor_effects(self, telemetry: Dict[str, float], controls: FaultControls) -> Dict[str, float]:
        out = dict(telemetry)

        leak = float(clamp(controls.leak_pct, 0.0, 100.0))
        drift = float(clamp(controls.sensor_drift_pct, 0.0, 20.0))

        out["leak_pct"] = leak
        out["tidal_volume_L"] = max(0.0, out.get("tidal_volume_L", 0.0) * (1.0 - leak / 100.0))
        out["pip_cmH2O"] = out.get("pip_cmH2O", 0.0) * (1.0 - 0.3 * leak / 100.0)
        out["plateau_cmH2O"] = out.get("plateau_cmH2O", 0.0) * (1.0 - 0.25 * leak / 100.0)

        drift_factor = 1.0 + drift / 100.0
        out["pip_cmH2O"] *= drift_factor
        out["plateau_cmH2O"] *= drift_factor
        out["flow_Lps"] *= drift_factor
        out["tidal_volume_L"] *= drift_factor
        out["sensor_drift_pct"] = drift

        valve_delay = float(max(0.0, controls.valve_delay_s))
        if valve_delay > 0.0:
            out["flow_Lps"] *= clamp(1.0 - (valve_delay / 1.0), 0.5, 1.0)

        return out

    def derive_gases(self, telemetry: Dict[str, float], controls: FaultControls, patient: PatientRiskFactors) -> Dict[str, float]:
        fio2 = float(clamp(telemetry.get("fio2", 0.5), 0.21, 1.0))
        vt = float(max(0.0, telemetry.get("tidal_volume_L", 0.0)))
        rr = float(max(1.0, telemetry.get("rr_bpm", 16.0)))
        leak = float(clamp(telemetry.get("leak_pct", 0.0), 0.0, 100.0))

        oxygen_ok = bool(controls.oxygen_supply_ok)
        effective_fio2 = fio2 if oxygen_ok else 0.21

        minute_vent = vt * rr
        oxygenation_factor = clamp(1.0 - 0.6 * clamp(patient.ards_severity, 0.0, 1.0), 0.3, 1.0)
        leak_penalty = clamp(1.0 - leak / 100.0, 0.0, 1.0)

        spo2 = 80.0 + 20.0 * clamp((effective_fio2 - 0.21) / 0.79, 0.0, 1.0) * oxygenation_factor * leak_penalty
        spo2 = clamp(spo2, 70.0, 100.0)

        mv_target = 7.0
        etco2 = 40.0 + 12.0 * clamp((mv_target - minute_vent) / mv_target, -1.0, 1.0)
        etco2 = clamp(etco2, 20.0, 70.0)

        out = dict(telemetry)
        out["spo2_pct"] = float(spo2)
        out["etco2_mmHg"] = float(etco2)
        return out

    def compute_alarms(self, telemetry: Dict[str, float]) -> List[Alarm]:
        pip = float(telemetry.get("pip_cmH2O", 0.0))
        plat = float(telemetry.get("plateau_cmH2O", 0.0))
        spo2 = float(telemetry.get("spo2_pct", 100.0))
        etco2 = float(telemetry.get("etco2_mmHg", 40.0))
        leak = float(telemetry.get("leak_pct", 0.0))
        batt = float(telemetry.get("battery_pct", 100.0))

        alarms: List[Alarm] = []

        alarms.append(Alarm("PIP High", pip > self._thr.pip_high_cmH2O, pip, f"> {self._thr.pip_high_cmH2O}", "critical"))
        alarms.append(Alarm("Plateau High", plat > self._thr.plateau_high_cmH2O, plat, f"> {self._thr.plateau_high_cmH2O}", "high"))
        alarms.append(Alarm("SpO2 Low", spo2 < self._thr.spo2_low_pct, spo2, f"< {self._thr.spo2_low_pct}", "critical"))
        alarms.append(Alarm("EtCO2 Low", etco2 < self._thr.etco2_low_mmHg, etco2, f"< {self._thr.etco2_low_mmHg}", "high"))
        alarms.append(Alarm("EtCO2 High", etco2 > self._thr.etco2_high_mmHg, etco2, f"> {self._thr.etco2_high_mmHg}", "high"))
        alarms.append(Alarm("Leak High", leak > self._thr.leak_high_pct, leak, f"> {self._thr.leak_high_pct}%", "high"))
        alarms.append(Alarm("Battery Low", batt < self._thr.battery_low_pct, batt, f"< {self._thr.battery_low_pct}%", "warning"))

        return alarms

    def temporal_validate(self, history: List[Dict[str, float]], alarms: List[Alarm]) -> Dict[str, bool]:
        window = rolling_window_last(history, seconds=self._persist_seconds, dt=self._dt_s)
        persistent: Dict[str, bool] = {}
        for a in alarms:
            if not a.active:
                persistent[a.name] = False
                continue
            if not window:
                persistent[a.name] = True
                continue
            key = _alarm_to_key(a.name)
            active_count = sum(1 for h in window if bool(h.get(key, False)))
            persistent[a.name] = (active_count / max(1, len(window))) >= 0.7
        return persistent

    def reason_fault(self, telemetry: Dict[str, float], persistent_map: Dict[str, bool]) -> Dict[str, str]:
        pip = float(telemetry.get("pip_cmH2O", 0.0))
        flow = float(telemetry.get("flow_Lps", 0.0))
        vt = float(telemetry.get("tidal_volume_L", 0.0))
        leak = float(telemetry.get("leak_pct", 0.0))
        spo2 = float(telemetry.get("spo2_pct", 100.0))
        fio2 = float(telemetry.get("fio2", 0.5))

        high_p = pip > self._thr.pip_high_cmH2O
        low_p = pip < 5.0
        low_flow = abs(flow) < 0.05
        low_vt = vt < 0.25
        high_leak = leak > self._thr.leak_high_pct
        low_spo2 = spo2 < self._thr.spo2_low_pct
        low_fio2 = fio2 < 0.3

        high_p = high_p and persistent_map.get("PIP High", True)
        high_leak = high_leak and persistent_map.get("Leak High", True)
        low_spo2 = low_spo2 and persistent_map.get("SpO2 Low", True)

        classification = "normal"
        root_cause = "none"
        if high_p and low_flow:
            classification = "obstruction"
            root_cause = "Airway obstruction or kinked circuit"
        elif low_p and low_vt and high_leak:
            classification = "disconnect"
            root_cause = "Circuit disconnect / major leak"
        elif low_spo2 and low_fio2:
            classification = "oxygen_supply_failure"
            root_cause = "Oxygen supply failure or low O2 source"
        elif (high_p or low_p) and not low_flow:
            classification = "sensor_fault"
            root_cause = "Pressure sensor drift/fault (flow appears normal)"

        return {"fault_class": classification, "root_cause": root_cause}

    def severity_score(
        self,
        telemetry: Dict[str, float],
        alarms: List[Alarm],
        patient: PatientRiskFactors,
    ) -> Tuple[float, str]:
        devs = []
        devs.append(normalize_deviation(float(telemetry.get("pip_cmH2O", 0.0)), 0.0, self._thr.pip_high_cmH2O))
        devs.append(normalize_deviation(float(telemetry.get("plateau_cmH2O", 0.0)), 0.0, self._thr.plateau_high_cmH2O))
        devs.append(normalize_deviation(float(telemetry.get("spo2_pct", 100.0)), self._thr.spo2_low_pct, 100.0))
        devs.append(normalize_deviation(float(telemetry.get("etco2_mmHg", 40.0)), self._thr.etco2_low_mmHg, self._thr.etco2_high_mmHg))
        devs.append(normalize_deviation(float(telemetry.get("leak_pct", 0.0)), 0.0, self._thr.leak_high_pct))
        param_dev = float(np.mean(devs)) if devs else 0.0

        abnormal_sensors = sum(1 for a in alarms if a.active)
        abnormal_norm = clamp(abnormal_sensors / 6.0, 0.0, 1.0)

        crit_map = {"warning": 0.3, "high": 0.7, "critical": 1.0}
        alarm_crit = max((crit_map.get(a.severity, 0.5) for a in alarms if a.active), default=0.0)

        patient_risk = clamp(patient.ards_severity, 0.0, 1.0)

        score = 0.4 * param_dev + 0.3 * abnormal_norm + 0.2 * alarm_crit + 0.1 * patient_risk
        score = float(clamp(score, 0.0, 1.0))

        if score < 0.3:
            level = "Normal"
        elif score < 0.6:
            level = "Warning"
        elif score < 0.8:
            level = "High Risk"
        else:
            level = "Critical"

        return score, level

    def explain(
        self,
        telemetry: Dict[str, float],
        alarms: List[Alarm],
        fault: Dict[str, str],
        severity_score: float,
        severity_level: str,
    ) -> Dict[str, str]:
        fault_class = fault.get("fault_class", "normal")
        root = fault.get("root_cause", "none")

        if severity_level in {"Critical", "High Risk"}:
            urgency = "Immediate"
        elif severity_level == "Warning":
            urgency = "Prompt"
        else:
            urgency = "Routine"

        if fault_class == "obstruction":
            action = "Check tubing/circuit for kinks, suction if needed, assess secretions."
        elif fault_class == "disconnect":
            action = "Inspect patient circuit connections, check cuff seal, and verify leak source."
        elif fault_class == "oxygen_supply_failure":
            action = "Verify wall O2 / cylinder pressure, check blender, and confirm FiO2 delivery."
        elif fault_class == "sensor_fault":
            action = "Cross-check with backup sensor, recalibrate/replace pressure sensor."
        else:
            action = "Continue monitoring; adjust ventilator settings per protocol if needed."

        active_alarms = [a.name for a in alarms if a.active]
        alarm_text = ", ".join(active_alarms) if active_alarms else "None"

        return {
            "fault_classification": fault_class,
            "root_cause": root,
            "severity": f"{severity_level} ({severity_score:.2f})",
            "recommended_action": action,
            "urgency": urgency,
            "active_alarms": alarm_text,
        }


def _alarm_to_key(name: str) -> str:
    return {
        "PIP High": "alarm_pip_high",
        "Plateau High": "alarm_plateau_high",
        "SpO2 Low": "alarm_spo2_low",
        "EtCO2 Low": "alarm_etco2_low",
        "EtCO2 High": "alarm_etco2_high",
        "Leak High": "alarm_leak_high",
        "Battery Low": "alarm_battery_low",
    }.get(name, f"alarm_{name.lower().replace(' ', '_')}")
