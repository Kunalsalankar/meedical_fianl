from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from lung_model import LungModel, LungParameters
from utils import clamp


@dataclass
class VentilatorSettings:
    inspiratory_flow_Lps: float = 0.5
    respiratory_rate_bpm: float = 16.0
    fio2: float = 0.5
    peep_cmH2O: float = 8.0
    blower_speed_pct: float = 55.0

    ie_ratio_insp: float = 1.0
    ie_ratio_exp: float = 2.0


@dataclass
class VentilatorInternalState:
    phase: str = "exp"
    phase_time_s: float = 0.0
    last_breath_start_s: float = 0.0
    pip_cmH2O: float = 0.0
    plateau_cmH2O: float = 0.0
    tidal_volume_L: float = 0.0


class VentilatorModel:
    def __init__(self, lung: LungModel, settings: VentilatorSettings):
        self._lung = lung
        self._settings = settings
        self._state = VentilatorInternalState()
        self._volume_at_insp_start_L = lung.volume_L

    @property
    def settings(self) -> VentilatorSettings:
        return self._settings

    def set_settings(self, settings: VentilatorSettings) -> None:
        self._settings = settings

    def _cycle_times(self) -> Tuple[float, float]:
        rr = max(1.0, float(self._settings.respiratory_rate_bpm))
        t_cycle = 60.0 / rr
        insp_frac = self._settings.ie_ratio_insp / max(1e-6, (self._settings.ie_ratio_insp + self._settings.ie_ratio_exp))
        t_insp = max(0.2, t_cycle * insp_frac)
        return t_cycle, t_insp

    def _exp_flow(self) -> float:
        c = max(1e-6, self._lung.params.compliance_L_per_cmH2O)
        r = max(1e-6, self._lung.params.resistance_cmH2O_per_Lps)
        v = max(0.0, self._lung.volume_L)
        dp = max(0.0, (v / c))
        return -dp / r

    def step(self, sim_time_s: float, dt_s: float) -> Dict[str, float]:
        t_cycle, t_insp = self._cycle_times()

        self._state.phase_time_s += dt_s
        if self._state.phase == "insp" and self._state.phase_time_s >= t_insp:
            self._state.phase = "exp"
            self._state.phase_time_s = 0.0
        elif self._state.phase == "exp" and self._state.phase_time_s >= (t_cycle - t_insp):
            self._state.phase = "insp"
            self._state.phase_time_s = 0.0
            self._state.last_breath_start_s = sim_time_s
            self._volume_at_insp_start_L = self._lung.volume_L
            self._state.pip_cmH2O = 0.0
            self._state.plateau_cmH2O = 0.0
            self._state.tidal_volume_L = 0.0

        # Apply PEEP to lung
        lp = self._lung.params
        self._lung.set_params(
            LungParameters(
                compliance_L_per_cmH2O=lp.compliance_L_per_cmH2O,
                resistance_cmH2O_per_Lps=lp.resistance_cmH2O_per_Lps,
                peep_cmH2O=float(self._settings.peep_cmH2O),
            )
        )

        if self._state.phase == "insp":
            flow = float(max(0.0, self._settings.inspiratory_flow_Lps))
        else:
            flow = float(self._exp_flow())

        lung_out = self._lung.step(flow_Lps=flow, dt_s=dt_s)

        paw = float(lung_out["paw_cmH2O"])
        v = float(lung_out["volume_L"])
        self._state.pip_cmH2O = max(self._state.pip_cmH2O, paw)

        c = max(1e-6, self._lung.params.compliance_L_per_cmH2O)
        self._state.plateau_cmH2O = max(self._state.plateau_cmH2O, (v / c) + self._settings.peep_cmH2O)

        if self._state.phase == "insp":
            self._state.tidal_volume_L = max(0.0, v - self._volume_at_insp_start_L)

        return {
            "phase": self._state.phase,
            "paw_cmH2O": paw,
            "pip_cmH2O": self._state.pip_cmH2O,
            "plateau_cmH2O": self._state.plateau_cmH2O,
            "peep_cmH2O": float(self._settings.peep_cmH2O),
            "tidal_volume_L": self._state.tidal_volume_L,
            "flow_Lps": float(lung_out["flow_Lps"]),
            "insp_flow_set_Lps": float(max(0.0, self._settings.inspiratory_flow_Lps)),
            "insp_flow_Lps": float(max(0.0, lung_out["flow_Lps"])),
            "rr_bpm": float(self._settings.respiratory_rate_bpm),
            "fio2": float(clamp(self._settings.fio2, 0.21, 1.0)),
            "blower_speed_pct": float(clamp(self._settings.blower_speed_pct, 0.0, 100.0)),
        }
