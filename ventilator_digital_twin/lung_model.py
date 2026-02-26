from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class LungParameters:
    """Single-compartment lung mechanics.

    - compliance_L_per_cmH2O: L/cmH2O
    - resistance_cmH2O_per_Lps: cmH2O/(L/s)
    """

    compliance_L_per_cmH2O: float = 0.05
    resistance_cmH2O_per_Lps: float = 8.0
    peep_cmH2O: float = 5.0


class LungModel:
    """SciPy ODE lung model.

    Pressure-volume relationship:
        P = (V / C) + (R * Flow) + PEEP

    State variable:
        V (L)
    Input:
        Flow (L/s)
    """

    def __init__(self, params: LungParameters, initial_volume_L: float = 0.0):
        self._params = params
        self._volume_L = float(max(0.0, initial_volume_L))

    @property
    def params(self) -> LungParameters:
        return self._params

    def set_params(self, params: LungParameters) -> None:
        self._params = params

    @property
    def volume_L(self) -> float:
        return self._volume_L

    def airway_pressure_cmH2O(self, flow_Lps: float) -> float:
        c = max(1e-6, self._params.compliance_L_per_cmH2O)
        r = max(0.0, self._params.resistance_cmH2O_per_Lps)
        v = max(0.0, self._volume_L)
        return (v / c) + (r * float(flow_Lps)) + float(self._params.peep_cmH2O)

    def step(self, flow_Lps: float, dt_s: float) -> Dict[str, float]:
        dt_s = float(max(1e-4, dt_s))
        flow_Lps = float(flow_Lps)

        def dVdt(_t: float, y: np.ndarray) -> np.ndarray:
            return np.array([flow_Lps], dtype=float)

        sol = solve_ivp(
            fun=dVdt,
            t_span=(0.0, dt_s),
            y0=np.array([self._volume_L], dtype=float),
            method="RK45",
            max_step=dt_s,
        )

        self._volume_L = float(max(0.0, sol.y[0, -1]))
        paw = float(self.airway_pressure_cmH2O(flow_Lps))

        return {
            "volume_L": self._volume_L,
            "flow_Lps": flow_Lps,
            "paw_cmH2O": paw,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_ards_params(ards_severity: float) -> LungParameters:
    """Return lung parameters consistent with ARDS severity.

    A severity value of 0 corresponds to a healthy adult lung;
    a value of 1.0 corresponds to very severe ARDS.  The mapping is
    largely arbitrary but should reflect the following trends:

    * Compliance decreases as severity increases (normal ~0.05 L/cmH2O,
      severe ~0.005 L/cmH2O).
    * Resistance increases with severity.
    * PEEP requirement often rises with worsening ARDS.

    The returned :class:`LungParameters` instance can be passed directly to
    :class:`LungModel` or used to seed other components.
    """
    s = float(max(0.0, min(1.0, ards_severity)))

    # linear interpolation between healthy and severe values
    compliance = (1.0 - s) * 0.05 + s * 0.005
    resistance = 8.0 + 12.0 * s  # from 8 to 20 cmH2O/(L/s)
    peep = 5.0 + 10.0 * s  # from 5 to 15 cmH2O

    return LungParameters(
        compliance_L_per_cmH2O=compliance,
        resistance_cmH2O_per_Lps=resistance,
        peep_cmH2O=peep,
    )
