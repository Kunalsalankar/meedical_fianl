from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlarmThresholds:
    pip_high_cmH2O: float = 40.0
    plateau_high_cmH2O: float = 30.0
    spo2_low_pct: float = 90.0
    etco2_low_mmHg: float = 25.0
    etco2_high_mmHg: float = 55.0
    leak_high_pct: float = 20.0
    battery_low_pct: float = 20.0


DEFAULT_THRESHOLDS = AlarmThresholds()
