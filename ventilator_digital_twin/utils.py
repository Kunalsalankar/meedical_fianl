from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    if abs(denom) < 1e-12:
        return default
    return numer / denom


def normalize_deviation(value: float, low: float, high: float) -> float:
    """Return deviation in [0, 1] where 0 is within limits and 1 is far outside.

    This is intentionally conservative and monotonic.
    """
    if low <= value <= high:
        return 0.0
    if value < low:
        span = max(1e-6, abs(low))
        return clamp((low - value) / span, 0.0, 1.0)
    span = max(1e-6, abs(high))
    return clamp((value - high) / span, 0.0, 1.0)


def rolling_window_last(items: List[dict], seconds: float, dt: float) -> List[dict]:
    if not items:
        return []
    n = int(max(1, seconds / max(dt, 1e-6)))
    return items[-n:]


@dataclass(frozen=True)
class Alarm:
    name: str
    active: bool
    value: float
    threshold: str
    severity: str
