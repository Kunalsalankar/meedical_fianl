from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> str:
    df = pd.read_csv(path)
    return df.to_csv(index=False)
