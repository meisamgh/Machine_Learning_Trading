"""Small trusted factor registry extracted from the legacy Alpha notebooks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_012(panel: pd.DataFrame) -> pd.Series:
    """Alpha 012: sign(delta(volume, 1)) * -delta(close, 1), per symbol."""
    grouped = panel.sort_index().groupby(level="symbol")
    return np.sign(grouped.volume.diff()) * -grouped.close.diff()


def alpha_101(panel: pd.DataFrame) -> pd.Series:
    """Alpha 101: (close - open) / (high - low + 0.001)."""
    return (panel.close - panel.open) / (panel.high - panel.low + 0.001)


TRUSTED_FACTORS = {
    "alpha_012": alpha_012,
    "alpha_101": alpha_101,
}


def compute_trusted_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute only reviewed factors and preserve the panel MultiIndex."""
    return pd.DataFrame({name: function(panel) for name, function in TRUSTED_FACTORS.items()})
