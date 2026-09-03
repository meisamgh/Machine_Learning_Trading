"""LONG/SHORT/NO_TRADE policy with an economic magnitude filter."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalPolicy:
    long_rank: float = 0.90
    short_rank: float = 0.10
    minimum_alpha: float = 0.10


def make_signals(
    prediction: pd.Series, policy: SignalPolicy | None = None
) -> pd.DataFrame:
    """Rank model predictions by date, then apply direction and magnitude requirements."""
    policy = policy or SignalPolicy()
    if "date" not in prediction.index.names:
        raise ValueError("prediction must use an index with a date level")
    ranks = prediction.groupby(level="date").rank(pct=True)
    decision = np.select(
        [
            (ranks >= policy.long_rank) & (prediction > policy.minimum_alpha),
            (ranks <= policy.short_rank) & (prediction < -policy.minimum_alpha),
        ],
        ["LONG", "SHORT"],
        default="NO_TRADE",
    )
    return pd.DataFrame(
        {"predicted_alpha": prediction, "prediction_rank": ranks, "decision": decision},
        index=prediction.index,
    )
