"""Trading candidate policies with explicit economic edge requirements."""
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
    """Backward-compatible rank + standardized-magnitude policy."""
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


def make_economic_signals(
    expected_alpha: pd.Series,
    estimated_round_trip_cost: pd.Series | float,
    long_rank: float = 0.90,
    short_rank: float = 0.10,
    minimum_net_edge: float = 0.001,
    eligible: pd.Series | None = None,
) -> pd.DataFrame:
    """Require the forecast edge to exceed explicit estimated trading costs.

    Costs and alpha are both decimal returns, e.g. 10 bps = 0.001.
    """
    if not 0 < short_rank < long_rank < 1:
        raise ValueError("require 0 < short_rank < long_rank < 1")
    costs = (
        pd.Series(float(estimated_round_trip_cost), index=expected_alpha.index)
        if np.isscalar(estimated_round_trip_cost)
        else estimated_round_trip_cost.reindex(expected_alpha.index)
    )
    eligibility = (
        pd.Series(True, index=expected_alpha.index)
        if eligible is None
        else eligible.reindex(expected_alpha.index, fill_value=False).astype(bool)
    )
    ranks = expected_alpha.groupby(level="date").rank(pct=True)
    long_edge = expected_alpha - costs
    short_edge = -expected_alpha - costs
    long = eligibility & (ranks >= long_rank) & (long_edge >= minimum_net_edge)
    short = eligibility & (ranks <= short_rank) & (short_edge >= minimum_net_edge)
    decision = np.select([long, short], ["LONG", "SHORT"], default="NO_TRADE")
    return pd.DataFrame(
        {
            "expected_alpha": expected_alpha,
            "estimated_round_trip_cost": costs,
            "expected_net_edge": np.where(
                expected_alpha >= 0, long_edge, short_edge
            ),
            "prediction_rank": ranks,
            "eligible": eligibility,
            "decision": decision,
        },
        index=expected_alpha.index,
    )
