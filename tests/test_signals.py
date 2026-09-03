import pandas as pd

from ml_trading.signals import SignalPolicy, make_signals


def test_rank_and_magnitude_are_both_required():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C", "D"]], names=["date", "symbol"]
    )
    prediction = pd.Series([-0.50, -0.01, 0.01, 0.50], index=index)
    result = make_signals(prediction, SignalPolicy(0.75, 0.25, 0.10))
    assert result.decision.tolist() == ["SHORT", "NO_TRADE", "NO_TRADE", "LONG"]
