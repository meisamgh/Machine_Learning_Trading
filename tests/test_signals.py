import pandas as pd

from ml_trading.signals import SignalPolicy, make_economic_signals, make_signals


def test_rank_and_magnitude_are_both_required():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C", "D"]], names=["date", "symbol"]
    )
    prediction = pd.Series([-0.50, -0.01, 0.01, 0.50], index=index)
    result = make_signals(prediction, SignalPolicy(0.75, 0.25, 0.10))
    assert result.decision.tolist() == ["SHORT", "NO_TRADE", "NO_TRADE", "LONG"]


def test_economic_signal_requires_edge_after_costs():
    index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C", "D"]], names=["date", "symbol"]
    )
    alpha = pd.Series([-0.004, -0.0005, 0.0005, 0.004], index=index)
    result = make_economic_signals(alpha, 0.001, 0.75, 0.25, minimum_net_edge=0.002)
    assert result.decision.tolist() == ["SHORT", "NO_TRADE", "NO_TRADE", "LONG"]
