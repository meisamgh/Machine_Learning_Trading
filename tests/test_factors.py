import pandas as pd

from ml_trading.factors import compute_trusted_factors


def test_factors_are_causal(panel):
    cutoff = panel.index.get_level_values("date").unique()[60]
    baseline = compute_trusted_factors(panel).loc[:cutoff]
    changed = panel.copy()
    future = changed.index.get_level_values("date") > cutoff
    changed.loc[future, ["open", "high", "low", "close", "volume"]] *= 5
    pd.testing.assert_frame_equal(baseline, compute_trusted_factors(changed).loc[:cutoff])


def test_factors_preserve_panel_index(panel):
    assert compute_trusted_factors(panel).index.equals(panel.index)
