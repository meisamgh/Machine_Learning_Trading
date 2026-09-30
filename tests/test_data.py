import numpy as np
import pandas as pd

from ml_trading.data import adjusted_price_panel, apply_point_in_time_universe, panel_fingerprint


def test_adjusted_prices_use_one_consistent_factor(panel):
    sample = panel.loc[(slice(None), "A"), :].copy()
    split_date = sample.index.get_level_values("date")[100]
    before = sample.index.get_level_values("date") < split_date
    sample.loc[before, "adjusted_close"] = sample.loc[before, "close"] / 2
    adjusted = adjusted_price_panel(sample)
    np.testing.assert_allclose(
        adjusted.loc[before, "px_open"], sample.loc[before, "open"] / 2
    )
    np.testing.assert_allclose(
        adjusted.loc[before, "px_close"], sample.loc[before, "adjusted_close"]
    )


def test_point_in_time_membership_fail_closed(panel):
    dates = panel.index.get_level_values("date")
    membership = pd.DataFrame(
        {
            "symbol": ["A"],
            "start_date": [dates.min()],
            "end_date": [pd.Timestamp("2020-12-31")],
        }
    )
    filtered = apply_point_in_time_universe(panel, membership)
    assert set(filtered.index.get_level_values("symbol")) == {"A"}
    assert filtered.index.get_level_values("date").max() <= pd.Timestamp("2020-12-31")


def test_fingerprint_changes_when_data_changes(panel):
    changed = panel.copy()
    changed.iloc[0, changed.columns.get_loc("close")] *= 1.01
    assert panel_fingerprint(panel) != panel_fingerprint(changed)
