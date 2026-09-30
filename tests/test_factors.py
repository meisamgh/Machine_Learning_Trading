import pandas as pd

from ml_trading.factors import (
    compute_feature_library,
    compute_trusted_factors,
    cross_sectional_transform,
)


def test_factors_are_causal(panel):
    cutoff = panel.index.get_level_values("date").unique()[200]
    baseline = compute_trusted_factors(panel).loc[:cutoff]
    changed = panel.copy()
    future = changed.index.get_level_values("date") > cutoff
    changed.loc[future, ["open", "high", "low", "close", "adjusted_close", "volume"]] *= 5
    pd.testing.assert_frame_equal(baseline, compute_trusted_factors(changed).loc[:cutoff])


def test_feature_library_and_cross_sectional_normalization(panel):
    features = compute_feature_library(panel)
    assert features.index.equals(panel.index)
    assert features.shape[1] >= 15
    transformed = cross_sectional_transform(features, mode="zscore")
    date = transformed.dropna().index.get_level_values("date")[0]
    day = transformed.xs(date, level="date")
    assert abs(day.mean().mean()) < 1e-8
