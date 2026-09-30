import numpy as np

from ml_trading.targets import continuous_residual_target, forward_adjusted_return


def test_target_uses_next_open_and_horizon_close_without_market(panel):
    plain = panel.drop(columns="market_return")
    target = continuous_residual_target(plain, horizon=5)
    symbol = "A"
    asset = plain.xs(symbol, level="symbol")
    date = asset.index[100]
    raw = asset.adjusted_close.iloc[105] / asset.open.iloc[101] - 1
    vol = asset.adjusted_close.pct_change().rolling(20).std().iloc[100]
    assert np.isclose(target.loc[(date, symbol)], raw / (vol * np.sqrt(5)))


def test_forward_return_survives_split_adjustment(panel):
    sample = panel.loc[(slice(None), "A"), :].copy()
    split_pos = 100
    dates = sample.index.get_level_values("date")
    pre = dates < dates[split_pos]
    sample.loc[pre, ["open", "high", "low", "close"]] *= 2
    returns = forward_adjusted_return(sample, horizon=5)
    assert np.isfinite(returns.dropna()).all()
