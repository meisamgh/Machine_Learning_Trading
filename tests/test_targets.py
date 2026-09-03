import numpy as np

from ml_trading.targets import continuous_residual_target


def test_target_uses_next_open_and_horizon_close_without_market(panel):
    plain = panel.drop(columns="market_return")
    target = continuous_residual_target(plain, horizon=5)
    symbol = "A"
    asset = plain.xs(symbol, level="symbol")
    date = asset.index[50]
    raw = asset.close.iloc[55] / asset.open.iloc[51] - 1
    vol = asset.adjusted_close.pct_change().rolling(20).std().iloc[50]
    assert np.isclose(target.loc[(date, symbol)], raw / (vol * np.sqrt(5)))
