"""Continuous targets aligned to after-close signals and next-open execution."""
from __future__ import annotations

import numpy as np
import pandas as pd


def continuous_residual_target(
    panel: pd.DataFrame,
    horizon: int = 5,
    volatility_window: int = 20,
    beta_window: int = 60,
) -> pd.Series:
    """Return volatility-adjusted market-residual alpha at signal date t.

    Entry is open[t+1], exit is close[t+horizon]. Beta and volatility are trailing only.
    """
    panel = panel.sort_index()
    grouped = panel.groupby(level="symbol")
    future = grouped.close.shift(-horizon) / grouped.open.shift(-1) - 1
    stock_return = grouped.adjusted_close.pct_change(fill_method=None)
    volatility = stock_return.groupby(level="symbol").transform(
        lambda x: x.rolling(volatility_window, min_periods=volatility_window).std()
    )
    if "market_return" not in panel:
        residual = future
    else:
        market = panel.market_return
        covariance = stock_return.groupby(level="symbol").transform(
            lambda x: x.rolling(beta_window, min_periods=30).cov(market.loc[x.index])
        )
        variance = market.groupby(level="symbol").transform(
            lambda x: x.rolling(beta_window, min_periods=30).var()
        )
        beta = covariance / variance.replace(0, np.nan)
        future_market = market.groupby(level="symbol").transform(
            lambda x: (1 + x.shift(-1)).rolling(horizon).apply(np.prod, raw=True).shift(
                -(horizon - 1)
            ) - 1
        )
        residual = future - beta * future_market
    return (residual / (volatility * np.sqrt(horizon)).replace(0, np.nan)).rename("target")
