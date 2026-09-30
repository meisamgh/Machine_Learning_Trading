"""Continuous forward targets aligned to after-close signals and next-open execution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import adjusted_price_panel


def forward_adjusted_return(panel: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return from adjusted open[t+1] to adjusted close[t+horizon]."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    px = adjusted_price_panel(panel)
    grouped = px.groupby(level="symbol")
    return (grouped.px_close.shift(-horizon) / grouped.px_open.shift(-1) - 1).rename(
        "forward_return"
    )


def trailing_volatility(panel: pd.DataFrame, window: int = 20) -> pd.Series:
    px = adjusted_price_panel(panel)
    returns = px.groupby(level="symbol").px_close.pct_change(fill_method=None)
    return returns.groupby(level="symbol").transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )


def continuous_residual_target(
    panel: pd.DataFrame,
    horizon: int = 5,
    volatility_window: int = 20,
    beta_window: int = 60,
) -> pd.Series:
    """Volatility-scaled market-residual alpha at signal date t.

    Entry is adjusted open[t+1], exit is adjusted close[t+horizon]. Beta and volatility use trailing
    data only. If ``market_return`` is unavailable, the target is volatility-adjusted forward return.
    """
    panel = panel.sort_index()
    px = adjusted_price_panel(panel)
    grouped = px.groupby(level="symbol")
    future = grouped.px_close.shift(-horizon) / grouped.px_open.shift(-1) - 1
    stock_return = grouped.px_close.pct_change(fill_method=None)
    volatility = stock_return.groupby(level="symbol").transform(
        lambda x: x.rolling(volatility_window, min_periods=volatility_window).std()
    )

    if "market_return" not in panel:
        residual = future
    else:
        market = panel["market_return"].astype(float)
        market_by_date = market.groupby(level="date").first()
        broadcast_market = pd.Series(
            panel.index.get_level_values("date").map(market_by_date), index=panel.index, dtype=float
        )

        covariance = stock_return.groupby(level="symbol").transform(
            lambda x: x.rolling(beta_window, min_periods=min(30, beta_window)).cov(
                broadcast_market.loc[x.index]
            )
        )
        market_variance_by_date = market_by_date.rolling(
            beta_window, min_periods=min(30, beta_window)
        ).var()
        market_variance = pd.Series(
            panel.index.get_level_values("date").map(market_variance_by_date), index=panel.index
        )
        beta = covariance / market_variance.replace(0, np.nan)

        future_market_by_date = (
            (1 + market_by_date.shift(-1))
            .rolling(horizon)
            .apply(np.prod, raw=True)
            .shift(-(horizon - 1))
            - 1
        )
        future_market = pd.Series(
            panel.index.get_level_values("date").map(future_market_by_date), index=panel.index
        )
        residual = future - beta * future_market

    scaled = residual / (volatility * np.sqrt(horizon)).replace(0, np.nan)
    return scaled.rename("target")


def label_end_dates(index: pd.MultiIndex, horizon: int = 5) -> pd.Series:
    """Return the actual trading-date label end for each row of a date/symbol panel."""
    dates = pd.Index(index.get_level_values("date").unique()).sort_values()
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for position, date in enumerate(dates):
        end_pos = position + horizon
        mapping[pd.Timestamp(date)] = (
            pd.Timestamp(dates[end_pos]) if end_pos < len(dates) else pd.NaT
        )
    return pd.Series(index.get_level_values("date").map(mapping), index=index, name="label_end")


def standardized_to_expected_alpha(
    prediction: pd.Series,
    panel: pd.DataFrame,
    horizon: int = 5,
    volatility_window: int = 20,
) -> pd.Series:
    """Convert model output back to an economically interpretable residual return estimate."""
    vol = trailing_volatility(panel, volatility_window).reindex(prediction.index)
    return (prediction * vol * np.sqrt(horizon)).rename("expected_alpha")
