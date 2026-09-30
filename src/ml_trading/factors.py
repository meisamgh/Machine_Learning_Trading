"""Causal factor library and cross-sectional preprocessing."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import adjusted_price_panel


def alpha_012(panel: pd.DataFrame) -> pd.Series:
    """Alpha 012: sign(delta(volume, 1)) * -delta(adjusted close, 1), per symbol."""
    px = adjusted_price_panel(panel)
    grouped = px.sort_index().groupby(level="symbol")
    return np.sign(grouped.volume.diff()) * -grouped.px_close.diff()


def alpha_101(panel: pd.DataFrame) -> pd.Series:
    """Alpha 101: adjusted (close - open) / (high - low + epsilon)."""
    px = adjusted_price_panel(panel)
    return (px.px_close - px.px_open) / (px.px_high - px.px_low + 1e-6)


def _grouped_adjusted(panel: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    px = adjusted_price_panel(panel)
    return px, px.groupby(level="symbol", group_keys=False)


def compute_feature_library(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a deliberately compact, diverse, causal daily feature library.

    All features use information known by close t and are suitable for next-open execution.
    Cross-sectional transformations are performed separately by date and therefore never use a
    future date.
    """
    px, grouped = _grouped_adjusted(panel)
    close = px["px_close"]
    open_ = px["px_open"]
    high = px["px_high"]
    low = px["px_low"]

    ret_1 = grouped.px_close.pct_change(fill_method=None)
    log_volume = np.log1p(px["volume"])
    volume_mean_20 = log_volume.groupby(level="symbol").transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )
    volume_std_20 = log_volume.groupby(level="symbol").transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )

    features: dict[str, pd.Series] = {
        "alpha_012": alpha_012(panel),
        "alpha_101": alpha_101(panel),
        "return_1d": ret_1,
        "reversal_1d": -ret_1,
        "intraday_return": close / open_ - 1,
        "range_pct": (high - low) / close,
        "close_location": (close - low) / (high - low).replace(0, np.nan),
        "gap_1d": open_ / grouped.px_close.shift(1) - 1,
        "log_volume_z20": (log_volume - volume_mean_20) / volume_std_20.replace(0, np.nan),
    }

    for horizon in (5, 20, 60):
        features[f"momentum_{horizon}d"] = close / grouped.px_close.shift(horizon) - 1
    for window in (5, 20, 60):
        features[f"volatility_{window}d"] = ret_1.groupby(level="symbol").transform(
            lambda x, w=window: x.rolling(w, min_periods=w).std()
        )
    dollar_volume = close * px["volume"]
    features["log_dollar_volume"] = np.log1p(dollar_volume)
    features["adv20_ratio"] = dollar_volume / dollar_volume.groupby(level="symbol").transform(
        lambda x: x.rolling(20, min_periods=20).mean()
    )

    frame = pd.DataFrame(features, index=panel.index)
    daily_return_median = frame["return_1d"].groupby(level="date").transform("median")
    frame["relative_return_1d"] = frame["return_1d"] - daily_return_median
    mom20_median = frame["momentum_20d"].groupby(level="date").transform("median")
    frame["relative_momentum_20d"] = frame["momentum_20d"] - mom20_median
    return frame.replace([np.inf, -np.inf], np.nan)


TRUSTED_FACTORS = {
    "alpha_012": alpha_012,
    "alpha_101": alpha_101,
}


def compute_trusted_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible reviewed factor subset."""
    return pd.DataFrame({name: function(panel) for name, function in TRUSTED_FACTORS.items()})


def cross_sectional_transform(
    features: pd.DataFrame,
    winsor_quantile: float = 0.01,
    mode: str = "zscore",
    neutralize: pd.Series | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Winsorize and normalize independently within each date.

    ``neutralize`` may contain a sector/industry label indexed like ``features``. Group means are
    removed within each date and label before the final cross-sectional normalization.
    """
    if "date" not in features.index.names:
        raise ValueError("features index must include date")
    result = features.copy()
    selected = list(columns or result.columns)

    def winsorize(group: pd.DataFrame) -> pd.DataFrame:
        lower = group.quantile(winsor_quantile)
        upper = group.quantile(1 - winsor_quantile)
        return group.clip(lower=lower, upper=upper, axis=1)

    result[selected] = result[selected].groupby(level="date", group_keys=False).apply(winsorize)

    if neutralize is not None:
        labels = neutralize.reindex(result.index)
        for column in selected:
            values = result[column]
            group_mean = values.groupby(
                [result.index.get_level_values("date"), labels], dropna=False
            ).transform("mean")
            result[column] = values - group_mean

    if mode == "rank":
        result[selected] = result[selected].groupby(level="date").rank(pct=True) - 0.5
    elif mode == "zscore":
        for column in selected:
            mean = result[column].groupby(level="date").transform("mean")
            std = result[column].groupby(level="date").transform("std").replace(0, np.nan)
            result[column] = (result[column] - mean) / std
    else:
        raise ValueError("mode must be 'rank' or 'zscore'")
    return result
