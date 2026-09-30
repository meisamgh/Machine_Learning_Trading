"""Point-in-time data validation, corporate-action normalization, and universe controls."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("open", "high", "low", "close", "adjusted_close", "volume")


@dataclass(frozen=True)
class DataValidationResult:
    rows: int
    symbols: int
    start: pd.Timestamp
    end: pd.Timestamp
    fingerprint: str


def validate_panel(panel: pd.DataFrame) -> DataValidationResult:
    """Validate the canonical date/symbol panel and return immutable dataset metadata."""
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "symbol"]:
        raise ValueError("panel must use a MultiIndex named ['date', 'symbol']")
    missing = [column for column in REQUIRED_OHLCV if column not in panel]
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    if panel.index.has_duplicates:
        raise ValueError("panel contains duplicate date/symbol rows")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel must be sorted by date and symbol")
    if (panel[["open", "high", "low", "close", "adjusted_close"]] <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    if (panel["volume"] < 0).any():
        raise ValueError("volume cannot be negative")
    dates = panel.index.get_level_values("date")
    return DataValidationResult(
        rows=len(panel),
        symbols=panel.index.get_level_values("symbol").nunique(),
        start=pd.Timestamp(dates.min()),
        end=pd.Timestamp(dates.max()),
        fingerprint=panel_fingerprint(panel),
    )


def panel_fingerprint(panel: pd.DataFrame) -> str:
    """Stable content fingerprint used to bind experiments/models to an input snapshot."""
    ordered = panel.sort_index().sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(ordered, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def adjusted_price_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Return internally consistent corporate-action-adjusted OHLC prices.

    When vendors only provide ``adjusted_close``, derive an adjustment factor from
    adjusted_close/close and apply the same factor to O/H/L. This avoids mixing raw entry prices
    with adjusted exit/volatility prices around splits and dividends.
    """
    validate_panel(panel)
    result = panel.copy()
    factor = result["adjusted_close"] / result["close"]
    factor = factor.replace([np.inf, -np.inf], np.nan)
    for column in ("open", "high", "low", "close"):
        explicit = f"adjusted_{column}"
        if explicit in result:
            result[f"px_{column}"] = result[explicit]
        elif column == "close":
            result[f"px_{column}"] = result["adjusted_close"]
        else:
            result[f"px_{column}"] = result[column] * factor
    return result


def apply_point_in_time_universe(
    panel: pd.DataFrame,
    membership: pd.DataFrame | pd.Series,
) -> pd.DataFrame:
    """Filter a panel with historical membership known for each date/symbol.

    Supported membership forms:
    1. MultiIndex(date, symbol) Series/DataFrame with a boolean ``in_universe`` value.
    2. Interval table with columns ``symbol``, ``start_date``, and optional ``end_date``.

    Missing membership is treated as *not eligible* (fail closed).
    """
    if isinstance(membership.index, pd.MultiIndex):
        if isinstance(membership, pd.DataFrame):
            if "in_universe" not in membership:
                raise ValueError("membership DataFrame must include in_universe")
            mask = membership["in_universe"]
        else:
            mask = membership
        aligned = mask.astype(bool).reindex(panel.index, fill_value=False)
        return panel.loc[aligned.to_numpy()].copy()

    required = {"symbol", "start_date"}
    if not required.issubset(membership.columns):
        raise ValueError("interval membership requires symbol and start_date columns")
    intervals = membership.copy()
    intervals["start_date"] = pd.to_datetime(intervals["start_date"])
    if "end_date" not in intervals:
        intervals["end_date"] = pd.NaT
    else:
        intervals["end_date"] = pd.to_datetime(intervals["end_date"])

    dates = panel.index.get_level_values("date")
    symbols = panel.index.get_level_values("symbol")
    eligible = np.zeros(len(panel), dtype=bool)
    for row in intervals.itertuples(index=False):
        end = pd.Timestamp.max if pd.isna(row.end_date) else pd.Timestamp(row.end_date)
        eligible |= (symbols == row.symbol) & (dates >= row.start_date) & (dates <= end)
    return panel.loc[eligible].copy()


def liquidity_eligibility(
    panel: pd.DataFrame,
    adv_window: int = 20,
    min_price: float = 5.0,
    min_adv_dollars: float = 1_000_000.0,
    min_history: int = 60,
) -> pd.Series:
    """Causal daily eligibility mask using only information available through close t."""
    px = adjusted_price_panel(panel)
    dollar_volume = px["px_close"] * px["volume"]
    adv = dollar_volume.groupby(level="symbol").transform(
        lambda x: x.rolling(adv_window, min_periods=adv_window).mean()
    )
    history = panel.groupby(level="symbol").cumcount() + 1
    eligible = (
        (px["px_close"] >= min_price)
        & (adv >= min_adv_dollars)
        & (history >= min_history)
    )
    return eligible.rename("eligible")


def attach_adv(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Attach causal adjusted dollar ADV, useful for execution/capacity models."""
    result = adjusted_price_panel(panel)
    dollar_volume = result["px_close"] * result["volume"]
    result["adv_dollars"] = dollar_volume.groupby(level="symbol").transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    return result
