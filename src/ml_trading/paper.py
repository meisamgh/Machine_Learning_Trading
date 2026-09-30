"""Broker abstraction, paper execution, and position reconciliation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .backtest import CostModel


class Broker(Protocol):
    def positions(self) -> pd.Series: ...

    def equity(self) -> float: ...

    def rebalance(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        adv_dollars: pd.Series | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame: ...


@dataclass
class PaperBroker:
    """Simple stateful paper broker for dry-runs of target-weight strategies."""

    initial_cash: float = 1_000_000.0
    cost_model: CostModel = field(default_factory=CostModel)
    cash: float = field(init=False)
    shares: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False)
    last_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def equity(self) -> float:
        marked = self.shares * self.last_prices.reindex(self.shares.index)
        market_value = float(marked.fillna(0).sum())
        return self.cash + market_value

    def positions(self) -> pd.Series:
        equity = self.equity()
        if equity <= 0 or self.shares.empty:
            return pd.Series(dtype=float, name="weight")
        market_value = self.shares * self.last_prices.reindex(self.shares.index)
        return (market_value / equity).rename("weight")

    def mark(self, prices: pd.Series) -> None:
        self.last_prices = prices.astype(float).copy()

    def rebalance(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        adv_dollars: pd.Series | None = None,
        timestamp: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        prices = prices.astype(float)
        if (prices <= 0).any():
            raise ValueError("paper broker prices must be positive")
        self.mark(prices)
        equity_before = self.equity()
        symbols = target_weights.index.union(self.shares.index)
        current_shares = self.shares.reindex(symbols, fill_value=0.0)
        px = prices.reindex(symbols)
        valid = px.notna() & (px > 0)
        desired_value = target_weights.reindex(symbols, fill_value=0.0) * equity_before
        current_value = current_shares * px
        trade_value = (desired_value - current_value).where(valid, 0.0)
        rows: list[dict[str, object]] = []

        for symbol, desired_trade_value in trade_value.items():
            if desired_trade_value == 0 or not np.isfinite(desired_trade_value):
                continue
            price = float(px.loc[symbol])
            adv = None
            if adv_dollars is not None and symbol in adv_dollars.index:
                adv = float(adv_dollars.loc[symbol])
            fill_fraction = self.cost_model.fill_fraction(abs(float(desired_trade_value)), adv)
            if fill_fraction <= 0:
                continue
            filled_value = float(desired_trade_value) * fill_fraction
            shares_delta = filled_value / price
            cost_bps = self.cost_model.one_way_cost_bps(abs(filled_value), adv)
            fee = abs(filled_value) * cost_bps / 10_000
            self.cash -= filled_value + fee
            current_shares.loc[symbol] += shares_delta
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "trade_value": filled_value,
                    "shares_delta": shares_delta,
                    "price": price,
                    "fill_fraction": fill_fraction,
                    "cost_bps": cost_bps,
                    "fee": fee,
                }
            )
        self.shares = current_shares[current_shares.abs() > 1e-12]
        self.mark(prices)
        return pd.DataFrame(rows)


def reconcile_positions(
    desired_weights: pd.Series,
    actual_weights: pd.Series,
    tolerance: float = 0.002,
) -> pd.DataFrame:
    """Report material target-vs-broker position mismatches."""
    symbols = desired_weights.index.union(actual_weights.index)
    desired = desired_weights.reindex(symbols, fill_value=0.0)
    actual = actual_weights.reindex(symbols, fill_value=0.0)
    difference = actual - desired
    result = pd.DataFrame({"desired": desired, "actual": actual, "difference": difference})
    result["breach"] = result["difference"].abs() > tolerance
    return result
