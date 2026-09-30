"""Cost-aware overlapping-cohort portfolio backtester."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import attach_adv


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 1.0
    half_spread_bps: float = 3.0
    slippage_bps: float = 2.0
    impact_bps_at_1pct_adv: float = 5.0
    borrow_bps_annual: float = 100.0
    max_participation: float = 0.05

    def scaled(self, multiplier: float) -> CostModel:
        """Scale monetary friction assumptions while leaving capacity unchanged."""
        if multiplier <= 0:
            raise ValueError("cost multiplier must be positive")
        return CostModel(
            commission_bps=self.commission_bps * multiplier,
            half_spread_bps=self.half_spread_bps * multiplier,
            slippage_bps=self.slippage_bps * multiplier,
            impact_bps_at_1pct_adv=self.impact_bps_at_1pct_adv * multiplier,
            borrow_bps_annual=self.borrow_bps_annual * multiplier,
            max_participation=self.max_participation,
        )

    def one_way_cost_bps(self, trade_notional: float, adv_dollars: float | None) -> float:
        base = self.commission_bps + self.half_spread_bps + self.slippage_bps
        if adv_dollars is None or not np.isfinite(adv_dollars) or adv_dollars <= 0:
            return base + self.impact_bps_at_1pct_adv
        participation = max(trade_notional / adv_dollars, 0.0)
        impact = self.impact_bps_at_1pct_adv * np.sqrt(participation / 0.01)
        return float(base + impact)

    def fill_fraction(self, trade_notional: float, adv_dollars: float | None) -> float:
        if trade_notional <= 0:
            return 1.0
        if adv_dollars is None or not np.isfinite(adv_dollars) or adv_dollars <= 0:
            return 0.0
        return float(min(1.0, self.max_participation * adv_dollars / trade_notional))


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame


def backtest_overlapping_cohorts(
    panel: pd.DataFrame,
    target_weights: pd.Series,
    horizon: int = 5,
    cost_model: CostModel | None = None,
    initial_equity: float = 1_000_000.0,
) -> BacktestResult:
    """Trade each signal at next open and hold a 1/horizon cohort through close[t+horizon].

    Each day's target portfolio is scaled by ``1/horizon`` so up to ``horizon`` overlapping cohorts
    coexist without multiplying the intended gross exposure. Entry/exit costs, capacity-limited
    partial fills, and short borrow are charged explicitly.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    cost_model = cost_model or CostModel()
    px = attach_adv(panel).sort_index()
    dates = pd.Index(px.index.get_level_values("date").unique()).sort_values()
    date_to_pos = {pd.Timestamp(date): i for i, date in enumerate(dates)}
    pnl = pd.Series(0.0, index=dates, name="net_return")
    gross_pnl = pd.Series(0.0, index=dates, name="gross_return")
    costs = pd.Series(0.0, index=dates, name="cost_return")
    gross_exposure = pd.Series(0.0, index=dates, name="gross_exposure")
    trade_rows: list[dict[str, object]] = []
    equity = initial_equity

    for signal_date, weights_day in target_weights.groupby(level="date"):
        signal_date = pd.Timestamp(signal_date)
        if signal_date not in date_to_pos:
            continue
        signal_pos = date_to_pos[signal_date]
        entry_pos = signal_pos + 1
        exit_pos = signal_pos + horizon
        if entry_pos >= len(dates) or exit_pos >= len(dates):
            continue
        entry_date, exit_date = pd.Timestamp(dates[entry_pos]), pd.Timestamp(dates[exit_pos])

        for index, raw_weight in weights_day.items():
            symbol = index[1]
            if not np.isfinite(raw_weight) or raw_weight == 0:
                continue
            cohort_weight = float(raw_weight) / horizon
            key_entry = (entry_date, symbol)
            if key_entry not in px.index:
                continue
            adv = float(px.loc[key_entry, "adv_dollars"])
            desired_notional = abs(cohort_weight) * equity
            fill_fraction = cost_model.fill_fraction(desired_notional, adv)
            if fill_fraction <= 0:
                continue
            weight = cohort_weight * fill_fraction
            notional = abs(weight) * equity
            one_way_bps = cost_model.one_way_cost_bps(notional, adv)
            entry_cost = abs(weight) * one_way_bps / 10_000
            exit_key = (exit_date, symbol)
            if exit_key not in px.index:
                continue
            exit_adv = float(px.loc[exit_key, "adv_dollars"])
            exit_cost_bps = cost_model.one_way_cost_bps(notional, exit_adv)
            exit_cost = abs(weight) * exit_cost_bps / 10_000
            costs.loc[entry_date] += entry_cost
            costs.loc[exit_date] += exit_cost

            previous_close: float | None = None
            for position in range(entry_pos, exit_pos + 1):
                date = pd.Timestamp(dates[position])
                key = (date, symbol)
                if key not in px.index:
                    break
                row = px.loc[key]
                if position == entry_pos:
                    asset_return = float(row.px_close / row.px_open - 1)
                else:
                    if previous_close is None:
                        break
                    asset_return = float(row.px_close / previous_close - 1)
                previous_close = float(row.px_close)
                contribution = weight * asset_return
                gross_pnl.loc[date] += contribution
                gross_exposure.loc[date] += abs(weight)
                if weight < 0:
                    borrow = abs(weight) * cost_model.borrow_bps_annual / 10_000 / 252
                    costs.loc[date] += borrow

            trade_rows.append(
                {
                    "signal_date": signal_date,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "symbol": symbol,
                    "desired_weight": cohort_weight,
                    "filled_weight": weight,
                    "fill_fraction": fill_fraction,
                    "entry_cost_bps": one_way_bps,
                    "exit_cost_bps": exit_cost_bps,
                }
            )

    pnl[:] = gross_pnl - costs
    daily = pd.concat([gross_pnl, costs, pnl, gross_exposure], axis=1)
    daily["equity"] = initial_equity * (1 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(daily=daily, trades=trades)


def cost_stress_backtests(
    panel: pd.DataFrame,
    target_weights: pd.Series,
    horizon: int = 5,
    base_cost_model: CostModel | None = None,
    multipliers: tuple[float, ...] = (1.0, 1.5, 2.0),
    initial_equity: float = 1_000_000.0,
) -> dict[float, BacktestResult]:
    """Run the identical portfolio under progressively harsher transaction-cost assumptions."""
    base = base_cost_model or CostModel()
    return {
        multiplier: backtest_overlapping_cohorts(
            panel,
            target_weights,
            horizon=horizon,
            cost_model=base.scaled(multiplier),
            initial_equity=initial_equity,
        )
        for multiplier in multipliers
    }
