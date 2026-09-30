"""Deterministic portfolio construction with exposure and concentration controls."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    gross_limit: float = 1.0
    net_target: float = 0.0
    max_weight: float = 0.02
    long_quantile: float = 0.90
    short_quantile: float = 0.10
    max_sector_gross: float = 0.25
    max_abs_beta: float = 0.10
    max_one_way_turnover: float = 0.50


def _cap_and_rescale(weights: pd.Series, gross_target: float, max_weight: float) -> pd.Series:
    result = weights.copy().fillna(0.0)
    if gross_target <= 0 or result.abs().sum() == 0:
        return result * 0.0
    for _ in range(20):
        result = result.clip(-max_weight, max_weight)
        gross = result.abs().sum()
        if gross == 0:
            return result
        scale = gross_target / gross
        candidate = result * scale
        if candidate.abs().max() <= max_weight + 1e-12:
            return candidate
        result = candidate
    return result.clip(-max_weight, max_weight)


def construct_weights(
    expected_alpha: pd.Series,
    signals: pd.DataFrame | None = None,
    constraints: PortfolioConstraints | None = None,
    sector: pd.Series | None = None,
    beta: pd.Series | None = None,
) -> pd.Series:
    """Construct daily market-neutral long/short weights from expected alpha.

    The function fails closed: rows marked ineligible/NO_TRADE never receive weight. Sector caps and
    beta de-risking are deterministic post-processing constraints, not learned transformations.
    """
    constraints = constraints or PortfolioConstraints()
    if constraints.gross_limit <= 0 or constraints.max_weight <= 0:
        raise ValueError("gross_limit and max_weight must be positive")
    result = pd.Series(0.0, index=expected_alpha.index, name="target_weight")

    for date, alpha_day in expected_alpha.groupby(level="date"):
        idx = alpha_day.index
        allowed = pd.Series(True, index=idx)
        if signals is not None:
            day_signals = signals.reindex(idx)
            allowed &= day_signals["decision"].isin(["LONG", "SHORT"])
        valid_alpha = alpha_day[allowed & alpha_day.notna()]
        if len(valid_alpha) < 4:
            continue
        ranks = valid_alpha.rank(pct=True)
        long_idx = ranks[ranks >= constraints.long_quantile].index
        short_idx = ranks[ranks <= constraints.short_quantile].index
        if len(long_idx) == 0 or len(short_idx) == 0:
            continue

        long_score = valid_alpha.loc[long_idx].clip(lower=0)
        short_score = (-valid_alpha.loc[short_idx]).clip(lower=0)
        if long_score.sum() == 0:
            long_score[:] = 1.0
        if short_score.sum() == 0:
            short_score[:] = 1.0
        half_gross = constraints.gross_limit / 2
        day_weights = pd.Series(0.0, index=idx)
        day_weights.loc[long_idx] = long_score / long_score.sum() * half_gross
        day_weights.loc[short_idx] = -short_score / short_score.sum() * half_gross

        if sector is not None:
            sector_day = sector.reindex(idx)
            for _, members in sector_day.groupby(sector_day, dropna=False):
                member_idx = members.index
                gross = day_weights.loc[member_idx].abs().sum()
                if gross > constraints.max_sector_gross:
                    day_weights.loc[member_idx] *= constraints.max_sector_gross / gross

        if beta is not None:
            beta_day = beta.reindex(idx).fillna(0.0)
            portfolio_beta = float((day_weights * beta_day).sum())
            beta_norm = float((beta_day**2).sum())
            if abs(portfolio_beta) > constraints.max_abs_beta and beta_norm > 0:
                target_beta = np.sign(portfolio_beta) * constraints.max_abs_beta
                day_weights -= ((portfolio_beta - target_beta) / beta_norm) * beta_day

        if len(day_weights):
            day_weights -= (day_weights.sum() - constraints.net_target) / len(day_weights)
        day_weights = _cap_and_rescale(
            day_weights, constraints.gross_limit, constraints.max_weight
        )
        result.loc[idx] = day_weights

    dates = pd.Index(result.index.get_level_values("date").unique()).sort_values()
    previous = pd.Series(dtype=float)
    for date in dates:
        day = result.xs(date, level="date").copy()
        symbols = day.index.union(previous.index)
        day_full = day.reindex(symbols, fill_value=0.0)
        prev_full = previous.reindex(symbols, fill_value=0.0)
        turnover = float((day_full - prev_full).abs().sum() / 2)
        if turnover > constraints.max_one_way_turnover > 0:
            scale = constraints.max_one_way_turnover / turnover
            day_full = prev_full + scale * (day_full - prev_full)
        result.loc[(date, slice(None))] = day_full.reindex(day.index).to_numpy()
        previous = day_full
    return result


def one_way_turnover(weights: pd.Series) -> pd.Series:
    """Daily one-way turnover for target-weight portfolios."""
    frame = weights.unstack("symbol").fillna(0.0).sort_index()
    return (frame.diff().abs().sum(axis=1) / 2).fillna(frame.iloc[0].abs().sum() / 2)
