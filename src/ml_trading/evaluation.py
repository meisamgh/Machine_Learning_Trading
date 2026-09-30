"""Predictive, economic, and multiple-testing-aware strategy evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from .models import daily_rank_ic


def annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    clean = returns.dropna()
    std = clean.std(ddof=1)
    if len(clean) > 1 and std > 0:
        return float(np.sqrt(periods) * clean.mean() / std)
    return float("nan")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()) if len(drawdown) else float("nan")


def performance_metrics(
    returns: pd.Series,
    target: pd.Series | None = None,
    predictions: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    clean = returns.dropna()
    metrics = {
        "annualized_return": float((1 + clean.mean()) ** 252 - 1) if len(clean) else float("nan"),
        "annualized_volatility": (
            float(clean.std(ddof=1) * np.sqrt(252))
            if len(clean) > 1
            else float("nan")
        ),
        "sharpe": annualized_sharpe(clean),
        "max_drawdown": max_drawdown(clean),
        "hit_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
    }
    if target is not None and predictions is not None:
        metrics["mean_rank_ic"] = daily_rank_ic(target, predictions)
    if trades is not None and len(trades):
        metrics["mean_fill_fraction"] = float(trades["fill_fraction"].mean())
        metrics["trade_count"] = float(len(trades))
    return metrics


def deflated_sharpe_probability(returns: pd.Series, n_trials: int = 1) -> float:
    """Probability a Sharpe exceeds the expected maximum selected from ``n_trials``.

    Implements the Bailey/Lopez-de-Prado deflated-Sharpe idea using the finite-sample probabilistic
    Sharpe denominator and an expected maximum Sharpe benchmark. Returns a probability in [0, 1].
    """
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return float("nan")
    sr = np.mean(r) / np.std(r, ddof=1)
    if n_trials <= 1:
        benchmark = 0.0
    else:
        sigma_sr = 1 / np.sqrt(len(r) - 1)
        gamma = 0.5772156649015329
        benchmark = sigma_sr * (
            (1 - gamma) * norm.ppf(1 - 1 / n_trials)
            + gamma * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    denominator = np.sqrt(
        max(
            1e-12,
            1
            - skew(r, bias=False) * sr
            + ((kurtosis(r, fisher=False, bias=False) - 1) / 4) * sr**2,
        )
    )
    z = (sr - benchmark) * np.sqrt(len(r) - 1) / denominator
    return float(norm.cdf(z))


def probability_of_backtest_overfitting(
    strategy_returns: pd.DataFrame,
    n_slices: int = 8,
) -> float:
    """Approximate CSCV probability of backtest overfitting across candidate strategies."""
    data = strategy_returns.dropna(how="any")
    if data.shape[1] < 2 or len(data) < n_slices or n_slices < 4 or n_slices % 2:
        return float("nan")
    slices = np.array_split(np.arange(len(data)), n_slices)
    logits: list[float] = []
    for train_slice_ids in combinations(range(n_slices), n_slices // 2):
        train_ids = np.concatenate([slices[i] for i in train_slice_ids])
        test_ids = np.concatenate([slices[i] for i in range(n_slices) if i not in train_slice_ids])
        train_scores = data.iloc[train_ids].apply(annualized_sharpe)
        winner = train_scores.idxmax()
        test_scores = data.iloc[test_ids].apply(annualized_sharpe).sort_values()
        rank = int(test_scores.index.get_loc(winner)) + 1
        percentile = (rank - 0.5) / len(test_scores)
        logits.append(float(np.log(percentile / (1 - percentile))))
    return float(np.mean(np.asarray(logits) <= 0)) if logits else float("nan")


@dataclass(frozen=True)
class ResearchGate:
    min_oos_sharpe: float = 0.0
    min_mean_rank_ic: float = 0.0
    max_drawdown: float = -0.30
    min_deflated_sharpe_probability: float = 0.90
    min_fill_fraction: float = 0.80

    def evaluate(self, metrics: dict[str, float]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        checks = {
            "sharpe": (metrics.get("sharpe"), lambda x: x >= self.min_oos_sharpe),
            "mean_rank_ic": (
                metrics.get("mean_rank_ic"),
                lambda x: x >= self.min_mean_rank_ic,
            ),
            "max_drawdown": (
                metrics.get("max_drawdown"),
                lambda x: x >= self.max_drawdown,
            ),
            "deflated_sharpe_probability": (
                metrics.get("deflated_sharpe_probability"),
                lambda x: x >= self.min_deflated_sharpe_probability,
            ),
            "mean_fill_fraction": (
                metrics.get("mean_fill_fraction"),
                lambda x: x >= self.min_fill_fraction,
            ),
        }
        for name, (value, predicate) in checks.items():
            if value is None or not np.isfinite(value) or not predicate(float(value)):
                failures.append(name)
        return len(failures) == 0, failures
