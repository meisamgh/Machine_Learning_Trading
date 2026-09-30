"""Prediction, execution, drift, and risk monitoring with fail-closed alerts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import annualized_sharpe, max_drawdown
from .models import daily_rank_ic


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins."""
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < bins or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(ref, edges, include_lowest=True).value_counts(sort=False, normalize=True)
    cur_counts = pd.cut(cur, edges, include_lowest=True).value_counts(sort=False, normalize=True)
    ref_pct = ref_counts.to_numpy() + 1e-6
    cur_pct = cur_counts.to_numpy() + 1e-6
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass(frozen=True)
class MonitoringPolicy:
    max_prediction_psi: float = 0.25
    max_feature_psi: float = 0.25
    max_drawdown: float = -0.15
    min_rolling_sharpe: float = -0.50
    min_rolling_rank_ic: float = -0.02
    min_fill_fraction: float = 0.80


def monitoring_snapshot(
    returns: pd.Series,
    predictions: pd.Series | None = None,
    realized_target: pd.Series | None = None,
    reference_predictions: pd.Series | None = None,
    feature_reference: pd.DataFrame | None = None,
    feature_current: pd.DataFrame | None = None,
    fills: pd.Series | None = None,
    policy: MonitoringPolicy | None = None,
) -> dict[str, object]:
    """Build a health snapshot and explicit NO_TRADE recommendation on material breaches."""
    policy = policy or MonitoringPolicy()
    recent_returns = returns.dropna().tail(63)
    metrics: dict[str, float] = {
        "rolling_sharpe": annualized_sharpe(recent_returns),
        "drawdown": max_drawdown(returns),
    }
    if predictions is not None and realized_target is not None:
        metrics["rolling_rank_ic"] = daily_rank_ic(
            realized_target.reindex(predictions.index), predictions
        )
    if predictions is not None and reference_predictions is not None:
        metrics["prediction_psi"] = population_stability_index(
            reference_predictions, predictions
        )
    if fills is not None and len(fills):
        metrics["mean_fill_fraction"] = float(fills.dropna().mean())
    if feature_reference is not None and feature_current is not None:
        feature_psis = {
            column: population_stability_index(
                feature_reference[column], feature_current[column]
            )
            for column in feature_reference.columns.intersection(feature_current.columns)
        }
        metrics["max_feature_psi"] = (
            float(np.nanmax(list(feature_psis.values())))
            if feature_psis
            else float("nan")
        )

    alerts: list[str] = []
    checks = [
        ("prediction_psi", lambda x: x <= policy.max_prediction_psi),
        ("max_feature_psi", lambda x: x <= policy.max_feature_psi),
        ("drawdown", lambda x: x >= policy.max_drawdown),
        ("rolling_sharpe", lambda x: x >= policy.min_rolling_sharpe),
        ("rolling_rank_ic", lambda x: x >= policy.min_rolling_rank_ic),
        ("mean_fill_fraction", lambda x: x >= policy.min_fill_fraction),
    ]
    for name, predicate in checks:
        value = metrics.get(name)
        if value is not None and np.isfinite(value) and not predicate(float(value)):
            alerts.append(name)
    return {
        "metrics": metrics,
        "alerts": alerts,
        "trading_state": "NO_TRADE" if alerts else "HEALTHY",
    }
