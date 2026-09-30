"""End-to-end deterministic research pipeline for ML, costs, portfolio, and gates."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .backtest import BacktestResult, CostModel, backtest_overlapping_cohorts, cost_stress_backtests
from .data import (
    apply_point_in_time_universe,
    liquidity_eligibility,
    panel_fingerprint,
    validate_panel,
)
from .evaluation import ResearchGate, deflated_sharpe_probability, performance_metrics
from .factors import compute_feature_library, cross_sectional_transform
from .models import WalkForwardResult, walk_forward_predict
from .portfolio import PortfolioConstraints, construct_weights
from .signals import make_economic_signals
from .targets import (
    continuous_residual_target,
    label_end_dates,
    standardized_to_expected_alpha,
)


@dataclass(frozen=True)
class ResearchConfig:
    horizon: int = 5
    volatility_window: int = 20
    beta_window: int = 60
    min_train_years: int = 3
    embargo_days: int = 5
    model_name: str = "ridge"
    model_param_grid: list[dict] = field(
        default_factory=lambda: [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]
    )
    winsor_quantile: float = 0.01
    transform_mode: str = "zscore"
    long_rank: float = 0.90
    short_rank: float = 0.10
    minimum_net_edge: float = 0.001
    n_model_trials: int = 3


@dataclass
class ResearchResult:
    fingerprint: str
    features: pd.DataFrame
    target: pd.Series
    walk_forward: WalkForwardResult
    economic_signals: pd.DataFrame
    weights: pd.Series
    backtest: BacktestResult
    cost_stress: dict[float, BacktestResult]
    metrics: dict[str, float]
    gate_passed: bool
    gate_failures: list[str]


def run_research_pipeline(
    panel: pd.DataFrame,
    membership: pd.DataFrame | pd.Series | None = None,
    config: ResearchConfig | None = None,
    cost_model: CostModel | None = None,
    constraints: PortfolioConstraints | None = None,
    gate: ResearchGate | None = None,
    sector: pd.Series | None = None,
) -> ResearchResult:
    """Run a fully OOS, cost-aware research experiment.

    This function intentionally has no live-broker side effect. A strategy must first pass the
    offline evidence gate, then be exercised through ``PaperBroker`` before any external broker
    adapter is considered.
    """
    config = config or ResearchConfig()
    cost_model = cost_model or CostModel()
    constraints = constraints or PortfolioConstraints()
    gate = gate or ResearchGate()
    panel = panel.sort_index()
    validate_panel(panel)
    if membership is not None:
        panel = apply_point_in_time_universe(panel, membership)
        validate_panel(panel)

    eligibility = liquidity_eligibility(panel)
    raw_features = compute_feature_library(panel)
    features = cross_sectional_transform(
        raw_features,
        winsor_quantile=config.winsor_quantile,
        mode=config.transform_mode,
        neutralize=sector,
    )
    target = continuous_residual_target(
        panel,
        horizon=config.horizon,
        volatility_window=config.volatility_window,
        beta_window=config.beta_window,
    )
    ends = label_end_dates(panel.index, config.horizon)
    wf = walk_forward_predict(
        features,
        target,
        ends,
        model_name=config.model_name,
        param_grid=config.model_param_grid,
        min_train_years=config.min_train_years,
        embargo_days=config.embargo_days,
    )
    expected_alpha = standardized_to_expected_alpha(
        wf.predictions,
        panel,
        horizon=config.horizon,
        volatility_window=config.volatility_window,
    )
    round_trip_floor = 2 * (
        cost_model.commission_bps + cost_model.half_spread_bps + cost_model.slippage_bps
    ) / 10_000
    economic_signals = make_economic_signals(
        expected_alpha,
        estimated_round_trip_cost=round_trip_floor,
        long_rank=config.long_rank,
        short_rank=config.short_rank,
        minimum_net_edge=config.minimum_net_edge,
        eligible=eligibility.reindex(expected_alpha.index, fill_value=False),
    )
    weights = construct_weights(
        expected_alpha,
        signals=economic_signals,
        constraints=constraints,
        sector=sector,
    )
    backtest = backtest_overlapping_cohorts(
        panel,
        weights,
        horizon=config.horizon,
        cost_model=cost_model,
    )
    stress = cost_stress_backtests(
        panel,
        weights,
        horizon=config.horizon,
        base_cost_model=cost_model,
        multipliers=(1.0, 1.5, 2.0),
    )
    metrics = performance_metrics(
        backtest.daily["net_return"],
        target=target.reindex(wf.predictions.index),
        predictions=wf.predictions,
        trades=backtest.trades,
    )
    metrics["deflated_sharpe_probability"] = deflated_sharpe_probability(
        backtest.daily["net_return"], n_trials=config.n_model_trials
    )
    for multiplier, stressed in stress.items():
        stressed_metrics = performance_metrics(stressed.daily["net_return"])
        metrics[f"sharpe_cost_{multiplier:.1f}x"] = stressed_metrics["sharpe"]
    passed, failures = gate.evaluate(metrics)
    return ResearchResult(
        fingerprint=panel_fingerprint(panel),
        features=features,
        target=target,
        walk_forward=wf,
        economic_signals=economic_signals,
        weights=weights,
        backtest=backtest,
        cost_stress=stress,
        metrics=metrics,
        gate_passed=passed,
        gate_failures=failures,
    )
