"""Leakage-aware, cost-aware machine-learning trading research primitives."""

from .backtest import BacktestResult, CostModel, backtest_overlapping_cohorts, cost_stress_backtests
from .data import (
    DataValidationResult,
    adjusted_price_panel,
    apply_point_in_time_universe,
    liquidity_eligibility,
    panel_fingerprint,
    validate_panel,
)
from .evaluation import ResearchGate, performance_metrics
from .factors import (
    alpha_012,
    alpha_101,
    compute_feature_library,
    compute_trusted_factors,
    cross_sectional_transform,
)
from .models import WalkForwardResult, build_model, walk_forward_predict
from .monitoring import MonitoringPolicy, monitoring_snapshot
from .paper import PaperBroker, reconcile_positions
from .pipeline import ResearchConfig, ResearchResult, run_research_pipeline
from .portfolio import PortfolioConstraints, construct_weights
from .signals import SignalPolicy, make_economic_signals, make_signals
from .targets import (
    continuous_residual_target,
    forward_adjusted_return,
    label_end_dates,
    standardized_to_expected_alpha,
)
from .validation import PurgedDateSplit, WalkForwardFold, make_walk_forward_folds

__all__ = [
    "BacktestResult",
    "CostModel",
    "DataValidationResult",
    "MonitoringPolicy",
    "PaperBroker",
    "PortfolioConstraints",
    "PurgedDateSplit",
    "ResearchConfig",
    "ResearchGate",
    "ResearchResult",
    "SignalPolicy",
    "WalkForwardFold",
    "WalkForwardResult",
    "adjusted_price_panel",
    "alpha_012",
    "alpha_101",
    "apply_point_in_time_universe",
    "backtest_overlapping_cohorts",
    "cost_stress_backtests",
    "build_model",
    "compute_feature_library",
    "compute_trusted_factors",
    "construct_weights",
    "continuous_residual_target",
    "cross_sectional_transform",
    "forward_adjusted_return",
    "label_end_dates",
    "liquidity_eligibility",
    "make_economic_signals",
    "make_signals",
    "make_walk_forward_folds",
    "monitoring_snapshot",
    "panel_fingerprint",
    "performance_metrics",
    "reconcile_positions",
    "run_research_pipeline",
    "standardized_to_expected_alpha",
    "validate_panel",
    "walk_forward_predict",
]
