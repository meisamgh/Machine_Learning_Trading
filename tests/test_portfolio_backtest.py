import numpy as np
import pandas as pd

from ml_trading.backtest import CostModel, backtest_overlapping_cohorts, cost_stress_backtests
from ml_trading.portfolio import PortfolioConstraints, construct_weights
from ml_trading.signals import make_economic_signals


def test_portfolio_respects_gross_and_name_caps(panel):
    dates = panel.index.get_level_values("date").unique()[-20:]
    index = panel.loc[(dates, slice(None)), :].index
    alpha = pd.Series(np.tile(np.linspace(-0.01, 0.01, 8), len(dates)), index=index)
    signals = make_economic_signals(alpha, 0.0001, 0.75, 0.25, 0.0001)
    constraints = PortfolioConstraints(
        gross_limit=1.0, max_weight=0.20, long_quantile=0.75, short_quantile=0.25
    )
    weights = construct_weights(alpha, signals=signals, constraints=constraints)
    gross = weights.abs().groupby(level="date").sum()
    assert (gross <= 1.0 + 1e-9).all()
    assert weights.abs().max() <= 0.20 + 1e-9


def test_backtest_applies_costs_and_partial_fills(panel):
    dates = panel.index.get_level_values("date").unique()[-30:-10]
    index = panel.loc[(dates, slice(None)), :].index
    weights = pd.Series(0.0, index=index)
    weights.loc[(slice(None), "A")] = 0.5
    weights.loc[(slice(None), "B")] = -0.5
    model = CostModel(max_participation=0.00001)
    result = backtest_overlapping_cohorts(panel, weights, horizon=5, cost_model=model)
    assert len(result.trades) > 0
    assert (result.daily["cost_return"] >= 0).all()
    assert (result.trades["fill_fraction"] <= 1).all()
    assert (result.trades["fill_fraction"] < 1).any()


def test_cost_stress_increases_total_cost(panel):
    dates = panel.index.get_level_values("date").unique()[-30:-10]
    index = panel.loc[(dates, slice(None)), :].index
    weights = pd.Series(0.0, index=index)
    weights.loc[(slice(None), "A")] = 0.10
    weights.loc[(slice(None), "B")] = -0.10
    results = cost_stress_backtests(panel, weights, horizon=5, multipliers=(1.0, 2.0))
    assert results[2.0].daily["cost_return"].sum() >= results[1.0].daily["cost_return"].sum()
