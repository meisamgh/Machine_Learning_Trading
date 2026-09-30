import numpy as np
import pandas as pd

from ml_trading.monitoring import MonitoringPolicy, monitoring_snapshot, population_stability_index


def test_psi_detects_large_shift():
    rng = np.random.default_rng(42)
    reference = pd.Series(rng.normal(0, 1, 1000))
    current = pd.Series(rng.normal(3, 1, 1000))
    assert population_stability_index(reference, current) > 0.25


def test_monitoring_fail_closed_on_drawdown():
    returns = pd.Series([0.0, -0.05, -0.06, -0.05, 0.01])
    snapshot = monitoring_snapshot(
        returns,
        policy=MonitoringPolicy(max_drawdown=-0.10, min_rolling_sharpe=-100),
    )
    assert snapshot["trading_state"] == "NO_TRADE"
    assert "drawdown" in snapshot["alerts"]
