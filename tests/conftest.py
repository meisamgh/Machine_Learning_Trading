import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def panel():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    rows = []
    market = rng.normal(0, 0.01, len(dates))
    for symbol in ("A", "B", "C", "D"):
        close = 100 * np.cumprod(1 + market + rng.normal(0, 0.01, len(dates)))
        open_ = close / (1 + rng.normal(0, 0.005, len(dates)))
        for position, date in enumerate(dates):
            rows.append(
                (
                    date, symbol, open_[position],
                    max(open_[position], close[position]) * 1.01,
                    min(open_[position], close[position]) * 0.99, close[position],
                    close[position], 1_000_000 + position, market[position],
                )
            )
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close",
        "adjusted_close", "volume", "market_return"]).set_index(["date", "symbol"]).sort_index()
