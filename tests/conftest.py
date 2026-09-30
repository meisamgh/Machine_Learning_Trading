import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def panel():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", "2025-12-31")
    rows = []
    market = rng.normal(0.0001, 0.01, len(dates))
    for symbol_i, symbol in enumerate(("A", "B", "C", "D", "E", "F", "G", "H")):
        idio = rng.normal(0, 0.012, len(dates))
        close = (50 + 10 * symbol_i) * np.cumprod(1 + 0.7 * market + idio)
        open_ = close / (1 + rng.normal(0, 0.004, len(dates)))
        volume = 2_000_000 + 200_000 * symbol_i + rng.integers(0, 200_000, len(dates))
        for position, date in enumerate(dates):
            rows.append(
                (
                    date,
                    symbol,
                    open_[position],
                    max(open_[position], close[position]) * 1.01,
                    min(open_[position], close[position]) * 0.99,
                    close[position],
                    close[position],
                    volume[position],
                    market[position],
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "market_return",
        ],
    ).set_index(["date", "symbol"]).sort_index()
