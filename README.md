# Machine Learning Trading Research

Leakage-aware Python primitives for offline financial machine-learning research. The repository
keeps the original exploratory notebooks as legacy material, then exposes a small tested package for
reviewed factors, forward-alpha targets, purged date splits, and rank-plus-magnitude trading
signals.

This is a research codebase, not a live trading system. It does not include broker execution,
position sizing, capital allocation, or a claim of profitable performance.

## What This Repo Contains

```text
Machine_Learning_Trading/
├── src/ml_trading/        # tested research primitives
├── tests/                 # unit tests for factors, targets, validation, and signals
├── configs/research.yaml  # default research-contract assumptions
├── docs/legacy_audit.md   # audit of the historical notebooks
└── *.ipynb                # legacy exploratory notebooks
```

The reusable package is intentionally narrow:

- `factors.py` computes a reviewed subset of causal Alpha-style features.
- `targets.py` builds continuous forward residual targets aligned to next-open execution.
- `validation.py` provides year-based purged train, validation, and test splits.
- `signals.py` converts predictions into `LONG`, `SHORT`, or `NO_TRADE` decisions only when both
  rank and absolute-alpha filters pass.

## Research Contract

- Features must be known at or before close `t`.
- Entry is assumed at open `t+1`.
- The default exit is close `t+5`.
- Targets are continuous risk-adjusted residual returns, not future absolute prices.
- Test data is not used for preprocessing, feature selection, thresholds, or early stopping.
- A high cross-sectional rank is not enough: predicted alpha must also clear an economic threshold.
- Failed data, model, validation, or research gates should default to `NO_TRADE`.

## Architecture

```text
validated panel
      |
trusted causal factors
      |
continuous residual target
      |
purged train / validation / test split
      |
out-of-sample predictions
      |
rank + magnitude policy
      |
LONG / SHORT / NO_TRADE
```

## Quick Start

```bash
make setup
make validate
```

Or install and test directly:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
```

## Example

```python
from ml_trading import SignalPolicy, continuous_residual_target, make_signals
from ml_trading.factors import compute_trusted_factors

# panel is a MultiIndex DataFrame indexed by date and symbol with OHLCV columns.
features = compute_trusted_factors(panel)
target = continuous_residual_target(panel, horizon=5)

# predictions is a Series indexed by date and symbol.
signals = make_signals(predictions, SignalPolicy(minimum_alpha=0.10))
```

## Data Expectations

Most functions expect a sorted pandas `MultiIndex` panel with `date` and `symbol` levels. The core
OHLCV columns used by the current primitives are:

- `open`
- `high`
- `low`
- `close`
- `adjusted_close`
- `volume`

An optional `market_return` column enables beta-adjusted residual targets. Without it, the target
falls back to volatility-adjusted forward return.

## Validation

The test suite covers the contract-level behavior that matters most for a research foundation:

- factors preserve the input panel index and use only reviewed formulas;
- targets align prediction date, next-open entry, and forward exit without lookahead preprocessing;
- date splits purge overlapping label intervals and apply an embargo;
- signal generation requires both cross-sectional rank and absolute predicted alpha.

Run the full local check with:

```bash
make validate
```

## Legacy Notebooks

The notebooks are retained for auditability and historical context. They are not treated as the
validated implementation surface. See [`docs/legacy_audit.md`](docs/legacy_audit.md) for the main
methodological risks and why the tested package separates reusable primitives from exploratory
analysis.

## Boundary

The larger end-to-end model comparison, cost-aware overlapping-cohort backtest, and offline evidence
gate live in the companion `Alpha-Quant-Research-v2` checkout. This repository is the clean reusable
research-primitives layer and deliberately avoids duplicating a second backtest engine.

## Disclaimer

This project is for education and offline research. Financial markets are noisy, historical tests are
easy to overfit, and no output from this repository should be treated as investment advice.
