# Machine Learning Trading Research

A professional, leakage-aware foundation for offline financial machine-learning research. The
repository preserves the original notebooks as **legacy experiments** and provides a small trusted
Python package for causal factors, continuous forward-alpha targets, purged validation, and
LONG/SHORT/NO-TRADE policies.

This project does not promise profit and contains no live broker integration.

## Research contract

- Features are known at or before close `t`.
- Entry occurs at open `t+1`.
- The default exit is close `t+5`.
- The target is continuous risk-adjusted residual return, never future absolute price.
- Test data is not used for preprocessing, feature selection, thresholds, or early stopping.
- A highly ranked prediction must also exceed an absolute economic threshold to become a trade.
- Failed data, model, or research gates default to `NO_TRADE`.

## Architecture

```text
validated panel
      ↓
trusted causal factors
      ↓
continuous residual target
      ↓
purged train / validation / test
      ↓
OOS predictions
      ↓
rank + magnitude policy
      ↓
LONG / SHORT / NO_TRADE
```

## Quick start

```bash
make setup
make validate
```

Core configuration is in `configs/research.yaml`. The package uses a `src/` layout and has no hidden
notebook dependency.

## Trusted API

```python
from ml_trading import SignalPolicy, continuous_residual_target, make_signals
from ml_trading.factors import compute_trusted_factors

features = compute_trusted_factors(panel)
target = continuous_residual_target(panel, horizon=5)
signals = make_signals(predictions, SignalPolicy(minimum_alpha=0.10))
```

## Repository boundary

The production-quality end-to-end model comparison, cost-aware overlapping-cohort backtest, and
offline evidence gate live in the companion `Alpha-Quant-Research-v2` checkout. This repository is
the clean reusable research-primitives layer. It deliberately does not duplicate a second backtest
engine or claim notebook results as validated performance.

See [`docs/legacy_audit.md`](docs/legacy_audit.md) for the specific methodological issues in the
historical notebooks.
