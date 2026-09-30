# Production-readiness contract

This repository deliberately separates **research**, **paper execution**, and a future external
broker adapter. Passing a historical backtest is never sufficient to authorize live capital.

## Required sequence

1. Validate point-in-time market/universe data and record its dataset fingerprint.
2. Build only close-t causal features and a next-open aligned forward target.
3. Produce predictions exclusively through purged chronological walk-forward folds.
4. Select hyperparameters on the validation block; evaluate each test block once.
5. Convert standardized forecasts back to expected return before comparing them with costs.
6. Construct constrained weights and simulate overlapping cohorts with partial fills, spread,
   commission, slippage, impact, borrow, and capacity limits.
7. Stress costs and reject strategies that are unstable across folds/regimes/sub-periods.
8. Apply the offline ResearchGate. A failed or incomplete gate means NO_TRADE.
9. Run the exact feature/signal/portfolio code through PaperBroker and reconciliation.
10. Monitor prediction/feature drift, realized IC, fill quality, Sharpe, and drawdown. A material
    monitoring breach means NO_TRADE.
11. Only after paper behavior matches the research assumptions should a separate authenticated
    broker adapter be considered. Broker-specific credentials and order-routing code are purposely
    not stored in this repository.

## Data requirements

A serious equity experiment should include point-in-time historical membership, delisted securities,
corporate actions, adjusted or consistently normalized OHLC prices, volume, and—where the strategy
needs them—sector/industry labels and benchmark returns. Missing membership is treated as ineligible.

`panel_fingerprint` binds an experiment/model artifact to its actual input values. For stronger
lineage in a team environment, keep immutable data snapshots in object storage or a data-versioning
system and persist the fingerprint with every run.

## Validation requirements

Random train/test splitting is prohibited for time-series trading research. The implemented annual
walk-forward scheme is expanding:

```
train history -> validation year -> embargo -> test year
```

Forward-label intervals are purged from train and validation boundaries. Preprocessing is inside the
model pipeline, so imputation and scaling are fit only on the historical training data for that model.
Cross-sectional normalization operates only within one date.

## Economic requirements

A prediction is not a trade. A candidate must clear an estimated round-trip cost plus a minimum edge
in the same raw-return units. The backtester then applies ADV-aware participation and impact, so a
large target can be partially filled rather than unrealistically assuming unlimited liquidity.

The default model is Ridge because a weak linear baseline is a useful falsification test. Optional
challengers include ElasticNet, sklearn histogram gradient boosting, XGBoost, and LightGBM. Added
complexity should be retained only if strictly out-of-sample economic results improve robustly.

## Research integrity

When many models, targets, features, or hyperparameters are tried, report the number of trials and use
multiple-testing-aware evidence such as `deflated_sharpe_probability`. When comparing many complete
strategy variants, `probability_of_backtest_overfitting` provides a CSCV-style overfitting diagnostic.
These statistics complement rather than replace economic reasoning, sub-period checks, and live paper
validation.
