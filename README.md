# Machine Learning Trading

A leakage-aware, cost-aware research and paper-trading stack for daily cross-sectional machine
learning strategies. The original notebooks remain as legacy experiments, but the supported surface
is the tested `ml_trading` package.

**This repository does not claim profitability.** Its purpose is to make it difficult to accidentally
turn leakage, survivorship bias, unrealistic fills, or repeated backtest selection into a fake alpha
claim.

## What is implemented

- point-in-time universe filtering with fail-closed membership;
- corporate-action-consistent adjusted OHLC handling;
- dataset fingerprints for experiment/model lineage;
- a causal feature library spanning momentum, reversal, volatility, intraday/range, relative-return,
  and liquidity/volume features plus reviewed Alpha-style factors;
- per-date winsorization, rank/z-score transforms, and optional sector neutralization;
- next-open to horizon-close continuous residual targets;
- purged, embargoed expanding walk-forward validation;
- fold-local imputation/scaling and validation-only hyperparameter selection;
- Ridge, ElasticNet, sklearn HistGradientBoosting, plus optional XGBoost/LightGBM challengers;
- strictly out-of-sample prediction assembly and daily rank IC;
- conversion from standardized model output back to expected raw residual return;
- economic signal gating: expected alpha must exceed estimated costs + required edge;
- constrained long/short weights with gross, net, name, sector, beta, and turnover controls;
- overlapping holding-period cohorts;
- costs for commissions, spread, slippage, market impact, short borrow, and ADV-based partial fills;
- Sharpe/drawdown/fill/IC metrics, Deflated-Sharpe-style evidence and CSCV-style PBO diagnostics;
- explicit offline research gates that default to NO_TRADE when evidence is missing or weak;
- a stateful `PaperBroker`, target-vs-actual reconciliation, drift metrics, and fail-closed monitoring;
- unit tests and GitHub Actions CI.

## Research contract

```text
point-in-time panel + membership
          |
          v
corporate-action-consistent prices
          |
          v
causal feature library (known by close t)
          |
          v
cross-sectional processing on date t only
          |
          v
open[t+1] -> close[t+5] residual target
          |
          v
purged walk-forward train / validation / test
          |
          v
strictly OOS predictions
          |
          v
expected raw alpha - expected cost - edge requirement
          |
          v
constrained target weights
          |
          v
overlapping-cohort cost/capacity backtest
          |
          v
research evidence gate
          |
          v
paper broker + reconciliation + monitoring
          |
          v
NO_TRADE on any material gate/health failure
```

The research code intentionally stops at a broker protocol and paper broker. A real broker requires a
separate authenticated adapter, broker-specific order semantics, exchange calendars, and operational
controls; credentials must never be committed to Git.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
make validate
```

For optional tree-boosting challengers:

```bash
python -m pip install -e ".[boosting,dev]"
```

## Canonical panel

Most functions consume a sorted `pandas.MultiIndex` DataFrame indexed by `date, symbol` with:

```text
open, high, low, close, adjusted_close, volume
```

Optional columns include `market_return` and explicit adjusted OHLC fields. If only
`adjusted_close` exists, the package derives a consistent adjustment factor and applies it to O/H/L,
rather than mixing adjusted and raw prices.

For a real historical equity experiment, supply point-in-time membership including delisted names.
Missing membership is treated as not eligible.

## End-to-end API

```python
from ml_trading import ResearchConfig, run_research_pipeline

result = run_research_pipeline(
    panel,
    membership=historical_membership,
    config=ResearchConfig(model_name="ridge"),
)

print(result.walk_forward.fold_report)
print(result.metrics)
print(result.gate_passed, result.gate_failures)
```

The default system can legitimately return zero trades. If forecasts do not clear expected costs or a
research/monitoring gate fails, forcing a position would be a bug.

## Models

Start simple. Ridge is the default falsification baseline. Compare challengers under exactly the same
walk-forward folds, cost assumptions, universe, target, and portfolio constraints. `xgboost` and
`lightgbm` are optional dependencies so the deterministic research spine remains lightweight.

Model preprocessing is fit inside each historical fold. The test block does not influence imputation,
scaling, feature selection, thresholds, early stopping, or hyperparameter choice.

## Backtest realism

`CostModel` includes:

```text
commission
half spread
slippage
ADV-dependent market impact
short borrow
ADV participation cap / partial fills
```

Signals generated after close `t` enter at adjusted open `t+1`. A horizon-5 strategy creates a new
1/5-sized cohort each signal day and holds each cohort through its configured exit close, so overlapping
positions are represented rather than pretending every prediction owns the entire book.

## Evidence and monitoring

Evaluate predictive quality and trading economics separately. Useful outputs include daily rank IC,
net Sharpe, drawdown, fill fraction, cost stress, and stability across walk-forward folds. When many
variants are tried, report the number of trials and use the multiple-testing diagnostics in
`evaluation.py`.

After offline research, use `PaperBroker` and `reconcile_positions` to exercise portfolio state before
any external broker integration. `monitoring_snapshot` checks drift, rolling IC, fill quality, Sharpe,
and drawdown; material breaches recommend `NO_TRADE`.

See [`docs/production_readiness.md`](docs/production_readiness.md) for the research-to-live checklist
and [`docs/legacy_audit.md`](docs/legacy_audit.md) for the historical notebook risks.

## Legacy notebooks

The notebooks are retained for auditability and ideas only. They are not the validated implementation
surface and their historical performance is not accepted as evidence.

## Disclaimer

For education and research only. Financial markets are noisy and non-stationary, transaction costs and
capacity can erase apparent alpha, and backtests are easy to overfit. Nothing in this repository is
investment advice or a guarantee of future performance.
