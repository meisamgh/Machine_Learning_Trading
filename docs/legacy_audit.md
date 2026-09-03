# Legacy notebook audit

The original notebooks remain unchanged as historical experiments. They are not imported by the
trusted package.

| Finding | Evidence | Risk | Professional replacement |
|---|---|---|---|
| Test-set early stopping | `Copy_of_TA_lib.ipynb` passes `X_test, y_test` to XGBoost `eval_set` | Test results influence training | Separate train, validation, and test blocks |
| Global regime fitting | `RegimeChange.ipynb` fits HMMs on the complete series | Future distribution informs past regimes | Fit every learned regime model inside each training fold |
| Global clustering | `Clustering_candle.ipynb` fits KMeans before a robust walk-forward study | Future observations determine cluster geometry | Fold-local transformer and explicit OOS predictions |
| Forward labels without purging | `Alpha_Factor_Anlysis.ipynb` uses `shift(-1)` | Adjacent folds can share label intervals | `PurgedDateSplit` removes overlapping labels and applies embargo |
| Same-bar execution ambiguity | `BackTest.ipynb` joins model positions directly to prices | Close-derived signals may trade at the same close | Signal after close, entry at next open |
| Notebook-only environment | Cells install dependencies and download changing external data | Runs are not reproducible | Versioned package metadata, configuration, tests, and local cache boundary |
| Unvalidated factor catalog | `Alpha_101` and `alpha101_functions.py` contain broad formula utilities | Formula mistranslation can masquerade as alpha | Small trusted registry with formula and causality tests |

No profitability claim from the legacy notebooks is treated as validated evidence.
