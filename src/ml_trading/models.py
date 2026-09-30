"""Fold-local model pipelines, tuning, walk-forward prediction, and artifact versioning."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import make_walk_forward_folds


@dataclass(frozen=True)
class ModelArtifactMetadata:
    model_name: str
    data_fingerprint: str
    feature_names: list[str]
    trained_through: str
    parameters: dict[str, Any]
    random_seed: int = 42


def _base_estimator(name: str, params: dict[str, Any], random_seed: int):
    if name == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)))
    if name == "elasticnet":
        return ElasticNet(
            alpha=float(params.get("alpha", 0.001)),
            l1_ratio=float(params.get("l1_ratio", 0.2)),
            max_iter=int(params.get("max_iter", 10000)),
            random_state=random_seed,
        )
    if name == "hist_gb":
        return HistGradientBoostingRegressor(
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            max_iter=int(params.get("max_iter", 200)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            random_state=random_seed,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install ml-trading-research[boosting] for xgboost") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            random_state=random_seed,
            n_jobs=int(params.get("n_jobs", 1)),
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("install ml-trading-research[boosting] for lightgbm") from exc
        return LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            num_leaves=int(params.get("num_leaves", 31)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            random_state=random_seed,
            n_jobs=int(params.get("n_jobs", 1)),
            verbosity=-1,
        )
    raise ValueError(f"unsupported model: {name}")


def build_model(name: str, params: dict[str, Any] | None = None, random_seed: int = 42) -> Pipeline:
    """Build a fold-local preprocessing + regression pipeline."""
    params = params or {}
    estimator = _base_estimator(name, params, random_seed)
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if name in {"ridge", "elasticnet"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def daily_rank_ic(y_true: pd.Series, prediction: pd.Series) -> float:
    """Mean daily Spearman information coefficient."""
    joined = pd.concat([y_true.rename("y"), prediction.rename("p")], axis=1).dropna()
    if joined.empty:
        return float("nan")
    values: list[float] = []
    for _, group in joined.groupby(level="date"):
        if len(group) < 3 or group["y"].nunique() < 2 or group["p"].nunique() < 2:
            continue
        result = spearmanr(group["y"], group["p"])
        if np.isfinite(result.statistic):
            values.append(float(result.statistic))
    return float(np.mean(values)) if values else float("nan")


def _select_params(
    model_name: str,
    param_grid: list[dict[str, Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    random_seed: int,
) -> tuple[dict[str, Any], float]:
    best_params = param_grid[0]
    best_score = -np.inf
    for params in param_grid:
        model = build_model(model_name, params, random_seed)
        model.fit(x_train, y_train)
        prediction = pd.Series(model.predict(x_validation), index=x_validation.index)
        score = daily_rank_ic(y_validation, prediction)
        score_for_compare = -np.inf if not np.isfinite(score) else score
        if score_for_compare > best_score:
            best_score = score_for_compare
            best_params = params
    return best_params, float(best_score)


@dataclass
class WalkForwardResult:
    predictions: pd.Series
    fold_report: pd.DataFrame
    fitted_models: dict[int, Pipeline]


def walk_forward_predict(
    features: pd.DataFrame,
    target: pd.Series,
    label_end: pd.Series,
    model_name: str = "ridge",
    param_grid: list[dict[str, Any]] | None = None,
    min_train_years: int = 3,
    embargo_days: int = 5,
    random_seed: int = 42,
) -> WalkForwardResult:
    """Tune only on each historical validation block and predict each test block exactly once."""
    aligned = features.join(target.rename("target"), how="inner").dropna(subset=["target"])
    x = aligned[features.columns]
    y = aligned["target"]
    ends = label_end.reindex(aligned.index)
    valid_end = ends.notna()
    x, y, ends = x.loc[valid_end], y.loc[valid_end], ends.loc[valid_end]
    folds = make_walk_forward_folds(x.index, ends, min_train_years, embargo_days)
    param_grid = param_grid or [{}]
    all_predictions: list[pd.Series] = []
    reports: list[dict[str, Any]] = []
    models: dict[int, Pipeline] = {}

    for fold in folds:
        train_idx, val_idx, test_idx = fold.train, fold.validation, fold.test
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
        best_params, validation_ic = _select_params(
            model_name,
            param_grid,
            x_train,
            y_train,
            x_val,
            y_val,
            random_seed + fold.fold,
        )
        # Once hyperparameters are frozen, all history strictly before the test block may train the
        # final fold model. Purging guarantees validation labels end before the test year.
        x_refit = pd.concat([x_train, x_val]).sort_index()
        y_refit = pd.concat([y_train, y_val]).sort_index()
        model = build_model(model_name, best_params, random_seed + fold.fold)
        model.fit(x_refit, y_refit)
        prediction = pd.Series(model.predict(x_test), index=x_test.index, name="prediction")
        test_ic = daily_rank_ic(y_test, prediction)
        all_predictions.append(prediction)
        models[fold.fold] = model
        reports.append(
            {
                "fold": fold.fold,
                "validation_year": fold.validation_year,
                "test_year": fold.test_year,
                "train_rows": len(x_train),
                "validation_rows": len(x_val),
                "test_rows": len(x_test),
                "validation_ic": validation_ic,
                "test_ic": test_ic,
                "parameters": json.dumps(best_params, sort_keys=True),
            }
        )
    predictions = pd.concat(all_predictions).sort_index()
    if predictions.index.has_duplicates:
        raise RuntimeError("walk-forward predictions must be strictly out-of-sample and unique")
    return WalkForwardResult(predictions, pd.DataFrame(reports), models)


def save_model_artifact(
    model: Pipeline,
    path: str | Path,
    metadata: ModelArtifactMetadata,
) -> tuple[Path, Path]:
    """Persist a model plus immutable metadata binding it to data/features/config."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8"
    )
    return model_path, metadata_path
