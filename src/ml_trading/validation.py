"""Leakage-aware purged chronological validation and walk-forward fold generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedDateSplit:
    validation_year: int
    test_year: int
    embargo_days: int = 5

    def split(
        self, index: pd.MultiIndex, label_end: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dates = index.get_level_values("date")
        validation_start = pd.Timestamp(f"{self.validation_year}-01-01")
        test_start = pd.Timestamp(f"{self.test_year}-01-01")
        test_end = pd.Timestamp(f"{self.test_year + 1}-01-01")
        ends = pd.to_datetime(label_end.reindex(index)).to_numpy()
        train = (dates < validation_start) & (ends < np.datetime64(validation_start))
        validation = (dates >= validation_start) & (dates < test_start) & (
            ends < np.datetime64(test_start)
        )
        test = (dates >= test_start + pd.Timedelta(days=self.embargo_days)) & (dates < test_end)
        return np.flatnonzero(train), np.flatnonzero(validation), np.flatnonzero(test)


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    validation_year: int
    test_year: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_walk_forward_folds(
    index: pd.MultiIndex,
    label_end: pd.Series,
    min_train_years: int = 3,
    embargo_days: int = 5,
) -> list[WalkForwardFold]:
    """Generate expanding annual folds: history -> validation year -> test year."""
    if min_train_years < 1:
        raise ValueError("min_train_years must be >= 1")
    years = sorted(pd.Index(index.get_level_values("date").year).unique())
    if len(years) < min_train_years + 2:
        raise ValueError("not enough calendar years for requested walk-forward design")
    folds: list[WalkForwardFold] = []
    for test_position in range(min_train_years + 1, len(years)):
        validation_year = int(years[test_position - 1])
        test_year = int(years[test_position])
        splitter = PurgedDateSplit(validation_year, test_year, embargo_days)
        train, validation, test = splitter.split(index, label_end)
        if len(train) and len(validation) and len(test):
            folds.append(
                WalkForwardFold(
                    fold=len(folds),
                    validation_year=validation_year,
                    test_year=test_year,
                    train=train,
                    validation=validation,
                    test=test,
                )
            )
    if not folds:
        raise ValueError("walk-forward configuration produced no non-empty folds")
    return folds
