"""Date-based expanding folds with interval purging and calendar embargo."""
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
        train = (dates < validation_start) & (label_end.to_numpy() < validation_start)
        validation = (dates >= validation_start) & (dates < test_start) & (
            label_end.to_numpy() < test_start
        )
        test = (dates >= test_start + pd.Timedelta(days=self.embargo_days)) & (dates < test_end)
        return np.flatnonzero(train), np.flatnonzero(validation), np.flatnonzero(test)
