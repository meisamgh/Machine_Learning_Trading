import pandas as pd

from ml_trading.targets import label_end_dates
from ml_trading.validation import PurgedDateSplit, make_walk_forward_folds


def test_purge_and_embargo_remove_overlap():
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    ends = pd.Series(index.get_level_values("date") + pd.Timedelta(days=10), index=index)
    train, validation, test = PurgedDateSplit(2020, 2021, 5).split(index, ends)
    assert (ends.iloc[train] < pd.Timestamp("2020-01-01")).all()
    assert (ends.iloc[validation] < pd.Timestamp("2021-01-01")).all()
    assert (index[test].get_level_values("date") >= pd.Timestamp("2021-01-06")).all()


def test_walk_forward_generates_unique_chronological_test_blocks(panel):
    ends = label_end_dates(panel.index, 5)
    folds = make_walk_forward_folds(panel.index, ends, min_train_years=3, embargo_days=5)
    test_dates = []
    for fold in folds:
        train_dates = panel.index[fold.train].get_level_values("date")
        validation_dates = panel.index[fold.validation].get_level_values("date")
        fold_test_dates = panel.index[fold.test].get_level_values("date")
        assert train_dates.max() < validation_dates.min() < fold_test_dates.min()
        test_dates.extend(fold_test_dates.unique())
    assert len(test_dates) == len(set(test_dates))
