import pandas as pd

from ml_trading.validation import PurgedDateSplit


def test_purge_and_embargo_remove_overlap():
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    ends = pd.Series(index.get_level_values("date") + pd.Timedelta(days=10), index=index)
    train, validation, test = PurgedDateSplit(2020, 2021, 5).split(index, ends)
    assert (ends.iloc[train] < pd.Timestamp("2020-01-01")).all()
    assert (ends.iloc[validation] < pd.Timestamp("2021-01-01")).all()
    assert (index[test].get_level_values("date") >= pd.Timestamp("2021-01-06")).all()
