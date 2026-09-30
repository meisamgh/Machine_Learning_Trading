from ml_trading.factors import compute_feature_library, cross_sectional_transform
from ml_trading.models import walk_forward_predict
from ml_trading.targets import continuous_residual_target, label_end_dates


def test_walk_forward_predictions_are_unique_and_oos(panel):
    features = cross_sectional_transform(compute_feature_library(panel), mode="rank")
    target = continuous_residual_target(panel, horizon=5)
    ends = label_end_dates(panel.index, 5)
    result = walk_forward_predict(
        features,
        target,
        ends,
        model_name="ridge",
        param_grid=[{"alpha": 0.1}, {"alpha": 1.0}],
        min_train_years=3,
        embargo_days=5,
    )
    assert len(result.predictions) > 0
    assert not result.predictions.index.has_duplicates
    assert set(result.fold_report["test_year"]) == set(
        result.predictions.index.get_level_values("date").year.unique()
    )
