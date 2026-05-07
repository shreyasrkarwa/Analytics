"""Tests for the Revenue Forecaster module."""

import pytest
import pandas as pd
import numpy as np

from unified_revenue_platform.revenue_forecaster import (
    RevenueForecaster,
    DEFAULT_FEATURES,
    ForecastResult,
)


@pytest.fixture
def training_data():
    """Generate synthetic training data for the forecaster."""
    rng = np.random.default_rng(42)
    n = 30  # Need enough for 5-fold CV

    data = {
        "open_pipeline_acv": rng.uniform(500_000, 5_000_000, n),
        "pipeline_coverage_ratio": rng.uniform(1.0, 5.0, n),
        "avg_deal_age_days": rng.uniform(20, 120, n),
        "pct_stage4_plus": rng.uniform(0.1, 0.6, n),
        "avg_win_rate_trailing": rng.uniform(0.15, 0.45, n),
        "manager_bias_index": rng.uniform(0.8, 1.3, n),
        "headcount_quota_ratio": rng.uniform(0.6, 1.2, n),
    }

    # Target: correlated with features (realistic)
    data["actual_closed_acv"] = (
        data["open_pipeline_acv"] * data["avg_win_rate_trailing"]
        * data["manager_bias_index"]
        + rng.normal(0, 50_000, n)
    )

    return pd.DataFrame(data)


class TestRevenueForecaster:
    """Test suite for RevenueForecaster."""

    def test_train_returns_forecast_result(self, training_data):
        """Training should return a ForecastResult with valid metrics."""
        forecaster = RevenueForecaster(n_estimators=50)
        result = forecaster.train(training_data, n_cv_splits=3)

        assert isinstance(result, ForecastResult)
        assert result.cv_mape >= 0
        assert result.cv_mae >= 0
        assert result.n_training_samples == len(training_data)

    def test_predict_after_training(self, training_data):
        """Predict should work after training."""
        forecaster = RevenueForecaster(n_estimators=50)
        forecaster.train(training_data, n_cv_splits=3)

        preds = forecaster.predict(training_data.head(5))
        assert len(preds) == 5
        assert all(p > 0 for p in preds)  # Revenue should be positive

    def test_predict_before_training_raises(self, training_data):
        """Predict before training should raise ValueError."""
        forecaster = RevenueForecaster()
        with pytest.raises(ValueError, match="not fitted"):
            forecaster.predict(training_data)

    def test_feature_importance_ranking(self, training_data):
        """Feature importance should rank all features."""
        forecaster = RevenueForecaster(n_estimators=50)
        forecaster.train(training_data, n_cv_splits=3)

        importance = forecaster.get_feature_importance()
        assert len(importance) == len(DEFAULT_FEATURES)
        assert importance["importance"].sum() > 0
        assert list(importance["rank"]) == list(range(1, len(DEFAULT_FEATURES) + 1))

    def test_custom_features(self, training_data):
        """Should work with a custom feature subset."""
        features = ["open_pipeline_acv", "pipeline_coverage_ratio"]
        training_data["actual_closed_acv"] = training_data["open_pipeline_acv"] * 0.3

        forecaster = RevenueForecaster(
            feature_cols=features,
            n_estimators=50,
        )
        result = forecaster.train(training_data, n_cv_splits=3)
        assert result.n_training_samples == len(training_data)

    def test_result_summary_format(self, training_data):
        """ForecastResult.summary() should return a formatted string."""
        forecaster = RevenueForecaster(n_estimators=50)
        result = forecaster.train(training_data, n_cv_splits=3)

        summary = result.summary()
        assert "CV MAPE" in summary
        assert "Top features" in summary
