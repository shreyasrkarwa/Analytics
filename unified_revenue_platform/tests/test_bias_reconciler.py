"""Tests for the Bias Reconciler module."""

import pytest
import pandas as pd
import numpy as np

from unified_revenue_platform.bias_reconciler import (
    BiasReconciler,
    reconcile_forecast,
    ReconciliationResult,
)


@pytest.fixture
def sandbagger_data():
    """Historical data for a known sandbagger (actual > commit)."""
    return pd.DataFrame({
        "manager_id": ["mgr_alice"] * 4,
        "fiscal_quarter": ["FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4"],
        "manual_commit": [300_000, 350_000, 400_000, 380_000],
        "actual_closed": [390_000, 420_000, 500_000, 475_000],
    })


@pytest.fixture
def happy_ears_data():
    """Historical data for a known happy ears (actual < commit)."""
    return pd.DataFrame({
        "manager_id": ["mgr_bob"] * 4,
        "fiscal_quarter": ["FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4"],
        "manual_commit": [500_000, 480_000, 520_000, 510_000],
        "actual_closed": [350_000, 370_000, 400_000, 380_000],
    })


@pytest.fixture
def mixed_data(sandbagger_data, happy_ears_data):
    """Combined data with multiple manager archetypes."""
    neutral = pd.DataFrame({
        "manager_id": ["mgr_carol"] * 4,
        "fiscal_quarter": ["FY24Q1", "FY24Q2", "FY24Q3", "FY24Q4"],
        "manual_commit": [400_000, 410_000, 390_000, 405_000],
        "actual_closed": [405_000, 400_000, 395_000, 410_000],
    })
    return pd.concat([sandbagger_data, happy_ears_data, neutral], ignore_index=True)


class TestBiasReconciler:
    """Test suite for BiasReconciler."""

    def test_detects_sandbagger(self, sandbagger_data):
        """Sandbagger should have bias_index > 1.10."""
        reconciler = BiasReconciler(sandbagger_data)
        bias, archetype, _ = reconciler.get_bias("mgr_alice")
        assert bias > 1.10
        assert archetype == "Sandbagger"

    def test_detects_happy_ears(self, happy_ears_data):
        """Happy Ears should have bias_index < 0.90."""
        reconciler = BiasReconciler(happy_ears_data)
        bias, archetype, _ = reconciler.get_bias("mgr_bob")
        assert bias < 0.90
        assert archetype == "Happy Ears"

    def test_detects_neutral(self, mixed_data):
        """Neutral manager should have 0.90 <= bias_index <= 1.10."""
        reconciler = BiasReconciler(mixed_data)
        bias, archetype, _ = reconciler.get_bias("mgr_carol")
        assert 0.90 <= bias <= 1.10
        assert archetype == "Neutral"

    def test_unknown_manager_returns_default(self, mixed_data):
        """Unknown manager should return bias=1.0, archetype=Unknown."""
        reconciler = BiasReconciler(mixed_data)
        bias, archetype, _ = reconciler.get_bias("mgr_nonexistent")
        assert bias == 1.0
        assert archetype == "Unknown"

    def test_org_summary_counts(self, mixed_data):
        """Org summary should correctly count each archetype."""
        reconciler = BiasReconciler(mixed_data)
        summary = reconciler.get_org_summary()
        assert summary["total_managers"] == 3
        assert summary["sandbaggers"] == 1
        assert summary["happy_ears"] == 1
        assert summary["neutral"] == 1

    def test_custom_thresholds(self, sandbagger_data):
        """Custom thresholds should change classification."""
        # With a very high sandbagger threshold, alice becomes Neutral
        reconciler = BiasReconciler(
            sandbagger_data,
            sandbagger_threshold=2.0,
            happy_ears_threshold=0.50,
        )
        _, archetype, _ = reconciler.get_bias("mgr_alice")
        assert archetype == "Neutral"


class TestReconcileForecast:
    """Test suite for the reconcile_forecast function."""

    def test_sandbagger_increases_forecast(self, sandbagger_data):
        """Bias correction should increase forecast for a sandbagger."""
        reconciler = BiasReconciler(sandbagger_data)
        result = reconcile_forecast(
            manager_id="mgr_alice",
            manual_commit=400_000,
            ml_baseline=490_000,
            bias_reconciler=reconciler,
        )
        # Adjusted forecast should exceed the raw commit
        assert result.adjusted_forecast > result.manual_commit
        assert result.hidden_upside > 0
        assert result.archetype == "Sandbagger"

    def test_happy_ears_decreases_forecast(self, happy_ears_data):
        """Bias correction should decrease forecast for happy ears."""
        reconciler = BiasReconciler(happy_ears_data)
        result = reconcile_forecast(
            manager_id="mgr_bob",
            manual_commit=500_000,
            ml_baseline=400_000,
            bias_reconciler=reconciler,
        )
        # Bias-corrected commit should be less than raw commit
        assert result.bias_corrected_commit < result.manual_commit
        assert result.hidden_upside < 0

    def test_bias_weight_effect(self, sandbagger_data):
        """Higher bias_weight should give more weight to human signal."""
        reconciler = BiasReconciler(sandbagger_data)
        result_low = reconcile_forecast(
            "mgr_alice", 400_000, 490_000, reconciler, bias_weight=0.3
        )
        result_high = reconcile_forecast(
            "mgr_alice", 400_000, 490_000, reconciler, bias_weight=0.8
        )
        # Higher weight on bias-corrected (sandbagger) commit -> higher forecast
        assert result_high.adjusted_forecast > result_low.adjusted_forecast

    def test_confidence_interval_contains_forecast(self, sandbagger_data):
        """Confidence interval should contain the point forecast."""
        reconciler = BiasReconciler(sandbagger_data)
        result = reconcile_forecast(
            "mgr_alice", 400_000, 490_000, reconciler
        )
        ci_low, ci_high = result.confidence_interval
        assert ci_low <= result.adjusted_forecast <= ci_high

    def test_to_dict_serialization(self, sandbagger_data):
        """to_dict should return a serializable dictionary."""
        reconciler = BiasReconciler(sandbagger_data)
        result = reconcile_forecast(
            "mgr_alice", 400_000, 490_000, reconciler
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "manager_id" in d
        assert "adjusted_forecast" in d
        assert isinstance(d["confidence_interval"], tuple)
