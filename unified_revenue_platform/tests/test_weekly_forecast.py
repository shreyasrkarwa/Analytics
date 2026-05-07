"""Tests for the Weekly Forecast module."""

import pytest
from unified_revenue_platform.weekly_forecast import (
    WeeklyForecastManager,
    ForecastSubmission,
)


@pytest.fixture
def forecast_mgr():
    return WeeklyForecastManager()


class TestWeeklyForecastManager:

    def test_submit_forecast(self, forecast_mgr):
        """Basic submission should succeed."""
        sub = forecast_mgr.submit_forecast(
            manager_id="mgr_1",
            forecast_period="FY25Q2",
            commit_amount=400_000,
            best_case_amount=500_000,
            submitted_by="mgr_1",
        )
        assert isinstance(sub, ForecastSubmission)
        assert sub.commit_amount == 400_000
        assert sub.week_over_week_change is None  # First submission

    def test_commit_exceeds_best_case_raises(self, forecast_mgr):
        """Commit > best case should raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            forecast_mgr.submit_forecast(
                manager_id="mgr_1",
                forecast_period="FY25Q2",
                commit_amount=600_000,
                best_case_amount=500_000,
                submitted_by="mgr_1",
            )

    def test_week_over_week_change(self, forecast_mgr):
        """Second submission should compute WoW change."""
        forecast_mgr.submit_forecast(
            "mgr_1", "FY25Q2", 400_000, 500_000, "mgr_1"
        )
        sub2 = forecast_mgr.submit_forecast(
            "mgr_1", "FY25Q2", 440_000, 550_000, "mgr_1"
        )
        assert sub2.previous_commit == 400_000
        assert sub2.week_over_week_change == pytest.approx(0.10, abs=0.001)

    def test_forecast_summary(self, forecast_mgr):
        """Summary should include one row per manager plus total."""
        forecast_mgr.submit_forecast("mgr_1", "FY25Q2", 400_000, 500_000, "mgr_1")
        forecast_mgr.submit_forecast("mgr_2", "FY25Q2", 300_000, 400_000, "mgr_2")

        summary = forecast_mgr.get_forecast_summary("FY25Q2")
        assert len(summary) == 3  # 2 managers + TOTAL
        total_row = summary[summary["manager_id"] == "TOTAL"]
        assert total_row["commit_amount"].values[0] == 700_000

    def test_detect_forecast_drift(self, forecast_mgr):
        """Should detect managers with significant forecast changes."""
        # Manager who revises upward significantly
        forecast_mgr.submit_forecast("mgr_1", "FY25Q2", 300_000, 400_000, "mgr_1")
        forecast_mgr.submit_forecast("mgr_1", "FY25Q2", 400_000, 500_000, "mgr_1")

        drifters = forecast_mgr.detect_forecast_drift("FY25Q2", drift_threshold=0.10)
        assert len(drifters) == 1
        assert drifters[0]["drift_direction"] == "upward"
        assert abs(drifters[0]["total_drift"] - 0.3333) < 0.01

    def test_submission_history(self, forecast_mgr):
        """Should track full submission history per manager."""
        for i in range(3):
            forecast_mgr.submit_forecast(
                "mgr_1", "FY25Q2", 300_000 + i * 10_000,
                400_000 + i * 10_000, "mgr_1"
            )
        history = forecast_mgr.get_submission_history("mgr_1", "FY25Q2")
        assert len(history) == 3
