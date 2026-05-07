"""Tests for the Dimension Engineering module."""

import pytest
import pandas as pd
from datetime import datetime

from unified_revenue_platform.dimension_engineering import (
    get_fiscal_quarter,
    tag_fiscal_quarters,
    compute_pipeline_coverage,
)


class TestFiscalQuarter:
    """Test suite for fiscal quarter computation."""

    def test_fy_start_august_q1(self):
        """August-October maps to Q1 with FY start in August."""
        assert get_fiscal_quarter(datetime(2024, 8, 1)) == "FY25Q1"
        assert get_fiscal_quarter(datetime(2024, 9, 15)) == "FY25Q1"
        assert get_fiscal_quarter(datetime(2024, 10, 31)) == "FY25Q1"

    def test_fy_start_august_q2(self):
        """November-January maps to Q2."""
        assert get_fiscal_quarter(datetime(2024, 11, 1)) == "FY25Q2"
        assert get_fiscal_quarter(datetime(2025, 1, 31)) == "FY25Q2"

    def test_fy_start_august_q3(self):
        """February-April maps to Q3."""
        assert get_fiscal_quarter(datetime(2025, 2, 1)) == "FY25Q3"
        assert get_fiscal_quarter(datetime(2025, 4, 30)) == "FY25Q3"

    def test_fy_start_august_q4(self):
        """May-July maps to Q4."""
        assert get_fiscal_quarter(datetime(2025, 5, 1)) == "FY25Q4"
        assert get_fiscal_quarter(datetime(2025, 7, 31)) == "FY25Q4"

    def test_fy_start_january(self):
        """With FY starting January, calendar and fiscal align."""
        assert get_fiscal_quarter(datetime(2025, 1, 15), fy_start_month=1) == "FY25Q1"
        assert get_fiscal_quarter(datetime(2025, 4, 15), fy_start_month=1) == "FY25Q2"
        assert get_fiscal_quarter(datetime(2025, 7, 15), fy_start_month=1) == "FY25Q3"
        assert get_fiscal_quarter(datetime(2025, 10, 15), fy_start_month=1) == "FY25Q4"

    def test_none_input(self):
        """None date should return None."""
        assert get_fiscal_quarter(None) is None

    def test_tag_fiscal_quarters_adds_columns(self):
        """tag_fiscal_quarters should add fiscal_quarter and is_current_quarter."""
        df = pd.DataFrame({
            "CloseDate": [datetime(2025, 1, 15), datetime(2025, 5, 20)],
        })
        result = tag_fiscal_quarters(df)
        assert "fiscal_quarter" in result.columns
        assert "is_current_quarter" in result.columns
        assert len(result) == 2


class TestPipelineCoverage:
    """Test suite for pipeline coverage computation."""

    def test_healthy_coverage(self):
        """3x+ coverage should be classified as Healthy."""
        opps = pd.DataFrame({
            "OwnerId": ["rep1", "rep1", "rep1"],
            "ACV__c": [100_000, 200_000, 150_000],
            "IsClosed": [False, False, False],
            "is_current_quarter": [True, True, True],
        })
        quotas = pd.DataFrame({
            "OwnerId": ["rep1"],
            "quota_acv": [100_000],
        })
        result = compute_pipeline_coverage(opps, quotas)
        assert result.iloc[0]["coverage_status"] == "Healthy"
        assert result.iloc[0]["coverage_ratio"] == 4.5

    def test_at_risk_coverage(self):
        """<1.5x coverage should be classified as At Risk."""
        opps = pd.DataFrame({
            "OwnerId": ["rep1"],
            "ACV__c": [50_000],
            "IsClosed": [False],
            "is_current_quarter": [True],
        })
        quotas = pd.DataFrame({
            "OwnerId": ["rep1"],
            "quota_acv": [100_000],
        })
        result = compute_pipeline_coverage(opps, quotas)
        assert result.iloc[0]["coverage_status"] == "At Risk"

    def test_closed_opps_excluded(self):
        """Closed opportunities should not count toward pipeline."""
        opps = pd.DataFrame({
            "OwnerId": ["rep1", "rep1"],
            "ACV__c": [500_000, 200_000],
            "IsClosed": [True, False],
            "is_current_quarter": [True, True],
        })
        quotas = pd.DataFrame({
            "OwnerId": ["rep1"],
            "quota_acv": [300_000],
        })
        result = compute_pipeline_coverage(opps, quotas)
        # Only the open $200K should count
        assert result.iloc[0]["open_pipeline_acv"] == 200_000
