"""
Unified Revenue Platform (URP)
==============================

A production-grade framework for building unified revenue intelligence
systems in B2B enterprise environments. Integrates CRM ingestion,
hierarchical target cascading, human-bias reconciliation, and
ML-driven revenue forecasting into a single coherent platform.

Architecture:
    Layer 1 - Ingestion:    Salesforce CRM -> Delta Lake Bronze
    Layer 2 - Engineering:  Bronze -> Silver (fiscal dims, coverage ratios)
    Layer 3 - Intelligence: Quota cascading, bias reconciliation, ML forecast
    Layer 4 - Presentation: Role-based views with comp-plan visibility

Modules:
    ingestion              - Salesforce extraction and Delta Lake upsert
    dimension_engineering   - Fiscal calendar, pipeline coverage, win rates
    quota_cascader         - DAG-based hierarchical target allocation
    bias_reconciler        - Manager commit bias detection and correction
    revenue_forecaster     - Ensemble ML model with MLflow tracking
    weekly_forecast        - Structured forecast submission workflow

Author: Shreyas Karwa
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Shreyas Karwa"

from .quota_cascader import QuotaCascader
from .bias_reconciler import BiasReconciler, reconcile_forecast
from .revenue_forecaster import RevenueForecaster
from .dimension_engineering import (
    get_fiscal_quarter,
    compute_pipeline_coverage,
    compute_win_rate_by_segment,
)

__all__ = [
    "QuotaCascader",
    "BiasReconciler",
    "reconcile_forecast",
    "RevenueForecaster",
    "get_fiscal_quarter",
    "compute_pipeline_coverage",
    "compute_win_rate_by_segment",
]
