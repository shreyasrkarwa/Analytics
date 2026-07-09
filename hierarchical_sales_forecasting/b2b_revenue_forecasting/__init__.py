"""
Core modules for Hierarchical B2B Forecasting.
"""
from b2b_revenue_forecasting.hierarchy import SalesHierarchy, HierarchyValidationError
from b2b_revenue_forecasting.quota_cascader import (
    QuotaCascader,
    GateAllocationError,
    HedgeByDepth,
)
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster
from b2b_revenue_forecasting.metric_spec import MetricSpec
from b2b_revenue_forecasting.batch import cascade_many, route_targets
from b2b_revenue_forecasting.pins import Pin, apply_pins

__version__ = "0.16.0"

__all__ = [
    "SalesHierarchy",
    "HierarchyValidationError",
    "QuotaCascader",
    "GateAllocationError",
    "HedgeByDepth",
    "CommitReconciler",
    "PipelineAdjuster",
    "MetricSpec",
    "cascade_many",
    "route_targets",
    "Pin",
    "apply_pins",
    "__version__",
]
