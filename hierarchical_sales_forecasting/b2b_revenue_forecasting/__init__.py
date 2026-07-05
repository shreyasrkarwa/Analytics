"""
Core modules for Hierarchical B2B Forecasting.
"""
from b2b_revenue_forecasting.hierarchy import SalesHierarchy, HierarchyValidationError
from b2b_revenue_forecasting.quota_cascader import QuotaCascader, GateAllocationError
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster
from b2b_revenue_forecasting.metric_spec import MetricSpec
from b2b_revenue_forecasting.batch import cascade_many

__version__ = "0.7.0"

__all__ = [
    "SalesHierarchy",
    "HierarchyValidationError",
    "QuotaCascader",
    "GateAllocationError",
    "CommitReconciler",
    "PipelineAdjuster",
    "MetricSpec",
    "cascade_many",
    "__version__",
]
