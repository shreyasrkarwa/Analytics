# Unified Revenue Platform (URP)

A production-grade framework for building unified revenue intelligence systems in B2B enterprise environments. Integrates CRM ingestion, hierarchical target cascading, human-bias reconciliation, and ML-driven revenue forecasting into a single coherent platform.

## Architecture

The architecture is system-agnostic. Our reference implementation uses Salesforce, Databricks, and Anaplan, but any CRM, data platform, and planning tool can fill these roles.

```
┌────────────────────────┐   ┌──────────────────────────────────┐
│      CRM System        │   │   Financial Planning System      │
│   What happened        │   │   What should happen             │
│   (raw CRM events)     │   │   (board-approved targets)       │
└────────┬───────────────┘   └──────────────┬───────────────────┘
         │                                  │
┌────────▼──────────────────────────────────▼────────────────────┐
│                       LOGIC LAYER                              │
│    Data Platform (e.g., Databricks / Delta Lake / PySpark)     │
│    • Dimension engineering        • Target cascading (DAG)     │
│    • Commit reconciliation        • ML forecast ensemble       │
└────────────────────────────┬───────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                      PRESENTATION LAYER                        │
│          BI Tool / Internal Web App — Role-Based Views          │
└────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `ingestion` | CRM extraction (e.g., Salesforce SOQL) with lakehouse merge-upsert |
| `dimension_engineering` | Fiscal calendar tagging, pipeline coverage, segmented win rates |
| `quota_cascader` | DAG-based hierarchical target allocation with hedge multipliers |
| `bias_reconciler` | Manager commit bias detection (Sandbagger / Happy Ears / Neutral) |
| `revenue_forecaster` | Gradient Boosting ensemble with time-series CV and optional MLflow |
| `weekly_forecast` | Structured forecast submission workflow replacing manual spreadsheets |
| `data_generator` | Synthetic B2B data for testing and demonstration |

## Requirements

```
pandas >= 1.5.0
numpy >= 1.23.0
networkx >= 2.8.0
scikit-learn >= 1.1.0
```

Optional: `pyspark`, `delta-spark`, `mlflow`, `simple-salesforce`

## Quick Start

```python
from unified_revenue_platform.data_generator import (
    generate_org_hierarchy, generate_opportunities, generate_historical_commits
)
from unified_revenue_platform.quota_cascader import QuotaCascader
from unified_revenue_platform.bias_reconciler import BiasReconciler

# Generate synthetic org hierarchy and data
dag, metadata = generate_org_hierarchy(n_ics=50)
opps_df = generate_opportunities(dag, metadata)
commits_df = generate_historical_commits(dag, metadata)

# Cascade quotas through the hierarchy
historical = opps_df[opps_df["IsWon"]].rename(columns={"rep_name": "node_id", "ACV__c": "acv_closed"})
cascader = QuotaCascader(dag=dag, historical_attainment=historical)
quotas = cascader.cascade(root="CRO", macro_target=500_000_000, hedge_multiplier=1.05)

# Detect manager bias
reconciler = BiasReconciler(commits_df)
print(reconciler.get_org_summary())
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Demo Pipeline

```bash
python demo_pipeline.py
```

Runs the full 7-stage pipeline with synthetic data: hierarchy generation, opportunity creation, dimension engineering, quota cascading, bias detection, ML forecasting, and weekly forecast submission.

## Production Benchmarks

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Forecast MAPE | ~23% | ~7.3% | **-68%** |
| Ad-hoc data request time | 2.1 days | 0.2 days | **-90%** |
| Manager spreadsheets | 43 | 0 | **-100%** |
| IC weekly active usage | — | 94% | Baseline |

## Related Publications

- Karwa, S. (2026). "Architecting a Unified Revenue Platform." *Towards AI*.
- Karwa, S. (2026). "Unified Revenue Intelligence Platforms: Architecture, Algorithms, and Empirical Evaluation in Enterprise B2B Environments." *Expert Systems with Applications* (submitted).
- Karwa, S. (2026). [Hierarchical Sales Target Cascading Using DAGs in Python](https://medium.com/towards-artificial-intelligence/hierarchical-sales-target-cascading-using-directed-acyclic-graphs-dags-in-python-1426c7980b87). *Towards AI*.
- Karwa, S. (2026). "Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises." *SSRN Working Paper*.

## License

MIT License
