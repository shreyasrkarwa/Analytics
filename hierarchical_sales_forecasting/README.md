# B2B Revenue Forecasting (`b2b_revenue_forecasting`)

[![PyPI version](https://badge.fury.io/py/b2b-revenue-forecasting.svg)](https://badge.fury.io/py/b2b-revenue-forecasting)
[![Tests](https://github.com/shreyasrkarwa/Analytics/actions/workflows/test.yml/badge.svg)](https://github.com/shreyasrkarwa/Analytics/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An open-source Python framework designed mathematically for **Enterprise RevOps and Data Strategy** teams. 

Unlike traditional bottom-up time-series libraries (which are strictly built for B2C retail/inventory forecasting and rely on mathematical averages), this package is explicitly architected to handle the realities of B2B enterprise sales: **Hierarchical Quotas, Managerial Cascading, Pipeline Health Analysis, and "Sandbagging" Biases.**

---

## 🚀 Features

| Module | Purpose |
|--------|---------|
| **`SalesHierarchy`** | Build flexible org charts as DAGs from flat CRM data — supports 3-level startups to 10-level enterprises |
| **`QuotaCascader`** | Distribute macro-targets top-down using rolling N-quarter capacity models with configurable managerial hedges |
| **`CommitReconciler`** | Detect sandbagging and "happy ears" bias via historical Bias Quotients, then auto-correct forecasts |
| **`PipelineAdjuster`** | Diagnose pipeline health with per-region thresholds and redistribute IC quotas using zero-sum logic |

### What's New in v0.2.0

- **`PipelineAdjuster`**: Post-cascade pipeline health analyzer with `diagnose()` and `adjust()` modes
- **Flexible quarter support**: `QuotaCascader` now auto-discovers any number of `_Attainment` columns (4, 8, 12 quarters)
- **New IC handling**: Partial-history imputation and equal-share allocation for brand-new hires
- **CRO overrides**: Lock specific IC quotas via `new_ic_overrides` to bypass the algorithm
- **Per-node hedging**: Apply different hedge multipliers to different regions/managers
- **GitHub Actions CI/CD**: Automated testing on Python 3.9–3.12

---

## 📦 Installation

```bash
pip install b2b-revenue-forecasting
```

---

## 💻 Quickstart

### 1. Build the Org Hierarchy

```python
import pandas as pd
from b2b_revenue_forecasting.hierarchy import SalesHierarchy

# ⚠️ Use keep_default_na=False if your data has 'NA' as a region name
df = pd.read_csv('your_crm_data.csv', keep_default_na=False)

# Works with any depth: 3 levels or 10 levels
hierarchy = SalesHierarchy()
hierarchy.from_dataframe(
    df, 
    path_cols=['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC'], 
    metrics_cols=['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment',
                  'Current_Pipeline']
)

print(f"Nodes: {len(hierarchy.graph.nodes)}")
print(f"ICs:   {len(hierarchy.get_leaves('Global_Corp'))}")
```

### 2. Cascade Quotas Top-Down

```python
from b2b_revenue_forecasting.quota_cascader import QuotaCascader

cascader = QuotaCascader(hierarchy)

# Basic: distribute $100M evenly by historical capacity
quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0)

# With 5% hedge at every management level (compounds: 1.05^5 ≈ 27.6% overassignment)
quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0, hedge_multiplier=1.05)

# Per-node hedge: NA gets aggressive 10%, others standard 5%
quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0, hedge_multiplier={
    'Global_Corp': 1.05, 'NA': 1.10, 'EMEA': 1.05, 'APAC': 1.05
})

# CRO override: strategic hire gets exactly $500K regardless of history
quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0,
    hedge_multiplier=1.05,
    new_ic_overrides={'IC_Strategic_Hire': 500_000.0}
)
```

### 3. Detect & Fix Forecasting Bias

```python
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler

historical = pd.DataFrame({
    'Manager_ID':              ['Mgr_A', 'Mgr_A', 'Mgr_B', 'Mgr_B'],
    'Historical_Commit':       [200_000,  250_000, 300_000,  350_000],
    'Historical_Actual_Closed': [300_000,  375_000, 270_000,  280_000],
})

reconciler = CommitReconciler(historical)

# Mgr_A is a sandbagger (bias = 1.5x) — commit inflated automatically
adjusted = reconciler.reconcile_forecast('Mgr_A', current_commit=100_000)
# → $150,000

# Blend with ML baseline (50/50 average)
blended = reconciler.reconcile_forecast('Mgr_A', 100_000, machine_forecast=120_000)
# → $135,000
```

### 4. Pipeline Health Diagnosis & Redistribution

```python
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster

adjuster = PipelineAdjuster(hierarchy, quotas, pipeline_attr='Current_Pipeline')

# Configure per-region coverage thresholds (ICs inherit from ancestors)
thresholds = {
    'NA':       {'healthy': 1.5, 'at_risk': 0.8},
    'EMEA':     {'healthy': 2.5, 'at_risk': 1.2},
    'APAC':     {'healthy': 3.0, 'at_risk': 1.5},
    '_default': {'healthy': 2.0, 'at_risk': 1.0}
}

# Diagnose — returns a DataFrame with risk status for every node
diagnosis = adjuster.diagnose(thresholds)
print(diagnosis.groupby('Risk_Status')['Node'].count())

# Flag-only mode — returns original quotas unchanged (for pre-approval review)
flagged = adjuster.adjust(mode='flag_only', coverage_thresholds=thresholds)

# Redistribute mode — zero-sum IC adjustment within each manager's team
adjusted = adjuster.adjust(
    mode='redistribute',
    coverage_thresholds=thresholds,
    max_adjustment_pct=0.20,                          # ±20% cap per IC
    locked_nodes={'IC_Protected': 500_000.0}           # CRO-locked ICs excluded
)
# ✅ Manager totals preserved | ✅ Donors give, receivers get | ✅ 20% cap enforced
```

---

## 🧠 Key Concepts

### Managerial Hedge (Overassignment Buffer)
A multiplier applied at each management level to create mathematical safety. A 5% hedge across 5 layers compounds to ~27.6% total overassignment (`1.05⁵`), ensuring the enterprise hits its number even if some ICs miss.

### Bias Quotient
```
Bias Quotient = Σ(Actual Closed) / Σ(Committed)
```
- **> 1.0** = Sandbagger (closes more than committed → inflate their forecast)
- **= 1.0** = Neutral
- **< 1.0** = Happy Ears (over-promises → deflate their forecast)

### Pipeline Coverage Ratio
```
Coverage = Current Pipeline / Cascaded Quota
```
| Coverage | Status | Action |
|----------|--------|--------|
| ≥ healthy threshold | 🟢 Healthy | May donate quota |
| ≥ at_risk threshold | 🟡 Moderate | No action |
| ≥ 1.0 | 🟠 At Risk | May receive quota |
| < 1.0 | 🔴 Critical | Urgent — pipeline below target |

### New IC Handling
| Scenario | Behavior |
|----------|----------|
| Full history | Proportional by total capacity |
| Partial history (e.g., 1 of 4 quarters) | Zero quarters imputed with own non-zero average |
| Brand new (all zeros) | Equal share of team target |
| CRO override | Fixed amount, excluded from pool |

---

## 🧪 Testing

```bash
# Run all tests
cd hierarchical_sales_forecasting
pip install -e .
python -m pytest tests/ -v

# Run the full demo
python demo_full_pipeline.py
```

---

## 📄 Publications

This framework is the subject of peer-reviewed research and technical publications:

| Publication | Venue | Status |
|-------------|-------|--------|
| [Hierarchical Sales Target Cascading using DAGs in Python](https://medium.com/towards-artificial-intelligence/hierarchical-sales-target-cascading-using-directed-acyclic-graphs-dags-in-python-1426c7980b87) | **Towards AI** | ✅ Published |
| Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises | **SSRN** (Preprint) | ⏳ Under Review |
| Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises | **Journal of Revenue and Pricing Management** (Springer) | ⏳ Under Review |

If you use this package in your research, please cite:

```
Karwa, S. (2026). Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation
in B2B Enterprises: A Methodological Framework. SSRN Working Paper.
```

---

## 📋 Requirements

- Python ≥ 3.8
- pandas ≥ 1.0.0
- networkx ≥ 2.5
- numpy ≥ 1.19.0

---

## 🤝 Contributing

Built explicitly for RevOps analysts, Data Scientists, and VP Revenue Operations executing scaling go-to-market strategies. Contributions, issues, and pull requests are warmly welcomed!

- **Report bugs**: [GitHub Issues](https://github.com/shreyasrkarwa/Analytics/issues)
- **Source code**: [GitHub](https://github.com/shreyasrkarwa/Analytics/tree/master/hierarchical_sales_forecasting)

---

## 📄 License

MIT License — see [LICENSE](https://opensource.org/licenses/MIT) for details.
