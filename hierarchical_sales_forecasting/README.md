# B2B Revenue Forecasting (`b2b_revenue_forecasting`)

[![PyPI version](https://badge.fury.io/py/b2b-revenue-forecasting.svg)](https://badge.fury.io/py/b2b-revenue-forecasting)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source Python framework designed mathematically for **Enterprise RevOps and Data Strategy** teams. 

Unlike traditional bottom-up time-series libraries (which are strictly built for B2C retail/inventory forecasting and rely on mathematical averages), this package is explicitly architected to handle the realities of B2B enterprise sales: **Hierarchical Quotas, Managerial Cascading, and "Sandbagging" Biases.**

## 🚀 Features

* **`QuotaCascader`**: Distribute massive macro-targets down complex top-down org charts using rolling 4-quarter capacity models. Automatically enforce **"Managerial Hedges"** (e.g. VPs overassigning by 5%) to create safe mathematical buffers recursively down the DAG.
* **`CommitReconciler`**: Reconcile mathematical pipeline probabilities with human logic. Automatically calculate an individual manager's historical `Bias Quotient` to detect sandbagging or "happy ears", and auto-adjust their current commits back to reality.
* **Flexible Organization DAGs**: Ingest standard flattened dataframes mapping out jagged corporate reporting lines automatically using `networkx`.

## 📦 Installation

```bash
pip install b2b-revenue-forecasting
```

## 💻 Quickstart

### 1. The Quota Cascader (Target Setting)
Mathematically cascade an Enterprise target downwards, while enforcing a 5% safety hedge at every level based on 4 rolling quarters of node-capacity.

```python
import pandas as pd
from b2b_revenue_forecasting.hierarchy import SalesHierarchy
from b2b_revenue_forecasting.quota_cascader import QuotaCascader

df = pd.read_csv('your_crm_data.csv')

# Build the Org Hierarchy (Dynamically scales to 5, 8, or 10 nodes deep)
hierarchy = SalesHierarchy()
hierarchy.from_dataframe(
    df, 
    path_cols=['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC'], 
    metrics_cols=['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment']
)

# Cascade $100M with a 5% managerial hedge to create commit safety
cascader = QuotaCascader(hierarchy)
quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0, hedge_multiplier=1.05)
```

### 2. The Commit Reconciler (Removing Human Bias)
Use historical CRM data to fix bad managerial forecasting in real-time.

```python
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler

reconciler = CommitReconciler(historical_dataframe)

# Manager A historically closes 1.5x what they commit (A "Sandbagger")
# If they commit $100k today, our algorithm automatically corrects it to $150k.
adjusted_forecast = reconciler.reconcile_forecast(
    manager_id='Manager_A', 
    current_commit=100_000, 
    machine_forecast=120_000
)
```

## 🤝 Contributing
Built explicitly for RevOps analysts, Data Scientists, and VP Revenue Operations executing scaling go-to-market strategies. Contributions, issues, and pull requests are warmly welcomed!
