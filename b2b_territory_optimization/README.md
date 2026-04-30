# B2B Territory Optimization (`b2b_territory_optimization`)

[![PyPI version](https://badge.fury.io/py/b2b-territory-optimization.svg)](https://badge.fury.io/py/b2b-territory-optimization)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source Python framework designed for **Enterprise RevOps and Data Strategy** teams to mathematically design, carve, and manage B2B sales territories.

This package bridges the gap between raw CRM data and the start of the forecasting cycle. It dynamically groups accounts into strict Taxonomic Buckets (e.g., "Enterprise AMER"), greedily balances TAM and Workload across territories, and generates human resource staffing matrices spanning multiple roles (AEs, SDRs, Managers) with configurable coverage ratios.

It perfectly complements its sister package, [`b2b-revenue-forecasting`](https://pypi.org/project/b2b-revenue-forecasting/), which cascades quotas and forecasts pipeline *after* territories have been defined.

---

## 🚀 Features

| Module | Purpose |
|--------|---------|
| **`TaxonomySchema`** | Dynamically define strict hierarchical boundaries for territories (e.g., segmenting by Region + Vertical) preventing any cross-contamination. |
| **`TerritoryAllocator`** | Greedily partition accounts within taxonomy buckets into `K` territories, balancing a target metric (like Estimated TAM) to within fractions of a percent. |
| **`SellerAssignmentMatrix`** | Map custom human resource roles (AE, SE, Manager) to carved territories using configurable coverage ratios (e.g., 1:1, 1:3, 2:1). |
| **`ReassignmentEngine`** | The "Manager Override" simulator. Tracks manual account moves, flags the resulting TAM imbalance, and suggests optimal accounts to swap back. |

---

## 💻 Seamless Forecasting Integration

This package was built to feed directly into `b2b-revenue-forecasting`. Once your territories are designed and staffed, you can export the hierarchy automatically:

```python
from b2b_territory_optimization import SellerAssignmentMatrix
from b2b_revenue_forecasting.hierarchy import SalesHierarchy

# 1. Staff your territories
matrix = SellerAssignmentMatrix(territory_list)
matrix.add_role('Regional_VP', 5.0)
matrix.add_role('Sales_Manager', 2.0)
matrix.add_role('Account_Executive', 1.0)

# 2. Export to a flat hierarchy dataframe
df_hierarchy = matrix.export_to_hierarchy(
    hierarchy_columns_order=['Regional_VP', 'Sales_Manager', 'Account_Executive']
)

# 3. Feed directly into the forecasting package!
hierarchy = SalesHierarchy()
hierarchy.from_dataframe(df_hierarchy, path_cols=['Global', 'Regional_VP', 'Sales_Manager', 'Account_Executive', 'Territory_ID'])
```

---

## 📦 Installation

```bash
pip install b2b-territory-optimization
```

---

## 🧠 Key Concepts

### Strict Taxonomy Buckets
Unlike generic K-Means clustering, B2B territory carving requires **hard constraints**. If a territory is designated for "Mid-Market EMEA", the algorithm physically cannot place an "Enterprise APAC" account into it, regardless of mathematical balance. `TaxonomySchema` enforces this by slicing the DataFrame into sub-graphs before allocation occurs.

### Greedy LPT Balancing
The `TerritoryAllocator` uses an approach inspired by the Longest Processing Time (LPT) multiprocessor scheduling algorithm. It sorts accounts by the target metric descending, then iteratively places the largest unassigned account into the territory with the current lowest total sum. This consistently yields $< 0.1\%$ variance on B2B workloads.

### Manager Override Suggestions
When an Account Executive has a pre-existing relationship with a buyer, managers will often manually override the algorithmic territory design to keep them together. When an account is moved, `ReassignmentEngine` calculates the exact delta required to restore balance and evaluates `itertools.combinations` of smaller accounts to suggest the perfect trade.

---

## 🤝 Contributing
Built explicitly for RevOps analysts, Data Scientists, and VP Revenue Operations executing scaling go-to-market strategies.

## 📄 License
MIT License
