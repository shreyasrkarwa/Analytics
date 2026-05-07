# Architecting a Unified Revenue Platform

*A four-layer architecture for unifying CRM ingestion, target cascading, ML forecasting, and role-based analytics into a single platform*

**Shreyas Karwa**

*Photo by [Carlos Muza](https://unsplash.com/@kmuza) on [Unsplash](https://unsplash.com/photos/hpjSkU2UYSU) — search: "data analytics dashboard" or "person looking at analytics on screen"*

---

## 1. The Fragmentation Problem

In enterprise B2B sales organizations, analytics infrastructure tends to grow organically. One dashboard monitors current-quarter bookings. Another tracks open pipeline and coverage ratios. A third shows individual contributor (IC) attainment against quota. A Confluence page, updated weekly (sometimes manually, sometimes with AI assistance), stitches together a narrative that ties the pieces together.

Each of these tools works fine in isolation. The dashboards are accurate. The data is trustworthy. The problem is not quality — it is **fragmentation**. A frontline sales manager who needs a complete picture of their team's health must navigate to three or four different systems, mentally reconcile the information, and synthesize a story. An executive preparing for a QBR does the same thing, but across ten teams. The data team, meanwhile, spends a disproportionate share of sprint capacity fielding one-off requests that exist only because no single system provides the full view.

This is a **time sink**, not a trust problem. The individual pieces are fine. The cost is the human effort required to assemble them into something actionable — effort that scales linearly with organizational complexity and repeats every single week.

The solution is not another dashboard. It is a fundamental rearchitecting of the data layer, the logic layer, and the presentation layer into a single coherent system — what we call a **Unified Revenue Platform (URP)**. This article walks through the full technical architecture of a URP, covering every layer from raw CRM ingestion through ML-driven forecasting to front-end design principles that can drive high organic adoption.

---

## 2. System Architecture

The URP operates across four distinct planes, each with a single responsibility. Data flows upward from source systems through transformation and intelligence layers to the presentation layer:

```
┌────────────────────────┐   ┌──────────────────────────────────┐
│      CRM System        │   │   Financial Planning System      │
│  (e.g., Salesforce)    │   │   (e.g., Anaplan)                │
│   What happened        │   │   What should happen             │
│   (raw CRM events)     │   │   (board-approved targets)       │
└────────┬───────────────┘   └──────────────┬───────────────────┘
         │ Raw events                       │ Macro targets
         │                                  │
┌────────▼──────────────────────────────────▼────────────────────┐
│                       LOGIC LAYER                              │
│    Data Platform (e.g., Databricks / Delta Lake / PySpark)     │
│                                                                │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│   │  Dimension   │  │    Target    │  │   ML Forecasting     │ │
│   │ Engineering  │  │  Cascading   │  │  + Bias Correction   │ │
│   │ (Silver)     │  │  (DAG)       │  │  (Ensemble)          │ │
│   └─────────────┘  └──────────────┘  └──────────────────────┘ │
│                                                                │
└────────────────────────────┬───────────────────────────────────┘
                             │ Reads aggregated metrics
                             │
┌────────────────────────────▼───────────────────────────────────┐
│                      PRESENTATION LAYER                        │
│          BI Tool / Internal Web App — Role-Based Views          │
│          Design principle: never computes, only reads           │
└────────────────────────────────────────────────────────────────┘
```

**Separation of concerns:** The CRM system owns *what happened* (raw events — opportunities, accounts, stage changes). The financial planning system owns *what should happen* (board-approved revenue targets, headcount plans). The data platform owns *what it means* (transformed dimensions, cascaded quotas, ML forecasts). The presentation layer never computes — it only reads pre-aggregated metrics.

This separation is load-bearing. When a VP asks "why did our EMEA number change?", the answer traces cleanly to a single authoritative layer. There is no ambiguity about which system computed the number.

Throughout this article, we'll use Salesforce as the CRM, Databricks (with Delta Lake and PySpark) as the data platform, and Anaplan as the planning system for concrete code examples. The architecture itself is system-agnostic — any CRM (HubSpot, Dynamics 365), any data platform (Snowflake, BigQuery), and any planning tool (Adaptive Insights, Pigment) can fill these roles as long as the layer boundaries are respected.

---

## 3. Layer 1 — Data Ingestion

The foundation is a near-real-time pipeline from the CRM system into a Bronze layer on the data platform. The Bronze layer is deliberately raw — no transformations, no business logic. Its job is fidelity to the source system.

As a concrete example, here is how you can extract opportunities from Salesforce using the [simple-salesforce](https://github.com/simple-salesforce/simple-salesforce) Python library and land them in Delta Lake.

### 3.1 Extracting Opportunities with Full Org Lineage

A useful engineering pattern: pulling the full manager hierarchy directly inside the CRM query. In Salesforce, this means using SOQL relationship traversal (`Owner.Manager.Manager.Name`) to stamp every opportunity with its complete organizational lineage at extraction time — avoiding expensive post-hoc joins downstream.

```python
from simple_salesforce import Salesforce
import pandas as pd
from datetime import datetime, timedelta

def extract_opportunities(sf: Salesforce, lookback_days: int = 90) -> pd.DataFrame:
    """
    Pull all open and recently-closed opportunities from Salesforce.
    The manager hierarchy is flattened into rep_name, mgr1_name, mgr2_name
    via SOQL relationship traversal — a single API call.
    """
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime('%Y-%m-%dT%H:%M:%SZ')

    soql = f"""
        SELECT Id, Name, AccountId, Account.Name, OwnerId,
               Owner.Name, Owner.Manager.Name, Owner.Manager.Manager.Name,
               Amount, ACV__c, StageName, CloseDate,
               ForecastCategoryName, Probability, Type, LeadSource,
               CreatedDate, LastModifiedDate, IsClosed, IsWon,
               Segment__c, Region__c, Sub_Region__c
        FROM Opportunity
        WHERE (IsClosed = false OR CloseDate >= {cutoff})
          AND IsDeleted = false
        ORDER BY LastModifiedDate DESC
    """

    records = sf.query_all(soql)['records']
    df = pd.DataFrame(records).drop(columns=['attributes'], errors='ignore')

    # Flatten nested Owner.Manager hierarchy
    df['rep_name']  = df['Owner'].apply(lambda x: x.get('Name') if isinstance(x, dict) else None)
    df['mgr1_name'] = df['Owner'].apply(
        lambda x: x.get('Manager', {}).get('Name') if isinstance(x, dict) else None
    )
    df['mgr2_name'] = df['Owner'].apply(
        lambda x: x.get('Manager', {}).get('Manager', {}).get('Name')
        if isinstance(x, dict) and isinstance(x.get('Manager'), dict) else None
    )
    df = df.drop(columns=['Owner'], errors='ignore')

    df['CloseDate'] = pd.to_datetime(df['CloseDate'])
    df['Amount']    = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
    df['ACV__c']    = pd.to_numeric(df['ACV__c'], errors='coerce').fillna(0)

    return df
```

### 3.2 Bronze Layer: Merge-Upsert

Extracted records land in a Bronze table via merge-upsert. New records are inserted; existing records are updated using the CRM record Id as the natural merge key. In a Databricks environment, this can use Delta Lake's `MERGE INTO` semantics:

```python
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

def upsert_to_bronze(df: pd.DataFrame, delta_path: str):
    """Merge-upsert CRM records into a Bronze Delta table."""
    spark = SparkSession.builder.appName("URP-Ingestion").getOrCreate()
    spark_df = spark.createDataFrame(df)

    if DeltaTable.isDeltaTable(spark, delta_path):
        bronze = DeltaTable.forPath(spark, delta_path)
        bronze.alias("target").merge(
            spark_df.alias("source"), "target.Id = source.Id"
        ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
    else:
        spark_df.write.format("delta").save(delta_path)

    print(f"Upserted {len(df):,} records to {delta_path}")
```

---

## 4. Layer 2 — Dimension Engineering (The Silver Layer)

Raw CRM dimensions are generic — they know nothing about your company's fiscal calendar, go-to-market motion, or organizational structure. The Silver pipeline transforms raw data into analytically meaningful dimensions.

### 4.1 Fiscal Quarter Tagging

Enterprise sales runs on fiscal calendars that rarely align with calendar years. A company with an August 1 fiscal year start needs every date mapped to its fiscal quarter:

```python
def get_fiscal_quarter(close_date, fy_start_month: int = 8) -> str:
    """
    Map a calendar date to a fiscal quarter string.
    Example: September 2024 → 'FY25Q1' (with August FY start).
    """
    if close_date is None:
        return None
    month, year = close_date.month, close_date.year
    fiscal_month = (month - fy_start_month) % 12 + 1
    fiscal_quarter = (fiscal_month - 1) // 3 + 1
    fiscal_year = year if month >= fy_start_month else year - 1
    return f"FY{str(fiscal_year + 1)[-2:]}Q{fiscal_quarter}"

# Tag the full DataFrame
opps_df['fiscal_quarter'] = opps_df['CloseDate'].apply(get_fiscal_quarter)
opps_df['is_current_quarter'] = opps_df['fiscal_quarter'] == get_fiscal_quarter(datetime.utcnow())
```

### 4.2 Pipeline Coverage Ratio

Pipeline coverage answers the most operationally critical question in enterprise sales: *"For every dollar of quota, how many dollars of open pipeline does this rep carry?"*

```python
import numpy as np

def compute_pipeline_coverage(opps_df, quotas_df):
    """
    coverage_ratio = open_pipeline_acv / quota_acv

    Healthy (>= 3.0x): enough pipeline to absorb ~33% win rate
    Moderate (1.5-3.0x): some risk of missing target
    At Risk (< 1.5x): insufficient pipeline
    """
    open_pipeline = (
        opps_df[(opps_df['IsClosed'] == False) & opps_df['is_current_quarter']]
        .groupby('OwnerId')
        .agg(open_pipeline_acv=('ACV__c', 'sum'))
        .reset_index()
    )

    coverage = open_pipeline.merge(quotas_df, on='OwnerId', how='left')
    coverage['coverage_ratio'] = np.where(
        coverage['quota_acv'] > 0,
        (coverage['open_pipeline_acv'] / coverage['quota_acv']).round(2),
        np.nan
    )
    coverage['coverage_status'] = np.select(
        [coverage['coverage_ratio'] >= 3.0, coverage['coverage_ratio'] >= 1.5],
        ['Healthy', 'Moderate'],
        default='At Risk'
    )
    return coverage
```

The 3x healthy threshold is not arbitrary — it reflects the empirical observation that qualified B2B enterprise opportunities close at roughly 25–35%, so 3x pipeline is needed to absorb deal slippage and still hit quota.

### 4.3 Segmented Win Rates

Win rates vary dramatically by market segment and region. Enterprise deals might close at 20%, while SMB closes at 45%. Computing win rates at the Segment × Region level is essential for realistic pipeline-to-quota planning:

```python
def compute_win_rate_by_segment(opps_df, trailing_quarters: int = 4):
    """Compute rolling win rate and ASP by Segment x Region."""
    closed = opps_df[opps_df['IsClosed'] == True]
    recent_qs = closed['fiscal_quarter'].drop_duplicates().sort_values(ascending=False).head(trailing_quarters)
    trailing = closed[closed['fiscal_quarter'].isin(recent_qs)]

    return (
        trailing.groupby(['Segment__c', 'Region__c'])
        .agg(
            total_deals=('Id', 'count'),
            won_deals=('IsWon', 'sum'),
            avg_deal_size=('ACV__c', lambda x: x[trailing.loc[x.index, 'IsWon']].mean()),
        )
        .assign(win_rate=lambda d: (d['won_deals'] / d['total_deals']).round(3))
        .reset_index()
    )
```

---

## 5. Layer 3 — The Intelligence Layer

This is where the platform earns its value. The intelligence layer houses three engines that transform raw metrics into decision-ready outputs:

1. **Target Cascading** — deterministically distributes a macro revenue target (e.g., $500M) through the organizational hierarchy using a DAG-based algorithm with configurable hedge multipliers.

2. **Commit Reconciliation** — detects and corrects systematic manager forecast bias (sandbagging vs. over-commitment) using a Bias Index framework.

3. **ML Revenue Forecast** — an ensemble model that blends pipeline mechanics, historical patterns, and behavioral correction signals.

Each of these engines is a substantial technical contribution. Rather than duplicating the full treatment here, I'll describe how they integrate within the platform and point to the detailed articles:

**Target cascading** models the org hierarchy as a directed acyclic graph (DAG) and traverses top-down, injecting a configurable hedge multiplier at each level. A $500M target cascaded through 5 levels with 5% hedging produces $552.56M in aggregate IC quotas — an intentional structural buffer that protects the top-line number. The full algorithm, including proofs and edge cases, is covered in my companion article: [Hierarchical Sales Target Cascading Using DAGs in Python](https://medium.com/towards-artificial-intelligence/hierarchical-sales-target-cascading-using-directed-acyclic-graphs-dags-in-python-1426c7980b87).

**Commit reconciliation** uses a per-manager Bias Index (β) computed from trailing quarters of actual-vs-committed revenue. A sandbagger (β > 1.10) consistently delivers above their commit; a "happy ears" manager (β < 0.90) consistently misses. The adjusted forecast blends ML baseline with bias-corrected commits: `F_adj = (1−w) × F_ml + w × (commit × β)`. This framework is also detailed in the [companion article](https://medium.com/towards-artificial-intelligence/hierarchical-sales-target-cascading-using-directed-acyclic-graphs-dags-in-python-1426c7980b87).

**The ML revenue forecast** is where the integration pays off. The model uses a 7-feature vector that draws from all three signal types:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# The 7-feature pipeline encodes three signal types:
FEATURES = [
    'open_pipeline_acv',       # Mechanical: total open pipeline
    'pipeline_coverage_ratio', # Mechanical: pipeline / quota
    'avg_deal_age_days',       # Historical: mean opportunity age
    'pct_stage4_plus',         # Mechanical: fraction in late stages
    'avg_win_rate_trailing',   # Historical: trailing 4Q win rate
    'manager_bias_index',      # Behavioral: from Bias Index framework
    'headcount_quota_ratio',   # Organizational: capacity utilization
]

def train_revenue_forecast(features_df, target_col='actual_closed_acv'):
    """
    Train a Gradient Boosting model with time-series cross-validation.
    The key design choice: manager_bias_index is a first-class feature,
    not a post-hoc adjustment.
    """
    X = features_df[FEATURES].fillna(0).values
    y = features_df[target_col].values

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

    # Time-series CV to prevent future data leakage
    tscv = TimeSeriesSplit(n_splits=5)
    mape_scores = []
    for train_idx, val_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        mask = y[val_idx] != 0
        if mask.sum() > 0:
            mape_scores.append(mean_absolute_percentage_error(y[val_idx][mask], preds[mask]))

    model.fit(X, y)  # Final fit on all data
    cv_mape = np.mean(mape_scores)

    # Feature importance
    importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print(f"CV MAPE: {cv_mape:.2%}")
    for feat, imp in importance:
        print(f"  {feat:30s} {imp:.3f}")

    return model
```

The critical insight: including `manager_bias_index` as a first-class ML feature — rather than applying it as a post-hoc correction — allows the model to learn interaction effects. A sandbagger with strong late-stage pipeline is qualitatively different from a neutral manager with the same pipeline. In practice, behavioral features like the Bias Index tend to rank among the most important predictors — often contributing more to accuracy than raw pipeline volume features. This aligns with the broader finding in the judgmental forecasting literature that systematic human biases, when properly quantified, carry significant predictive signal (Lawrence et al., 2006).

---

## 6. Layer 4 — Presentation: Designing for Adoption

A technically superior platform dies without adoption. The presentation layer was designed around one non-negotiable principle: **build for the IC's compensation plan, and every other user follows.**

### 6.1 Role-Based Drill-Down

The platform exposes a unified view hierarchy where every level is visible to every user:

```
Company-Level View
  └── Segment View (Enterprise / Mid-Market / SMB)
        └── Region View (AMER / EMEA / APAC)
              └── Manager Team View
                    └── IC Individual View
                          └── Deal-Level View (individual opportunities)
```

An IC can see company-level pacing. A CRO can drill into a single rep's pipeline. This radical transparency was a deliberate design choice. When ICs understand how their individual number fits into the company story, quota attainment becomes self-reinforcing.

### 6.2 The Four Core Modules

The presentation layer is organized into four modules that mirror the revenue lifecycle:

**Pipeline Generation** tracks new pipeline created in the current quarter, broken down by source (outbound, inbound, partner, AE-sourced). The key metric is pipeline creation velocity — is the org building enough coverage to hit *next* quarter's targets?

**ACV Attainment** shows real-time closed ACV versus quota across every organizational dimension. This is the compensation mirror — every IC can see exactly where they stand, without a spreadsheet.

**Forecasting** exposes the blended ML + bias-corrected forecast with confidence intervals. Managers submit weekly commits through this module, triggering real-time re-scoring. All submissions are timestamped and auditable.

**Coverage Mapping** visualizes pipeline coverage ratios (Healthy / Moderate / At Risk) against cascaded quotas for every team.

### 6.3 Structured Forecast Submission

A key goal of the platform is to eliminate the need for manual forecasting spreadsheets by providing a structured weekly submission workflow. Each submission can be versioned, attributed, and made immediately visible to the full management chain:

```python
from datetime import datetime

def submit_weekly_forecast(
    manager_id: str, forecast_period: str,
    commit_amount: float, best_case_amount: float,
    submitted_by: str, notes: str = ""
) -> dict:
    """
    Record a manager's weekly forecast submission.
    In production, this writes to a Gold Delta table and triggers
    async ML re-scoring for the submitting manager's team.
    """
    submission = {
        'manager_id':      manager_id,
        'forecast_period':  forecast_period,
        'commit_amount':    commit_amount,
        'best_case_amount': best_case_amount,
        'submitted_by':     submitted_by,
        'submitted_at':     datetime.utcnow().isoformat(),
        'notes':            notes,
    }

    # In production: persist to Delta, trigger re-scoring, notify chain
    # spark.createDataFrame([submission]).write.format("delta")
    #     .mode("append").save("dbfs:/mnt/urp/gold/weekly_forecasts")

    return submission
```

A function like this can consolidate what might otherwise be dozens of individual manager spreadsheets into a single auditable workflow — with every submission versioned and immediately visible to leadership.

---

## 7. What to Expect: Measurable Outcomes

The value of a unified platform shows up in a few measurable areas. Based on industry benchmarks and the operational patterns described above, here are the types of improvements this architecture is designed to deliver:

**Forecast accuracy.** Manual enterprise sales forecasting processes typically produce MAPE in the 20–25% range at quarter-start (Mentzer & Moon, 2005). An ML ensemble that incorporates pipeline mechanics, historical win rates, and behavioral bias correction — as described in Section 5 — can bring this significantly lower. The key lever is the Bias Index: incorporating it as a first-class feature rather than a post-hoc adjustment captures interaction effects that neither signal provides alone.

**Operational efficiency.** When every metric lives in a single platform, ad-hoc data requests drop sharply. Instead of analysts pulling one-off reports from multiple systems, managers and executives can self-serve through the drill-down hierarchy. The data team's sprint capacity shifts from answering questions to building capabilities.

**Spreadsheet elimination.** A structured forecast submission workflow (Section 6.3) can consolidate the many individual manager spreadsheets that accumulate in a typical sales org. Each submission becomes versioned, auditable, and immediately visible — removing the weekly collection cycle entirely.

**Organic adoption.** The compensation-plan alignment principle (Section 8) is the most reliable driver of platform usage. When ICs can see their real-time pacing against the exact KPIs that determine their paycheck, they tend to adopt the platform without mandates or training campaigns — the system earns usage by being better at the thing spreadsheets were doing.

---

## 8. Three Design Principles

Building this platform surfaced three principles that generalize beyond revenue analytics to any enterprise data platform:

**Centralize process, not just data.** A dashboard that shows data is a viewer. A platform that absorbs the actual business workflow — weekly forecast submissions, target cascading, commit reconciliation — is infrastructure. The difference is whether people use it *for* their job or *in addition to* their job. Every spreadsheet that still exists alongside your platform is a design failure.

**Layer your architecture for speed and logic.** Real-time CRM connections provide recency. A dedicated data platform provides analytical depth. Financial planning systems provide the authoritative constraint. Each layer has a single job. The most common architectural mistake is letting the visualization layer own business logic — the moment your BI tool computes a metric, you've created a system where the number changes depending on which dashboard you open.

**Build for comp-plan visibility.** Any enterprise data platform seeking adoption should identify the metric that most directly affects individual compensation and make it the most prominent, most accurate, most real-time number in the system. Training sessions, executive mandates, and Slack reminders tend to produce temporary compliance. Compensation-linked visibility tends to produce sustained organic adoption — because the platform becomes the fastest way for ICs to answer the question they care about most.

---

## 9. Code and Resources

The complete implementation — ingestion pipelines, dimension engineering, ML forecasting, and a demo pipeline with synthetic data — is available on GitHub:

**Repository:** [github.com/shreyasrkarwa/Analytics/unified_revenue_platform](https://github.com/shreyasrkarwa/Analytics/tree/main/unified_revenue_platform)

All code uses standard open-source libraries: `pandas`, `numpy`, `networkx`, `scikit-learn`, and optionally `pyspark` + `delta-spark` for the Spark components. Clone the repo and run `python demo_pipeline.py` for a full end-to-end walkthrough with synthetic data.

**Companion articles:**
- [Hierarchical Sales Target Cascading Using DAGs in Python](https://medium.com/towards-artificial-intelligence/hierarchical-sales-target-cascading-using-directed-acyclic-graphs-dags-in-python-1426c7980b87) — the full treatment of the quota cascading algorithm and bias reconciliation framework referenced in Section 5.
- Karwa, S. (2026). "Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises." *SSRN Working Paper*.

---

## References

1. Armbrust, M., et al. (2020). Delta Lake: High-performance ACID table storage over cloud object stores. *Proceedings of the VLDB Endowment*, 13(12), 3411–3424.
2. Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*, 114(526), 804–819.
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
4. Lawrence, M., Goodwin, P., O'Connor, M., & Onkal, D. (2006). Judgmental forecasting: A review of progress over the last 25 years. *International Journal of Forecasting*, 22(3), 493–518.
5. Davenport, T. H., & Harris, J. G. (2007). *Competing on Analytics: The New Science of Winning.* Harvard Business School Press.

---

*Shreyas Karwa is a Senior Analytics Engineer specializing in enterprise revenue intelligence systems. He is the author of the open-source [b2b-revenue-forecasting](https://pypi.org/project/b2b-revenue-forecasting/) and [b2b-territory-optimization](https://pypi.org/project/b2b-territory-optimization/) packages on PyPI. Connect on [GitHub](https://github.com/shreyasrkarwa) or [LinkedIn](https://www.linkedin.com/in/shreyaskarwa/).*
