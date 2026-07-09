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
| **`MetricSpec`** | Declare which historical metrics (NetNewACV, CloudSeats, DC seats, LTM expansion, …) drive cascading, in what direction (proportional or inverse), and at what weight — with auto-suggested weights from correlation analysis |
| **`CommitReconciler`** | Detect sandbagging and "happy ears" bias via historical Bias Quotients, then auto-correct forecasts |
| **`PipelineAdjuster`** | Diagnose pipeline health with per-region thresholds and redistribute IC quotas using zero-sum logic |

### What's New in v0.13.0 — pins that behave: any level, a basis, and safe re-hedging ([#28](https://github.com/shreyasrkarwa/Analytics/issues/28) · [#21](https://github.com/shreyasrkarwa/Analytics/issues/21) · [#23](https://github.com/shreyasrkarwa/Analytics/issues/23))

`new_ic_overrides` pins now work at **any level** — pin a manager and the subtree total is fixed and cascades within; jagged-hierarchy leaf pins are honored too (both were silently ignored). Conservation is guaranteed: unpinned siblings share the exact remainder, the brand-new carve-out is capped at the pool, and a pin *exceeding* the pool floors siblings at $0 (never negative) with a loud warning plus `overpinned_amount`/`overpinned_nodes` in `gating_report()`. Pins also gained a **basis** (`override_basis`): `"base"` (default — pin the un-hedged plan number, hedged derived) or `"cascaded"` (pin the exact final number, base derived) — previously pinned reps silently received no hedge. And for post-cascade edits: do the math on `base_quota`, roll parents up on base, then derive the hedged layer with `cascader.rehedge(edited_base)` (per-node ratios via `cascader.hedge_ratios()`) — summing hedged leaves into parents double-counts buffers up the tree.

### What's New in v0.12.0 — small slices split proportionally, not equally ([issue #33](https://github.com/shreyasrkarwa/Analytics/issues/33))

Correlation-based weight suggestion is undefined on tiny slices (n ≤ 2) and zero-variance columns — the *common* case in per-group batch runs. Previously those candidates were zeroed, and an all-zero slate made the cascade equal-split: siblings with a 6× seat difference got identical quotas, silently. Now `on_degenerate="proportional"` (the new default) keeps each degenerate candidate's **declared** weight, so allocation stays proportional to the blended metric values — with any number of metrics, directions included. The old behavior is one keyword away (`on_degenerate="equal"`), `"raise"` fails fast, degenerate candidates are flagged in the report (`degenerate` / `fallback` fields) and named in a warning, and missing columns still get weight 0 (absent data ≠ thin data).

### What's New in v0.11.0 — per-depth hedging with `HedgeByDepth` ([issue #13](https://github.com/shreyasrkarwa/Analytics/issues/13))

Hedge policies are usually stated by *level*, not by node: "front-line managers carry 10%, their directors 5%." `HedgeByDepth` expresses exactly that and works everywhere `hedge_multiplier` does — including `cascade_many`, where per-node dicts were structurally impossible:

```python
from b2b_revenue_forecasting import cascade_many, HedgeByDepth

quotas_long, _ = cascade_many(
    hierarchy_df, target_df, group_keys=[...], target_col=..., taxonomy=[...],
    metrics=[...],
    hedge_multiplier=HedgeByDepth(
        from_leaves={1: 1.10, 2: 1.05},   # deepest mgr 10%, next level 5%
        default=1.0,
    ),
)
```

`from_leaves` counts distance to the farthest descendant IC (correct in jagged hierarchies); `from_root` uses `node_depths()`-style depth; both can combine (multiplying). The spec resolves against each hierarchy at cascade time, so `base_quota` reconciliation and all audit columns behave exactly as with a hand-built dict.

### What's New in v0.10.2 — degenerate slices warn before equal-splitting ([issue #8](https://github.com/shreyasrkarwa/Analytics/issues/8))

`suggest_weights` has long degraded gracefully on thin data (single row, zero variance, all-null columns → weight 0 + rationale, never an exception, batch runs shielded by `cascade_many`'s skip mode). Now it also *tells you*: when every candidate comes back with weight 0, one `UserWarning` explains that the slice carries no usable correlation signal and that cascading will fall back to an equal split among siblings — so the fallback is never a silent surprise. Missing `target_column` still raises (typos should be loud).

### What's New in v0.10.1 — weight-normalization semantics at point of use ([issue #11](https://github.com/shreyasrkarwa/Analytics/issues/11))

Docs release. The "raw weight ≠ influence" nuance is now explained where you actually set weights — on `MetricSpec.weight`, in the `cascade_quota(metrics=)` docstring, and in a new "How Weights Become Influence" section below, all with the same worked example (`[1.0, 0.5, 0.0]` → `[66.7%, 33.3%, 0%]`; a raw `0.067` alongside `[1.0, 0.98, 0.4]` is **2.7%** of the influence, not 6.7%). The documented examples are pinned by a unit test so they can't drift from the implementation.

### What's New in v0.10.0 — one-call gating report ([issue #10](https://github.com/shreyasrkarwa/Analytics/issues/10))

`cascader.gating_report()` consolidates the whole gating story of the last cascade into one dict: which nodes were gated (`gated_node_ids`, with `gated_leaf_ids` split out), which were funded anyway as a last resort (`gate_relaxed_node_ids`), how much target is explicitly unallocated (`unallocated_amount` / `unallocated_nodes`), and the per-cascade reconciliation numbers — `leaf_quota_sum` (hedged), `leaf_base_sum` (un-hedged), `base_gap`, and a single `reconciles` boolean asserting every input dollar is either on an IC or reported as unallocated. No more manual diagnostics comparing root target to leaf sums.

### What's New in v0.9.0 — gate semantics you can configure ([issue #9](https://github.com/shreyasrkarwa/Analytics/issues/9))

Gates are no longer hardwired to "gated iff `value <= threshold`". `MetricSpec.gate_mode` picks the PASS predicate: `"gt"` (default — unchanged behavior), `"ge"` ("at least N seats": `MetricSpec('Seats', columns=['Seats'], gate_threshold=5, gate_mode='ge')`), `"lt"`/`"le"` (gate territories with *too much* of a signal, e.g. `gate_threshold=100, gate_mode='le'` to exclude churn-heavy reps), and `"truthy"` (boolean entitlement flags, threshold ignored). The exact predicate is documented on `MetricSpec`; all modes compose with AND across multiple gates and inherit the `gate_fallback` no-stranding guarantees.

### What's New in v0.8.0 — analysis-ready outputs ([issue #7](https://github.com/shreyasrkarwa/Analytics/issues/7))

Exports now carry your source attributes — no more manual merges. Declare descriptive columns once (`from_dataframe(metadata_cols=['Rep_Name', 'Segment', 'Geo'])`; they're stored raw and never treated as signal), then emit them with `quotas_to_dataframe(metadata_cols=[...])`. Or skip storage entirely and left-join any frame onto the leaf rows: `quotas_to_dataframe(source_df=df, source_join_col='node_5_rep_no')` — the join is keyed on **original** ids via the new `hierarchy.id_map`, so it survives collision renames, and an `original_id` column appears automatically whenever a node was renamed. `cascade_many` accepts `metadata_cols=` too.

### What's New in v0.7.2 — one graph accessor, plus hierarchy helpers ([issue #5](https://github.com/shreyasrkarwa/Analytics/issues/5))

`.graph` is now the canonical name for the underlying `nx.DiGraph` on every class (`SalesHierarchy`, `QuotaCascader`, `PipelineAdjuster`), with `.hierarchy` kept as a working alias on both `SalesHierarchy` and `QuotaCascader` — no more `AttributeError` whichever one you reach for. New read-only helpers mean you rarely need the raw graph at all: `hierarchy.roots()`, `hierarchy.leaves(root=None)`, `hierarchy.managers(root=None)`, and `hierarchy.node_depths()` (handy for building per-level hedge dicts).

### What's New in v0.7.1 — `MetricSpec` columns resolve intuitively ([issue #6](https://github.com/shreyasrkarwa/Analytics/issues/6))

Specs returned by `suggest_weights` are now directly usable — no more `for s in suggested: s.columns = [s.name]`. Column resolution order: explicit `columns=` always wins → the `Q1_<name>…Q<lookback>_<name>` convention → **new**: the plain attribute named exactly `<name>` (so a spec called `knowledge_workers` finds your `knowledge_workers` column automatically). And if an active metric ends up with zero signal across the whole tree, `cascade_quota` warns and names the columns it tried — silent no-op metrics are gone. The full `name`/`columns` contract is documented on the `MetricSpec` dataclass.

### What's New in v0.7.0 — batch cascading with `cascade_many` ([issue #4](https://github.com/shreyasrkarwa/Analytics/issues/4))

Real planning cascades many targets across many segments — every `(sales_type, product, regional)` combination, for every quarter. `cascade_many` replaces the hand-rolled loop with one call: it prepares each combination once (filter → validated hierarchy → weights) and cascades every matching target row against it, returning tidy long frames tagged with your group keys.

```python
from b2b_revenue_forecasting import cascade_many, MetricSpec

quotas_long, weights_long = cascade_many(
    hierarchy_df,                     # taxonomy + metric columns, 1 row per rep
    target_df,                        # group keys + fiscal_quarter + target
    group_keys=["st1_sales_type", "base_product_r4f", "regional"],
    target_col="nn_acv_target",
    taxonomy=["regional", "node_3_region", "node_4_team", "node_5_rep_no"],
    metrics=[MetricSpec("knowledge_workers", direction="proportional",
                        weight=1.0, columns=["knowledge_workers"])],
    gate_metrics=[MetricSpec("dc_seats", columns=["dc_seats"])],
    hedge_multiplier=1.05,
)
# quotas_long: group keys + fiscal_quarter + node_id/depth/level +
#              cascaded_quota + base_quota + gate audit columns
quotas_long.to_csv("all_cascades.csv", index=False)
```

Extra `target_df` columns (like `fiscal_quarter`) act as sub-targets that reuse the prepared combination. Weights can be fixed, suggested once globally, or re-suggested per combination (`suggest_config=` + `weights_mode="per_group"`). Failing combinations warn and are skipped by default (`on_error="raise"` to fail fast). Every slice gets the full correctness stack: value coercion, duplicate-level healing, DAG validation, never-gated roots, and a base layer that reconciles at every depth.

### What's New in v0.6.1 — non-numeric metrics can't silently zero a slice ([issue #3](https://github.com/shreyasrkarwa/Analytics/issues/3))

A gate column holding `numpy.bool_` scalars or `"true"`/`"false"` strings used to aggregate to 0 for every leaf — gating entire slices to $0 with no traceback. Now every metric value is **coerced on ingest** (numpy scalars unboxed, boolean strings → bools, `"1,200"` / `"$500"` / `"12.5%"` → numbers), uncoercible cells **warn and are treated as missing**, and the cascader itself warns once per column if it ever meets a value it can't read. No API changes; the `MAX(CASE WHEN flag THEN 1 ELSE 0 END)` SQL workaround is no longer needed.

### What's New in v0.6.0 — dirty hierarchies can't crash the cascade ([issue #1](https://github.com/shreyasrkarwa/Analytics/issues/1))

Previously, a row with the same value at two adjacent levels (e.g., team `T1` AND rep `T1`) silently built a self-loop, and `cascade_quota` crashed with a cryptic `RecursionError` deep inside networkx. v0.6.0 makes malformed hierarchies either self-heal or fail loudly with an actionable message.

- **`on_collision` parameter on `from_dataframe`** — `"suffix"` (default, renames the deeper duplicate to `<value>__<level_column>` and warns), `"skip"` (drops the duplicate level, jagged-style), or `"error"` (raise naming the row).
- **Blank-string hygiene** — empty cells and literal `"nan"`/`"none"`/`"null"` strings (a `keep_default_na=False` hazard) are treated as missing levels instead of becoming a shared `"nan"` node. `'NA'` the region is still data.
- **`hierarchy.validate()`** + automatic DAG validation at the end of `from_dataframe` — cross-row cycles raise `HierarchyValidationError` naming the cycle path.
- **Fail-fast cascades** — `cascade_quota` checks the graph up front, and the recursive aggregators carry recursion-stack guards, so a cyclic graph can never RecursionError again. Diamond-shaped DAGs remain supported.

```python
h = SalesHierarchy()
h.from_dataframe(df, path_cols=taxonomy, metrics_cols=cols,
                 on_collision='suffix')   # default — shown for clarity
# -> UserWarning: 1 duplicate-level value(s) detected and renamed ...
h.validate()                              # explicit re-check, chainable
```

### What's New in v0.5.0 — no more stranded targets ([issue #12](https://github.com/shreyasrkarwa/Analytics/issues/12))

Previously, when a gate zeroed an **entire subtree** (e.g., a Migration cascade where no rep in the whole slice had DC entitlement), the target for that slice was silently dropped — depth-0 held the target while depth 1+ summed short. v0.5.0 guarantees **the base (un-hedged) quota sums to the macro target at every depth**.

- **`gate_fallback` parameter on `cascade_quota`** controls what happens when every child of a funded node is gated:
  - `"redistribute"` *(default)* — a fully-gated subtree's share flows to its nearest non-gated siblings (gates still roll up as before); if the *entire* level — even the whole tree — is gated, the gate is relaxed at that level as a last resort so the target still reaches ICs. No silent target loss, ever.
  - `"strand_at_root"` — children stay $0; the undistributable amount stays on the deepest non-gated ancestor and is reported via `cascader.unallocated` / `cascader.unallocated_nodes` plus an `is_unallocated` column in `quotas_to_dataframe`.
  - `"error"` — raises `GateAllocationError` so the caller decides.
- **The root is never gated to $0.** It always carries the macro target in every mode.
- **`cascader.base_quotas`** — every `cascade_quota` call now also computes the un-hedged cascade in the same pass, so `hedged_quota = base_quota × hedge^depth` decomposes without a second run. Pass `unhedged_quotas="auto"` to `quotas_to_dataframe` to get the audit columns for free.
- **`cascader.reconciliation_report(quotas, target=..., strict=True)`** — per-depth reconciliation DataFrame (`depth, n_nodes, total_quota, target, delta, reconciles`); `strict=True` raises listing every non-reconciling depth. Run it on `cascader.base_quotas` (hedged quotas legitimately grow with depth).
- **`gate_relaxed` column in `quotas_to_dataframe`** flags nodes that received quota despite being gated because every sibling was also gated — so the last-resort fallback is always visible in the CSV.

```python
quotas = cascader.cascade_quota(
    'Enterprise_AMER', 1_000_000.0,
    hedge_multiplier=1.05,
    metrics=forward_metrics,
    gate_metrics=[MetricSpec('DC_Seats', columns=['DC_Seats'])],
    gate_fallback='redistribute',   # default — shown for clarity
)
# Base layer reconciles at EVERY depth, even with fully-gated teams:
cascader.reconciliation_report(cascader.base_quotas,
                               target=1_000_000.0, strict=True)
df = cascader.quotas_to_dataframe(quotas, unhedged_quotas='auto')
```

### What's New in v0.4.0

- **Gate metrics — hard kill-switches.** `cascade_quota(..., gate_metrics=[...])` excludes any node whose rolled-up gate value is at or below a threshold from the cascade entirely (quota = 0), redistributing its share among non-gated siblings. Designed for white-space planning: e.g., gating "migration NetNewACV" on `Unmigrated_Seats` zeros out territories with nothing left to migrate. Gates propagate upward naturally — a manager whose whole team fails the gate gets $0 too. Composes with AND across multiple gates. CRO overrides win over gates.
- **Two planning philosophies, both supported.** See the section below.
- **`is_gated` column in `quotas_to_dataframe`** when gates were used, so analysts can distinguish "$0 because gated" from "$0 because no signal."
- **`cascader.gated_nodes`** — the set of gated nodes from the most recent cascade, stored for inspection.

### Two Planning Philosophies

The package supports two philosophically distinct ways of building a quota plan. Both use the same primitives — pick the one that matches how your org thinks about fairness.

**Earned planning** — *"who has proven they can sell this?"*

Cascade on **historical** signals (past NetNewACV attainment, past cloud-seat adds, LTM expansion). Reconcile against **forward** pipeline (open opps + late-stage commit + best-case). Best when historical attainment is a clean signal of forward capacity (mature business, low churn in territories, stable rep tenure).

```python
historical_metrics = [
    MetricSpec('NetNewACV',  direction='proportional', weight=1.0, lookback=4),
    MetricSpec('CloudSeats', direction='proportional', weight=0.6, lookback=4),
    MetricSpec('DCSeats',    direction='inverse',      weight=0.4, lookback=4),
]
quotas = cascader.cascade_quota('Global_Corp', macro_target, metrics=historical_metrics)

# Reconcile against forward pipeline
adjuster = PipelineAdjuster(hierarchy, quotas,
                            pipeline_attr=['Open_Pipeline', 'Late_Stage_Commit'])
```

**White-space planning** — *"what can be achieved if we look at the opportunity in front of us?"*

Cascade on **forward-looking** signals (current installed seats, knowledge-worker counts, white-space indicators), with dampeners (LTM spend) and hard gates (unmigrated seats). Reconcile against **historical** attainment to flag where the plan asks for a step-up. Best when past performance is noisy (rapid growth, territory shuffles, recent re-orgs) and the org wants every rep to be measured against the opportunity in front of them.

```python
forward_metrics = [
    MetricSpec('Current_Seats_ProductX',  direction='proportional', weight=1.0,
               columns=['Current_Seats_ProductX']),
    MetricSpec('Knowledge_Workers_Count', direction='proportional', weight=0.7,
               columns=['Knowledge_Workers_Count']),
    MetricSpec('LTM_ExpansionSpent',      direction='inverse',      weight=0.5,
               columns=['LTM_ExpansionSpent']),
]
gate_metrics = [
    MetricSpec('Unmigrated_Seats', columns=['Unmigrated_Seats']),  # threshold defaults to 0
]
quotas = cascader.cascade_quota(
    'Global_Corp', macro_target,
    metrics=forward_metrics, gate_metrics=gate_metrics,
)

# Reconcile against historical attainment
adjuster = PipelineAdjuster(hierarchy, quotas, pipeline_attr=[
    'Q1_NetNewACV', 'Q2_NetNewACV', 'Q3_NetNewACV', 'Q4_NetNewACV',
])
diagnosis = adjuster.diagnose(coverage_thresholds={
    '_default': {'healthy': 1.0, 'at_risk': 0.75},   # ratios near 1.0, not 1.5–3x
})
```

Neither philosophy is "correct" — they answer different questions. The package supports either as a first-class flow, and you can blend them (some metrics historical, some forward) by mixing them in a single `metrics=` list.

### What's New in v0.3.x

- **Multi-metric cascading** via the new `MetricSpec` API — blend historical NetNewACV with any number of secondary signals (cloud seats, on-prem seats, LTM expansion spend, customer-sat scores, certification flags, anything else the analyst tracks), each marked as `proportional` or `inverse`, with per-metric weights and lookbacks
- **Direction is always a user input.** Domain knowledge ("more cloud seats means more ACV") trumps statistical sign. The package surfaces correlations and warns on mismatch but never overrides the analyst's call
- **`MetricSpec.suggest_weights(...)`** suggests weights (magnitude of correlation) for user-declared directions. For exploratory use, `MetricSpec.suggest_directions_and_weights(...)` infers both
- **Normalized-weights view** — `MetricSpec.normalized_weights(specs)` shows the post-normalization share each metric actually contributes; auto-printed before every multi-metric cascade and accessible via `cascader.weights_report`
- **Brand-new IC handling — either-or:** flag brand-new ICs in the same CSV the analyst already uploads (`brand_new_col='Is_Brand_New'` on `SalesHierarchy.from_dataframe`, then `new_ic_attr='_is_brand_new'` on `cascade_quota`), OR pick a rule (`new_ic_rule='all_metrics_zero'` / `'primary_metric_zero'`). Passing both raises `ValueError`
- **Any metric name, any numeric type** — including booleans (`Has_Active_Cert: True/False`). Boolean / 0-1 sparse metrics are auto-detected and excluded from zero-imputation so False isn't mistaken for missing data
- **`PipelineAdjuster` accepts multiple pipeline columns** — `pipeline_attr=['Open_Pipeline', 'Late_Stage_Commit', 'Best_Case_Adds']` sums them per IC into a combined dollar amount for the coverage ratio
- **CSV / SQL / dashboard exports** — every output converts to a DataFrame via `cascader.quotas_to_dataframe(...)`, `cascader.quotas_diff_to_dataframe(...)`, or `reconciler.reconcile_all(...)`. From there `.to_csv()`, `.to_sql()`, or `cascader.to_html_dashboard(...)` writes wherever you need
- **Hedge audit columns** — pass `unhedged_quotas=` to `quotas_to_dataframe` for `unhedged_quota`, `hedge_buffer`, and `overassignment_pct` columns showing exactly how much of each quota is hedge buffer
- **Fully backward compatible** — `cascade_quota(...)` without `metrics=` behaves exactly as in v0.2.x

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

### 3. Multi-Metric Cascading (v0.3+)

For real B2B planning, the metric you're cascading (e.g., NetNewACV) is rarely the only signal that should drive its allocation. Cloud-seat counts predict more new ACV; on-prem (DC) seat counts predict less; high LTM expansion spend means the account is already saturated. The `MetricSpec` API lets you mix any number of these into a single cascade.

**Direction is always your call.** You declare whether each metric is `proportional` (more → more quota) or `inverse` (more → less quota) up front. The package surfaces correlations and warns when the data sign disagrees, but never overrides your domain knowledge.

```python
from b2b_revenue_forecasting import MetricSpec

# Declare each metric's role — direction is required, weight is your knob
metrics = [
    MetricSpec('NetNewACV',     direction='proportional', weight=1.0, lookback=4),
    MetricSpec('CloudSeats',    direction='proportional', weight=0.5, lookback=4),
    MetricSpec('DCSeats',       direction='inverse',      weight=0.4, lookback=4),
    MetricSpec('ExpansionSpent',direction='inverse',      weight=0.7,
               columns=['LTM_ExpansionSpent']),  # single LTM column
]

quotas = cascader.cascade_quota(
    'Global_Corp', 100_000_000.0,
    hedge_multiplier=1.05,
    metrics=metrics,
)
```

**Any metric name, any data type works.** `Customer_Sat_Score`, `MQLs_Sourced_via_Outbound`, `Has_Active_Cert` (boolean), `Renewals_Caught_Up` (0/1 counter) — anything numeric, with any column name. Boolean and 0/1 sparse metrics are auto-detected and excluded from zero-imputation so `False` isn't treated as a missing value.

**How the blend works.** At every level, each child gets a share of the parent's quota equal to a weighted sum of its per-metric shares-of-siblings. Proportional metrics use raw shares; inverse metrics flip via reciprocal-then-normalize. The final per-child share is `Σ_m (weight_m × share_m(child))`, which sums to 1 across siblings.

**Don't know the weights?** Pass `direction=` on each candidate, let `suggest_weights()` propose magnitudes via Pearson correlation:

```python
suggestions, report = MetricSpec.suggest_weights(
    df,
    target_column='NetNewACV_4Q_sum',
    candidate_metrics=[
        {'name': 'CloudSeats',     'column': 'CloudSeats_4Q_sum',
         'direction': 'proportional', 'lookback': 4},
        {'name': 'DCSeats',        'column': 'DCSeats_4Q_sum',
         'direction': 'inverse',      'lookback': 4},
        {'name': 'ExpansionSpent', 'column': 'LTM_ExpansionSpent',
         'columns': ['LTM_ExpansionSpent'],
         'direction': 'inverse',      'lookback': 1},
    ],
)
# report['CloudSeats']['weight'] == 0.62, ['rationale'] explains why,
# ['direction_matches_data'] tells you if your call agrees with the sign

quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0, metrics=suggestions)
```

For pure exploration (you don't yet have a domain opinion), use `MetricSpec.suggest_directions_and_weights(...)` — it infers both from data. This is a sanity-check helper, not a production-planning API.

**Brand-new ICs — either-or, your choice of where they're listed.** The cleanest option keeps everything in the same CSV the analyst already uploads:

```python
# CSV has a column Is_Brand_New with True / 1 / "yes" for each new hire
hierarchy = SalesHierarchy()
hierarchy.from_dataframe(
    df, path_cols=[...], metrics_cols=[...],
    brand_new_col='Is_Brand_New',     # ingested as node attribute _is_brand_new
)

quotas = cascader.cascade_quota(
    'Global_Corp', 100_000_000.0,
    metrics=metrics,
    new_ic_attr='_is_brand_new',       # read the flag from the CSV
)
```

Or, if you don't want a separate column, pick an auto-detection rule:

```python
quotas = cascader.cascade_quota(
    'Global_Corp', 100_000_000.0,
    metrics=metrics,
    new_ic_rule='all_metrics_zero',    # or 'primary_metric_zero'
)
```

You pick one or the other — passing both an explicit identifier (`new_ic_attr` or `new_ic_ids`) AND `new_ic_rule` in the same call raises `ValueError`, because the two would silently disagree.

Brand-new ICs get an equal-share carve-out of the team target before the remainder is split proportionally — just like the single-metric path.

### 4. Detect & Fix Forecasting Bias

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

### 5. Export to CSV, SQL, or an Interactive Dashboard

Every output is a pandas DataFrame, so the same code writes anywhere:

```python
# CSV — analyst-ready, one row per node at every level
cascaded_df = cascader.quotas_to_dataframe(quotas, level_names=taxonomy)
cascaded_df.to_csv('cascaded_quotas.csv', index=False)

# CSV with hedge audit — also include the unhedged baseline
quotas_unhedged = cascader.cascade_quota(
    'Global_Corp', 100_000_000.0, hedge_multiplier=1.0,
    metrics=cascade_metrics, verbose=False,
)
cascader.quotas_to_dataframe(
    quotas, level_names=taxonomy, unhedged_quotas=quotas_unhedged,
).to_csv('cascaded_quotas_with_audit.csv', index=False)
# → adds unhedged_quota, hedge_buffer, overassignment_pct columns

# SQL — same DataFrames, any SQLAlchemy-compatible database
import sqlite3
with sqlite3.connect('cascade.db') as conn:
    cascaded_df.to_sql('cascaded_quotas', conn, if_exists='replace', index=False)
    cascader.weights_report.to_sql('normalized_weights', conn,
                                    if_exists='replace', index=False)
# Postgres / Snowflake / BigQuery: swap conn for a SQLAlchemy engine

# Interactive HTML dashboard — Chart.js, self-contained, shareable
cascader.to_html_dashboard(
    quotas, output_path='cascade_dashboard.html',
    title='Q1 Cascade — $100M Plan',
    unhedged_quotas=quotas_unhedged,
    adjusted_quotas=adjusted, diagnosis=diagnosis,
)
```

### 6. Pipeline Health Diagnosis & Redistribution

```python
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster

# Single pipeline column (backward compat)
adjuster = PipelineAdjuster(hierarchy, quotas, pipeline_attr='Current_Pipeline')

# Or sum multiple dollar-denominated pipeline columns from the same CSV
adjuster = PipelineAdjuster(hierarchy, quotas, pipeline_attr=[
    'Open_Pipeline', 'Late_Stage_Commit', 'Best_Case_Adds',
])

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

### How Weights Become Influence

Weights you set on `MetricSpec`s are **relative**, normalized to sum to 1 across **active** metrics (weight > 0) at cascade time; inactive metrics contribute exactly 0. A metric's real influence is `weight / sum(active weights)`:

```
raw weights [1.0, 0.5, 0.0]        ->  influence [66.7%, 33.3%, 0%]
raw weights [1.0, 0.98, 0.4, 0.067] -> 0.067 / 2.447 = 2.7% (not 6.7%!)
```

Always check the actual shares with `MetricSpec.normalized_weights(specs)` or `cascader.weights_report` — the same table auto-prints before every verbose multi-metric cascade, and it's the table to show stakeholders.

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
| ≥ healthy threshold | 🟢 Healthy | May receive quota |
| ≥ at_risk threshold | 🟡 Moderate | No action |
| ≥ 1.0 | 🟠 At Risk | May donate quota |
| < 1.0 | 🔴 Critical | Urgent — pipeline below target (May donate quota) |

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
| [Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6456999) | **SSRN** (Preprint) | ✅ Published |
| Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation in B2B Enterprises | **Journal of Revenue and Pricing Management** (Springer) | ⏳ Under Review |

If you use this package in your research, please cite:

```
Karwa, S. (2026). Graph-Theoretic Approaches to Hierarchical Revenue Target Allocation
in B2B Enterprises: A Methodological Framework. SSRN Working Paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6456999
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
