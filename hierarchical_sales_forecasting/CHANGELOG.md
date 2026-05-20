# Changelog

All notable changes to `b2b_revenue_forecasting` are documented here.
This project loosely follows [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-05

### Added
- **Gate metrics** — hard kill-switches via
  `cascade_quota(..., gate_metrics=[MetricSpec(...)])`. Any node whose
  rolled-up gate value is ≤ `gate_threshold` (default 0.0) is excluded
  from the cascade entirely (quota = 0), and its share is redistributed
  among non-gated siblings via the existing blend. Gates propagate
  upward naturally (a manager whose whole team fails the gate gets $0).
  Multiple gates compose with AND. CRO overrides win over gates.
  Designed for white-space planning: e.g., gating migration NetNewACV
  on `Unmigrated_Seats` zeros out territories with nothing left to
  migrate.
- **`MetricSpec.gate_threshold`** field (default 0.0) — meaningful only
  when the spec is passed in `gate_metrics`.
- **`cascader.gated_nodes`** set is populated after every cascade with
  gates so analysts can inspect which nodes were excluded.
- **`is_gated` column** in `quotas_to_dataframe` output (added
  automatically when gates were used) so CSVs distinguish "$0 because
  gated" from "$0 because no signal."
- **README documents two planning philosophies** (earned vs.
  white-space) as equal-status flows, with code samples for each.
- **Walkthrough Part B** added: white-space planning end-to-end
  (forward-looking cascade with gates → historical-attainment
  reconciliation).

### Notes
- Fully backward compatible. Cascades without `gate_metrics=` behave
  identically to v0.3.4.

## [0.3.4] — 2026-05

### Added
- **`QuotaCascader.to_html_dashboard(quotas, output_path, ...)`** — generates
  a self-contained interactive HTML dashboard (Chart.js via CDN). Shows
  per-region quota with optional unhedged-base overlay, top N IC quotas,
  top redistributions (original vs adjusted), pipeline-coverage risk
  status, and a per-region summary table. Opens in any browser, no
  server required.
- **Hedge audit columns in `quotas_to_dataframe`** — pass `unhedged_quotas`
  (a second cascade with `hedge_multiplier=1.0`) and the output gains
  `unhedged_quota`, `hedge_buffer`, and `overassignment_pct` columns so
  stakeholders can see exactly how much of each quota is hedge buffer.
- Walkthrough demonstrates writing all five outputs to a SQL database
  (SQLite shown; pattern works for Postgres/MySQL/Snowflake/BigQuery
  via SQLAlchemy + `df.to_sql()`).

## [0.3.3] — 2026-05

### Added
- **`QuotaCascader.quotas_to_dataframe(quotas, level_names=None)`** —
  converts a cascade result dict into a hierarchy-aware DataFrame
  (depth, level name, parent, is_leaf, cascaded_quota), sorted top-down
  from root to leaves. Ready for `.to_csv()` or `.to_sql()`.
- **`QuotaCascader.quotas_diff_to_dataframe(original, adjusted, ...)`** —
  before/after redistribution comparison with `delta` and `delta_pct`
  columns.
- **`CommitReconciler.reconcile_all(commits_df, ...)`** — batch reconcile
  every manager and return a DataFrame with bias quotient, bias label
  (sandbagger / happy_ears / truth_teller), adjusted commit, and
  optional blended forecast.

## [0.3.2] — 2026-05

### Added
- **`MetricSpec.normalized_weights(specs)`** — DataFrame with input
  weight, normalized share, direction, and active status per metric.
  Useful for explaining the cascade to stakeholders.
- **`MetricSpec.format_normalized_weights(specs)`** — pretty-printed
  string of the same view.
- **`QuotaCascader.cascade_quota(..., verbose=True)`** — in multi-metric
  mode, the normalized-weights table now auto-prints before every
  cascade. Also stored on `cascader.weights_report` for later access.
  Pass `verbose=False` to suppress in batch / test runs.

## [0.3.1] — 2026-05

### Added
- Walkthrough refresh: all secondary metrics now go through
  `suggest_weights` (including the boolean `Has_Active_Cert` via a
  per-IC 4Q sum); Pattern A weight override is demonstrated.

## [0.3.0] — 2026-05

### Added
- **Multi-metric cascading** via the new `MetricSpec` API. Blend
  historical NetNewACV with any number of secondary signals (cloud
  seats, on-prem seats, LTM expansion spend, certification flags,
  anything else the analyst tracks), each marked as `proportional` or
  `inverse`, with per-metric weights and lookbacks.
- **Direction is always a user input.** Domain knowledge ("more cloud
  seats means more ACV") trumps statistical sign. The package surfaces
  correlations and warns on mismatch but never overrides the analyst's
  call.
- **`MetricSpec.suggest_weights(...)`** — suggests weights (magnitude
  of Pearson correlation) for user-declared directions.
- **`MetricSpec.suggest_directions_and_weights(...)`** — exploratory
  helper that infers both direction and weight from data.
- **Brand-new IC handling — either-or:** flag brand-new ICs in the same
  CSV the analyst already uploads (`brand_new_col='Is_Brand_New'` on
  `SalesHierarchy.from_dataframe`, then `new_ic_attr='_is_brand_new'`
  on `cascade_quota`), OR pick a rule
  (`new_ic_rule='all_metrics_zero'` / `'primary_metric_zero'`).
  Passing both raises `ValueError`.
- **Any metric name, any numeric type** — including booleans
  (`Has_Active_Cert: True/False`). Boolean / 0-1 sparse metrics are
  auto-detected and excluded from zero-imputation so False isn't
  mistaken for missing data.
- **`PipelineAdjuster` accepts multiple pipeline columns** —
  `pipeline_attr=['Open_Pipeline', 'Late_Stage_Commit',
  'Best_Case_Adds']` sums them per IC into a combined dollar amount
  for the coverage ratio.

### Backward Compatibility
- `cascade_quota(...)` without `metrics=` behaves exactly as in v0.2.x.
  Legacy single-metric `_Attainment` path produces identical quotas to
  the centavo.

## [0.2.5] and earlier

See README "What's New in v0.2.0" section for prior history.
