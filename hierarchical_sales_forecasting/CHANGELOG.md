# Changelog

All notable changes to `b2b_revenue_forecasting` are documented here.
This project loosely follows [Semantic Versioning](https://semver.org/).

## [0.16.0] — 2026-07

The aggregate-pinning release — closes
[#22](https://github.com/shreyasrkarwa/Analytics/issues/22),
[#31](https://github.com/shreyasrkarwa/Analytics/issues/31), and
[#24](https://github.com/shreyasrkarwa/Analytics/issues/24).
Note several premises of #22/#31 were already addressed in v0.13.0
(any-level per-cascade pins, parent conservation, `override_basis`,
`rehedge`); what remained — and ships here — is pinning a node to an
exact total ACROSS many cascades.

### Added (issues #22/#31 — aggregate pins)
- **`Pin(node, total, basis='base'|'cascaded', scope={...},
  exclude=[...])`** and
  **`apply_pins(quotas_long, pins, freeze_nodes=None, row_keys=None)
  -> (edited_df, feasibility_report)`**, both exported at package root.
  Post-cascade on the long frame:
  - the pinned node's rows are scaled to SUM to the total,
    proportional to its baseline mix (products/quarters keep their
    shape); `scope` narrows which rows count (e.g. Q1 only);
  - within each cascade the delta is absorbed by eligible siblings
    proportional to their base — manager pins and manager absorbers
    rescale their whole subtrees, so every depth stays consistent;
  - parents conserve exactly wherever absorption succeeds; siblings
    floor at $0 (never negative), and anything unabsorbable is
    reported in the feasibility frame (`unabsorbed`,
    `feasible=False`) plus a warning;
  - all math on the BASE layer; hedged values re-derived from each
    row's own original ratio (`basis='cascaded'` totals the hedged
    layer instead, base derived);
  - provenance: `is_pinned` / `pin_type` ('leaf'/'subtree') columns;
    pins apply in order and pinned nodes never absorb for each other.
  Replaces the consumer's ~200-line rescale workaround.

### Added (issue #24 — freeze/exclude absorbers)
- Per-pin **`exclude=[...]`** and global **`freeze_nodes=[...]`**:
  protected nodes are never absorbers and never modified. (The same
  protection exists in-cascade via `PipelineAdjuster.adjust(
  locked_nodes=...)` for pipeline redistribution.)

## [0.15.0] — 2026-07

Implements [issue #14](https://github.com/shreyasrkarwa/Analytics/issues/14):
per-group / conditional gating in `cascade_many`.

### Added
- **`gate_metrics` accepts a callable** — evaluated once per
  combination with the group-key dict, returning that combination's
  gate list (or None for no gates). E.g. gate only Migration:
  `gate_metrics=lambda g: DC_GATE if g['st1_sales_type'] ==
  'Migration' else None`. A mapping-style policy is one line via
  `dict.get` (documented recipe). Replaces the split-and-concat
  two-call workaround — pinned as an exact-equivalence test. Bad
  return types and raising policies flow through the existing
  `on_error` / dropped-targets machinery. Static lists unchanged.

### Fixed
- **`quotas_long.attrs['dropped_targets']` is now a list of record
  dicts** instead of a DataFrame. Storing a DataFrame in `.attrs`
  made `pd.concat` on two cascade outputs raise "truth value of a
  DataFrame is ambiguous" (pandas compares attrs during finalize).
  Reconstruct with `pd.DataFrame(quotas.attrs['dropped_targets'])`;
  `return_dropped=True` remains the primary DataFrame channel.

## [0.14.0] — 2026-07

The routing release — closes
[#26](https://github.com/shreyasrkarwa/Analytics/issues/26),
[#25](https://github.com/shreyasrkarwa/Analytics/issues/25), and
[#32](https://github.com/shreyasrkarwa/Analytics/issues/32) (what
happens to target money with no hierarchy branch to land on).

### Added (issue #26 — dropped targets are data)
- **`cascade_many(return_dropped=True)`** returns a third frame with
  every dropped target row: all original columns plus a `reason`
  column. The same frame is ALWAYS attached as
  `quotas_long.attrs['dropped_targets']`. Warnings were already
  emitted per skipped combination; now the money is programmatically
  visible instead of log noise. (`on_error='raise'` remains the strict
  mode.)

### Added (issue #25 — cross-tree routing)
- **`route_targets(targets, quotas_long, recipients, target_col,
  recipient_keys=None, split='base_quota', rollup=True)`** — carry
  target rows on named recipient nodes in a DIFFERENT part of the
  tree. Mechanics: split on the BASE layer proportional to the
  recipients' existing `base_quota` (or `'equal'`, or any numeric
  column); each routed node's `cascaded_quota` is derived from its own
  existing hedge ratio (never re-hedged — the #21 contract); ancestor
  rollups emitted so the routed slice reconciles at every depth;
  every routed row keeps ALL columns of its originating target row
  plus `routed=True`, so the money stays attributable to the original
  segment. Additive by construction — concatenate to `quotas_long`.
  Per-product recipient exclusions are two calls with different
  filters (documented recipe).

### Closed by composition (issue #32 — unmatched-target routing)
- The requested `unmatched_targets='route'` + `route_map` is
  deliberately NOT built into `cascade_many` (per-combo routing
  semantics inside the batch loop are opaque). The supported pattern
  is two lines:
  `..., dropped = cascade_many(..., return_dropped=True)` →
  `route_targets(dropped, quotas, recipients=[...], ...)`.
  'error' was already available via `on_error='raise'`; 'skip' is the
  default.

## [0.13.0] — 2026-07

The override release — closes
[#28](https://github.com/shreyasrkarwa/Analytics/issues/28),
[#21](https://github.com/shreyasrkarwa/Analytics/issues/21), and
[#23](https://github.com/shreyasrkarwa/Analytics/issues/23) together
(pins x hedges x conservation).

### Fixed (issue #28 — override-path hardening)
- Verification first: the reported mechanism ("remainder split by
  shares of ALL children") does NOT exist — unpinned siblings were
  already renormalized, and the issue's 5-rep/$2.6M-pin repro conserves
  the parent exactly (now a pinned regression test). Probing the path
  surfaced four REAL defects, all fixed:
  - **Manager pins were silently ignored.** Pins now work at ANY
    level: pinning a manager fixes that subtree's total, which then
    cascades normally within the subtree.
  - **Jagged-leaf pins were silently ignored** (an IC whose siblings
    are managers). Now honored.
  - **Pin + all-remaining-brand-new over-distributed** (e.g. $6.6M
    from a $5M pool). The brand-new equal-share carve-out is now
    capped at the remaining pool; with no experienced siblings the
    pool splits equally. Parents conserve exactly.
  - **Pins exceeding the pool produced NEGATIVE sibling quotas,
    silently.** Siblings now floor at $0, a warning fires, and the
    excess is reported via `cascader.overpinned` /
    `cascader.overpinned_nodes` and new `overpinned_amount` /
    `overpinned_nodes` keys in `gating_report()` (whose `reconciles`
    goes False — children legitimately sum above the parent).
  - If EVERY child of a funded node is pinned and pool remains, the
    leftover is reported via the unallocated machinery instead of
    vanishing.

### Added (issue #23 — pin basis)
- **`override_basis` on `cascade_quota`** (flows through
  `cascade_many`): `"base"` *(default)* — the pin is the un-hedged
  plan number; the hedged value derives as pin x the node's compound
  hedge factor, so pinned nodes hedge like everyone else and BOTH
  layers conserve. `"cascaded"` — the pin is the exact final number;
  the base derives by dividing the factor out. Pre-0.13.0 the raw pin
  was used in both layers (pinned reps silently received no hedge);
  with no hedging the bases are identical, so unhedged flows are
  unaffected.

### Added (issue #21 — post-cascade edits)
- **`cascader.hedge_ratios()`** — per-node cascaded/base ratio from
  the last run.
- **`cascader.rehedge(edited_base)`** — recompute the hedged layer
  from an edited base layer, preserving each node's original hedge
  ratio. Documented workflow: do ALL pin math and parent rollups on
  `base_quota`, then derive `cascaded_quota` via `rehedge` — never sum
  hedged leaves into parents (that double-counts buffers up the tree,
  the #21 failure mode). Replaces the consumer's ~40-line workaround.

## [0.12.0] — 2026-07

Fixes [issue #33](https://github.com/shreyasrkarwa/Analytics/issues/33):
small/low-variance slices no longer silently equal-split. Correlation
is statistically undefined below 3 paired observations — and tiny
slices (a sub-region with 2 teams) are the COMMON case in per-group
batch runs. Zeroing those candidates' weights threw away the metric
VALUES, so siblings with a 6x seat difference received identical
quotas (field-reported).

### Changed
- **`on_degenerate` parameter** on `suggest_weights` and
  `suggest_directions_and_weights`:
  - `"proportional"` *(new default)* — statistically-degenerate
    candidates KEEP their declared weight (1.0 unless set), so the
    cascade splits proportionally to the blended metric values. Works
    with any number of metrics (each child's share is the weighted
    average of its per-metric share-of-siblings); directions still
    apply.
  - `"equal"` — pre-0.12.0 behavior: weight 0; if every candidate
    degrades, the cascade equal-splits (with the v0.10.2 warning).
  - `"raise"` — fail loudly for callers who handle thin slices
    upstream.
- Only true statistical degeneracy (n < 3 paired observations, zero
  variance) triggers the fallback. A MISSING column is absent data,
  not thin data: it stays weight 0 in every mode and never raises.

### Added
- Report entries now carry **`degenerate`** and **`fallback`** fields
  in every path, and a summary warning names the degenerate candidates
  in proportional mode — loud in the report, not just the log.

### Notes
- The issue's France-South repro is pinned by tests: the two teams now
  split ~70/30 under the default (blend of ~20%-apart knowledge
  workers and ~6x-apart cloud seats) instead of 50/50. Also pinned:
  single-metric 6x ratio, per_group batch runs, and missing-column
  behavior. Related feature request #34 (a fully correlation-free
  proportional weighting mode) remains open.

## [0.11.0] — 2026-07

Implements [issue #13](https://github.com/shreyasrkarwa/Analytics/issues/13):
per-DEPTH hedging. `hedge_multiplier` accepted a flat float or a
per-node dict — but depth-based policies ("deepest managers 1.10, the
level above 1.05") were impossible through `cascade_many`, which builds
each combination's hierarchy internally so node ids are never visible
to the caller.

### Added
- **`HedgeByDepth(from_leaves=None, from_root=None, default=1.0)`** —
  a depth-keyed hedge spec accepted anywhere `hedge_multiplier` is
  (exported at package root). Resolved at cascade time against the
  cascader's own graph into an ordinary per-node dict, so all
  downstream behavior (base layer, audit columns, reconciliation) is
  identical to passing that dict by hand — and it flows through
  `cascade_many` per combination automatically.
  - `from_leaves`: keyed by distance to the FARTHEST descendant leaf
    (1 = deepest manager whose children are ICs) — the natural basis
    for front-line-manager policies; correct in jagged hierarchies
    where root-depth is not equivalent.
  - `from_root`: keyed by `node_depths()`-style distance from the root.
  - Both bases may be combined; a node matched by both gets the
    PRODUCT. Leaves never carry a hedge. Invalid specs raise
    `ValueError` at construction.
  - `HedgeByDepth.resolve(graph)` is public for inspection.

### Notes
- The issue's worked example (`from_leaves={1: 1.10, 2: 1.05}`) is
  pinned by tests, including a jagged-hierarchy case, a
  spec-vs-hand-built-dict equality check, and a `cascade_many`
  end-to-end run asserting the base layer still reconciles at every
  depth.
- Float and per-node-dict behavior unchanged.

## [0.10.2] — 2026-07

Closes [issue #8](https://github.com/shreyasrkarwa/Analytics/issues/8):
`suggest_weights` on small / degenerate slices. Verification showed the
feared crashes no longer exist — every degenerate input (single row,
zero variance, all-null target/candidate, empty frame) already degrades
to weight 0 with a written rationale, batch runs are shielded by
`cascade_many(on_error='skip')`, and all-zero weights cascade as an
equal split. What was missing was a heads-up.

### Added
- **All-zero weights warning.** `suggest_weights` and
  `suggest_directions_and_weights` now emit one `UserWarning` when
  EVERY candidate degrades to weight 0, stating that cascading with
  these specs will fall back to an EQUAL SPLIT among siblings and
  including the per-metric rationales. Partial degradation (some
  candidates survive) stays silent, as does healthy data.

### Notes
- A missing `target_column` still raises `ValueError` — that's a
  configuration typo, not degenerate data, and should be loud.
- No behavior changes to weights or cascades.

## [0.10.1] — 2026-07

Implements [issue #11](https://github.com/shreyasrkarwa/Analytics/issues/11):
surface weight-normalization semantics where weights are set. The
"raw weight != influence" nuance lived only in the
`normalized_weights()` docstring; users setting `weight=0.067` were
surprised it wasn't 6.7% influence.

### Docs
- **`MetricSpec.weight`** now documents the normalization rule at the
  point of use, with a worked example: influence =
  `weight / sum(active weights)`; active = weight > 0; inactive
  metrics contribute exactly 0. `[1.0, 0.5, 0.0]` → `[66.7%, 33.3%,
  0%]`; `0.067` alongside `[1.0, 0.98, 0.4]` → 2.7%, not 6.7%.
- **`cascade_quota(metrics=)`** cross-references the same rule and
  points at `self.weights_report` / `MetricSpec.normalized_weights`.
- **README** gains a "How Weights Become Influence" section under Key
  Concepts.
- The documented examples are pinned by a unit test
  (`test_api_consistency.py::test_weight_normalization_docs_example`)
  so docs and implementation can't drift apart.

### Notes
- No code behavior changes.

## [0.10.0] — 2026-07

Implements [issue #10](https://github.com/shreyasrkarwa/Analytics/issues/10):
a consolidated gating report. The pieces existed since v0.5.0
(`gated_nodes`, `gate_relaxed_nodes`, `unallocated`,
`is_gated`/`gate_relaxed`/`is_unallocated` columns,
`reconciliation_report`) but had to be assembled by hand.

### Added
- **`QuotaCascader.gating_report(tolerance=0.01)`** — one dict
  summarizing the most recent cascade:
  `target`, `gated_count`, `gated_node_ids`, `gated_leaf_ids`,
  `gate_relaxed_node_ids`, `unallocated_amount`, `unallocated_nodes`,
  `leaf_quota_sum` (hedged), `leaf_base_sum` (un-hedged),
  `base_gap` (= target − leaf_base_sum − unallocated_amount), and
  `reconciles` — True iff every input dollar is either on an IC (base
  layer) or explicitly reported as unallocated. Raises `RuntimeError`
  if called before any cascade.
- `cascade_quota` now remembers its inputs/outputs
  (`cascader.last_target`, `cascader.last_quotas`) to power the report.

### Notes
- Purely additive; no behavior changes.

## [0.9.0] — 2026-07

Implements [issue #9](https://github.com/shreyasrkarwa/Analytics/issues/9):
configurable gate threshold & semantics. Gating was hardcoded to
"gated iff value <= gate_threshold"; boolean flags and
"too-much-of-X" signals needed SQL-side contortions.

### Added
- **`MetricSpec.gate_mode`** — defines when a node PASSES its gate
  (checked against the leaf-summed aggregate; failing ANY gate =
  gated):
  - `"gt"` *(default)* — pass iff `value > gate_threshold`
    (byte-for-byte the pre-0.9.0 behavior).
  - `"ge"` — pass iff `value >= gate_threshold` ("at least N seats",
    boundary counts).
  - `"lt"` / `"le"` — pass iff value is BELOW the threshold: gate
    reps/subtrees with too much of a signal (churn tickets, backlog).
  - `"truthy"` — pass iff `bool(value)`; threshold ignored (boolean
    entitlement flags).
  Invalid modes raise `ValueError` at construction. The exact
  predicate is documented on `MetricSpec` and in
  `QuotaCascader._passes_gate`.

### Notes
- For `"lt"`/`"le"` gates the leaf-summed rollup grows with subtree
  size, so a parent can fail while its children pass; any resulting
  fully-gated level is absorbed by the v0.5.0 `gate_fallback`
  machinery — reconciliation invariants hold for every mode.
- Defaults unchanged; existing cascades are unaffected.

## [0.8.0] — 2026-07

Implements [issue #7](https://github.com/shreyasrkarwa/Analytics/issues/7):
`quotas_to_dataframe` can now carry source hierarchy attributes and a
sanitized→original id mapping, so exports are analysis-ready without a
manual merge (or un-sanitizing step).

### Added
- **`metadata_cols` on `from_dataframe`** — descriptive columns (rep
  name, segment, geo, employee id, ...) attached to leaf nodes AS-IS:
  never coerced, never aggregated, never read as metric signal.
- **`metadata_cols` on `quotas_to_dataframe`** — emits stored node
  attributes as output columns (NaN on nodes without them).
- **`source_df` + `source_join_col` on `quotas_to_dataframe`** —
  LEFT-JOINs the original source frame onto leaf rows. The join is
  keyed on ORIGINAL ids, so it works even for nodes renamed by the
  v0.6.0 collision policy. Overlapping columns get a `_source` suffix.
- **`SalesHierarchy.id_map`** — `{sanitized_id: original_value}`
  recorded whenever `on_collision='suffix'` renames a node; an
  **`original_id` column** appears automatically in
  `quotas_to_dataframe` output whenever any rename occurred.
- **`cascade_many(metadata_cols=...)`** — passthrough: metadata rides
  along on every combination's leaf rows and is excluded from metric
  aggregation.

### Notes
- No behavior changes for existing calls; all new parameters are
  optional and off by default.

## [0.7.2] — 2026-07

Fixes [issue #5](https://github.com/shreyasrkarwa/Analytics/issues/5):
inconsistent graph accessor naming. `SalesHierarchy` exposed `.graph`
while `QuotaCascader` called the same object `self.hierarchy`, so
`hierarchy.hierarchy` raised `AttributeError` and helpers needing the
raw graph required reading the source.

### Changed
- **`.graph` is the canonical accessor everywhere.** `QuotaCascader`
  now stores the graph as `.graph` (matching `SalesHierarchy` and
  `PipelineAdjuster`). Backward compatible: `cascader.hierarchy` still
  works as a documented read-only alias, and `SalesHierarchy` gains a
  forgiving `.hierarchy` alias too — both names resolve to the same
  `nx.DiGraph` on both classes.

### Added
- **Read-only helpers on `SalesHierarchy`** so consumers rarely need
  the raw graph:
  - `roots()` — nodes with no parent.
  - `leaves(root=None)` — all ICs, or just those under `root`.
  - `managers(root=None)` — all non-leaf nodes, or those under `root`.
  - `node_depths()` — `{node_id: depth}` from the root(s), e.g. for
    building per-level hedge dicts.

### Notes
- No behavior changes — cascade outputs are identical.

## [0.7.1] — 2026-07

Fixes [issue #6](https://github.com/shreyasrkarwa/Analytics/issues/6):
the `MetricSpec.name` / `columns` coupling. Specs returned by
`suggest_weights` (which carry `columns=None`) resolved to the
`Q1_<name>...Q4_<name>` convention; when the data actually lives in a
single column named exactly `<name>`, the metric silently read nothing
and contributed 0 — forcing the
`for s in suggested: s.columns = [s.name]` workaround.

### Fixed
- **Plain-`<name>` runtime fallback.** When `columns` is unset and NONE
  of the `Qi_<name>` convention columns exist on a leaf, the cascader
  now reads the attribute named exactly `<name>`. Specs from
  `suggest_weights` are directly usable; the workaround is obsolete.
  Resolution order: explicit `columns=` always wins → `Qi_<name>`
  convention → plain `<name>` fallback. Applies to cascade metrics,
  gate metrics, and brand-new-IC detection alike.
- **Tree-wide zero-signal warning.** `cascade_quota` now warns when an
  active metric aggregates to zero across the entire tree, naming the
  columns it tried — a mis-resolved column can no longer silently
  degrade an allocation.

### Docs
- The `name`/`columns`/`resolved_columns()` contract is now documented
  on the `MetricSpec` dataclass (including the note that
  `suggest_weights`' `column=` field is correlation-only and is never
  copied into `columns`).

### Notes
- No API changes. Cascades whose metrics already resolved correctly
  are byte-for-byte unchanged; the fallback only activates where the
  package previously read silent zeros.

## [0.7.0] — 2026-07

Implements [issue #4](https://github.com/shreyasrkarwa/Analytics/issues/4):
a native batch / multi-combination cascade API. Replaces the ~150 lines
of consumer boilerplate (filter per combo → build hierarchy → suggest
weights → cascade → tag → concat) with one call that centralizes all
the correctness work from v0.5.0–v0.6.1.

### Added
- **`cascade_many(hierarchy_df, target_df, group_keys, target_col,
  taxonomy, ...)`** — top-level function (also exported from the
  package root). For each unique `group_keys` combination in
  `target_df` it filters `hierarchy_df`, builds a validated
  `SalesHierarchy` once, resolves weights, and cascades EVERY matching
  target row — extra `target_df` columns (e.g. `fiscal_quarter`) act
  as sub-targets that reuse the prepared combination
  ("prepare once / cascade many").
  - Returns `(quotas_long_df, weights_long_df)`: tidy long frames
    tagged with the group keys, sub-target columns, and the input
    target. Quota rows carry `cascaded_quota`, `base_quota`
    (un-hedged), `hedge_buffer`, and the gate audit columns
    (`is_gated` / `gate_relaxed` / `is_unallocated`) when applicable.
  - **Weights**: pass a fixed `metrics=[MetricSpec, ...]` slate, or
    `suggest_config={"target_column": ..., "candidate_metrics": [...]}`
    with `weights_mode="global"` (suggest once on the full frame) or
    `"per_group"` (re-suggest per combination).
  - **Error handling**: `on_error="skip"` (default) warns per failed
    combination and continues, with a summary warning at the end;
    `"raise"` fails fast.
  - All other cascade options pass through: `gate_metrics`,
    `hedge_multiplier`, `gate_fallback`, `new_ic_*`, `brand_new_col`,
    `on_collision`.
  - Root detection: each combination's slice must resolve to exactly
    one `taxonomy[0]` value; otherwise a clear error suggests adding
    the root column to `group_keys`.

### Notes
- Every combination benefits from the full correctness stack:
  metric-value coercion (v0.6.1), duplicate-level healing and DAG
  validation (v0.6.0), never-gated root and per-depth reconciliation
  of the base layer (v0.5.0).

## [0.6.1] — 2026-07

Fixes [issue #3](https://github.com/shreyasrkarwa/Analytics/issues/3):
non-numeric metric/gate columns silently aggregated to 0. A gate column
holding `numpy.bool_` scalars or `"true"`/`"false"` strings read as 0
for every leaf, gating whole slices to $0 with no traceback — the
worst kind of bug to diagnose.

### Fixed
- **Metric values are coerced on ingest.** `from_dataframe` now runs
  every declared `metrics_cols` cell through a coercion layer:
  numpy scalars are unboxed (`np.int64` → int, `np.bool_` → bool),
  `"true"/"yes"/"false"/"no"` strings become bools, and numeric text
  is parsed tolerating thousands commas, currency prefixes, and
  trailing `%` (`"1,200"` → 1200.0, `"$500"` → 500.0, `"12.5%"` →
  12.5). Bools stay bools so boolean auto-detection (no
  zero-imputation) keeps working.
- **Uncoercible values warn and become missing.** Cells with no
  numeric interpretation (e.g. `"N/A - pending"`) are dropped like
  NaN, with one summary warning listing row/column examples — never
  stored where they'd read as 0.
- **The cascader can no longer skip values silently.** Both
  `_aggregate_node_metric` and the legacy `'_Attainment'` capacity
  path coerce at read time (covering graphs built manually via
  `add_node` with raw numpy/string values) and warn once per column
  when they encounter an uncoercible value.

### Notes
- No API changes. Results are identical for clean numeric data.
- The SQL-side workaround
  (`MAX(CASE WHEN flag THEN 1 ELSE 0 END)`) is no longer needed.

## [0.6.0] — 2026-07

Fixes [issue #1](https://github.com/shreyasrkarwa/Analytics/issues/1):
`RecursionError` when a hierarchy row holds the same value at two
adjacent levels. `from_dataframe` used to build the `parent -> child`
edge without checking `parent != child`, producing a self-loop (or a
cross-row cycle); the cascader's recursive aggregation then recursed
forever, crashing with a cryptic `RecursionError` deep inside networkx.

### Fixed
- **Self-loops can no longer be built from data.** `from_dataframe`
  resolves each row's path with a collision policy (below) so
  `team='T1', rep='T1'` never creates `T1 -> T1`.
- **Blank-string levels no longer collapse branches.** Cells that are
  empty/whitespace or the literal strings "nan"/"none"/"null" (common
  with `keep_default_na=False`) are now treated as missing — like real
  NaN — instead of becoming a shared node named "nan". `'NA'` (North
  America) is deliberately still treated as data.
- **Cyclic graphs fail fast with a clear error.** `cascade_quota`
  validates the graph up front and the recursive helpers carry a
  recursion-stack guard, so any cycle raises
  `HierarchyValidationError` naming the cycle instead of a
  `RecursionError`. Diamond-shaped DAGs (a node reachable via two
  branches) remain allowed, matching previous aggregation behavior.

### Added
- **`on_collision` parameter on `from_dataframe`** —
  `"suffix"` (default) | `"skip"` | `"error"`.
  - `"suffix"`: the deeper duplicate is deterministically renamed to
    `<value>__<level_column>` (e.g. `T1__node_5_rep_no`), the edge is
    kept, and one summary warning lists examples. Non-colliding node
    ids are unchanged.
  - `"skip"`: the duplicate level is dropped from that row's path
    (jagged-hierarchy semantics).
  - `"error"`: raise immediately, naming the row and value.
- **`SalesHierarchy.validate()`** — asserts the graph is a DAG;
  raises `HierarchyValidationError` naming any self-loop or cycle.
  Called automatically at the end of `from_dataframe`; call it
  manually after building via `add_edge()`.
- **`HierarchyValidationError`** exported at package top level.

### Notes
- Backward compatible for clean data: node ids, edges, and cascade
  results are unchanged when no row has duplicate/blank levels.

## [0.5.0] — 2026-07

Fixes [issue #12](https://github.com/shreyasrkarwa/Analytics/issues/12):
fully-gated subtrees no longer strand the target. Previously, when a
gate zeroed an entire slice (e.g., a Migration combo where NO rep had
DC entitlement), `cascade_quota` set the root to $0 and the target was
silently lost — or, with the common depth-0 patch-back workaround,
depth 1+ summed short of the target (the observed $52,869.30
Enterprise_AMER depth-1 shortfall).

### Fixed
- **Fully-gated subtrees redistribute instead of stranding.** A gated
  subtree's share flows to its nearest non-gated siblings (as before,
  via upward gate rollup); when EVERY child of a funded node is gated —
  including a fully-gated root — the gate is relaxed at that level as a
  last resort so the target still reaches ICs. The base (un-hedged)
  cascade now sums to the macro target at **every** depth. No silent
  target loss.
- **The root is never gated to $0** in any mode; it always carries the
  macro target.

### Added
- **`gate_fallback` parameter on `cascade_quota`** —
  `"redistribute"` (new default) | `"strand_at_root"` | `"error"`.
  - `"strand_at_root"`: children stay $0; the undistributable amount
    remains on the deepest non-gated ancestor and is reported via
    `cascader.unallocated` (total) and `cascader.unallocated_nodes`
    (per node), plus an `is_unallocated` column in
    `quotas_to_dataframe`. Explicit opt-in for "don't force money into
    gated territory" — depth sums below will not reconcile.
  - `"error"`: raises the new `GateAllocationError` (exported at
    package top level).
- **`cascader.base_quotas`** — every `cascade_quota` call also computes
  the un-hedged cascade (hedge_multiplier=1.0) in the same call, so
  `hedged = base × hedge^depth` decomposes without a second run.
- **`unhedged_quotas="auto"` in `quotas_to_dataframe`** — uses
  `base_quotas` for the `unhedged_quota` / `hedge_buffer` /
  `overassignment_pct` audit columns; no second cascade needed.
- **`cascader.reconciliation_report(quotas, target=None, tolerance=0.01,
  strict=False)`** — per-depth reconciliation DataFrame
  (`depth, n_nodes, total_quota, target, delta, reconciles`);
  `strict=True` raises `AssertionError` listing every non-reconciling
  depth.
- **`cascader.gate_relaxed_nodes`** + **`gate_relaxed` column** in
  `quotas_to_dataframe` — nodes that received quota despite being gated
  because every sibling was also gated (redistribute last resort), so
  the fallback is always visible.
- **`__version__`** exposed at package top level.

### Changed
- **Behavior change (intentional):** cascades whose gate previously
  zeroed an entire tree returned all-$0; they now distribute the target
  by blend weights (default `gate_fallback="redistribute"`). Restore
  something like the old behavior — without the silent loss — via
  `gate_fallback="strand_at_root"`.
- Partial gating (some siblings pass) is unchanged: gated nodes still
  get $0 and siblings absorb their share.

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
