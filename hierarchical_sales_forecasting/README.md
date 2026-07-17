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

### What's New in v0.36.0 — stakeholder edits, spelled ([#57](https://github.com/shreyasrkarwa/Analytics/issues/57))

```python
# "cut 75% of these reps' Cloud quota, move it 60/40 to those two"
edited, report = reallocate(quotas_long, sources=['DACH3_r1', 'DACH3_r2'], fraction=0.75,
                            weights={'r_a': 0.6, 'r_b': 0.4},
                            scope={'base_product_r4f': 'Cloud'})

# "re-split this team's Migration by dc_seats"
edited, report = resplit_by_metric(quotas_long, 'CENTRAL6-MIGRATION', 'dc_seats',
                                   scope={'st1_sales_type': 'Migration'})
```

Both are thin sugar over the pin engine — pinned equivalent to their hand-built compositions, parent-conserving, `reconcile()`-clean, freeze-aware, with the family's roles/`exact` reports. `redistribute(x)` is now provably `reallocate([x], fraction=1.0)`. One deliberate semantic: `resplit_by_metric` ignores `is_pinned` on the children (a re-split is an overwrite — `freeze_nodes` is the opt-out). The per-(product×quarter×sales_type) scaling loops in §3b/§3c/§3d each collapse to a call.

### What's New in v0.35.0 — identities enforced, overshoots surfaced ([#54](https://github.com/shreyasrkarwa/Analytics/issues/54), [#55](https://github.com/shreyasrkarwa/Analytics/issues/55))

`enforce_identities()` is `reconcile()`'s fixing twin: top-down, every parent's base is the hard budget — pinned children hold, free children stretch, and overshooting pins scale down proportionally (`'scale_pins'`), raise (`'error'`), or stay (`'allow'`). Hedged identities restore themselves via per-row ratios (no `hedge=` needed — proven by requiring `reconcile()` to pass in the tests), pins are never inflated implicitly, and clean frames come back bit-identical. And `apply_pins` gained the missing *cross-pin* check: individually-valid pins that collectively break a parent's slice — which v0.29.0 was accidentally silencing as "intentional" — now always land in `attrs['overshoot_report']` per (parent, combo), warn once under the default `'allow'`, and can be auto-fitted with `on_overshoot='scale_pins'` (scaled pins honestly flagged in the feasibility report).

### What's New in v0.34.0 — `cascade_levels`, fully auditable ([#56](https://github.com/shreyasrkarwa/Analytics/issues/56))

The level-by-level driver was computing `weights_long`, `combo_report` and `id_map` for every transition — and throwing all three away. Its outputs now carry them on `attrs` as records tagged with `transition` + `level`: `pd.DataFrame(res.attrs['weights_long'])` is the authoritative all-transitions weights table (with v0.33.0's per-row policy provenance included), `combo_report` covers every (transition × parent) including skip reasons, and `id_map` maps collision renames per transition. Full parity with `cascade_many` — the hand-built `all_weights` cell deletes.

### What's New in v0.33.0 — callables see the whole cascade identity ([#51](https://github.com/shreyasrkarwa/Analytics/issues/51), [#52](https://github.com/shreyasrkarwa/Analytics/issues/52), [#53](https://github.com/shreyasrkarwa/Analytics/issues/53))

The nastiest bug of this cycle, confirmed and fixed: `metrics`/`gate_metrics` callables only ever saw the group keys, so per-sub-target routing ("Migration → dc_seats") silently fell through to the wrong branch — plausible-looking, money-wrong output. Callables are now evaluated **per target row** with `{**group_keys, **sub_target_columns}` (hierarchies still built once per combination; `None` still falls through; errors still skip combos atomically), and `cascade_levels` threads the root key so transitions see the full identity too. That makes #53's ask the already-documented one-liner — `metrics=lambda g: BY_TYPE.get(g['st1_sales_type'])` — pinned equivalent to manual split-and-concat. `weights_long` records gain sub-target tags when slates vary, and `combo_report` says `weights_source='mixed'` when a policy partially decides. And #52 (alleged upper-casing): disproven — the package never case-normalizes; values arrive verbatim, now guaranteed in the docs and pinned by a mixed-case test.

### What's New in v0.32.0 — pipeline health, finally batch-native ([#15](https://github.com/shreyasrkarwa/Analytics/issues/15))

The oldest open gap: `PipelineAdjuster` — risk bands, per-region thresholds, zero-sum IC redistribution — only spoke single-cascade. Now:

```python
out = adjust_many(quotas_long, ['Open_Pipeline', 'Late_Stage_Commit'],
                  mode='redistribute',
                  coverage_thresholds={'NA': {'healthy': 1.5, 'at_risk': 0.8}},
                  locked_nodes={'IC_007'})
```

Per cascade, `adjust_many` rebuilds the graph + quota dict from the frame (pipeline columns carried via `metadata_cols`) and drives the *real* `PipelineAdjuster` — nothing re-implemented, pinned by an equivalence-anchor test. Every row gains `pipeline` / `coverage_ratio` / thresholds / `risk_status`; redistribute mode rebalances ICs zero-sum per team on the cascaded layer, re-derives base from each row's own hedge ratio (so `reconcile()` stays clean — pinned), recomputes `share_of_parent`, and leaves a `quota_delta` audit trail. Managers never move.

### What's New in v0.31.1 — the family gate-bridge, as a recipe ([#27](https://github.com/shreyasrkarwa/Analytics/issues/27))

The Cloud↔DC entitlement bridge is deliberately a *recipe*, not an API (see Recipes below): the edition map is catalog knowledge that changes with your product line, and whether multi-counterpart entitlement sums or maxes is your call — the same reasoning that kept `grain`/`dedup_key` out in v0.19.1. The generic halves the package owns (per-combination conditional gates, gate modes, gating reports) already compose with a 6-line pandas bridge, now documented and pinned by a regression test so it can't rot.

### What's New in v0.31.0 — every sanitized id maps back, everywhere ([#18](https://github.com/shreyasrkarwa/Analytics/issues/18))

Collision renames (`on_collision='suffix'`) are for correctness; getting back to your source ids should never require `split('__')[0]` surgery. Three batch-level holes fixed: `original_id` no longer goes NaN for clean combinations in a mixed batch (self-mapping now holds batch-wide), a new `original_parent` column maps the parent side back too (children of a renamed *manager* re-join cleanly), and `attrs['id_map']` carries the per-combination sanitized→original records. The `<id>__<level_column>` suffix format is documented as stable — and as something you should never parse.

### What's New in v0.30.0 — `reconcile()`: the post-run checklist, as one call ([#46](https://github.com/shreyasrkarwa/Analytics/issues/46))

```python
frame = reconcile(quotas_long, hedge=HedgeByDepth(from_leaves={1: 1.10, 2: 1.05}))
assert frame.ok.all()          # conservation at every parent + hedge identity at every node
```

Every hand-written scratch cell — d0==d1, d2==base×1.05, d3==base×1.155, per-parent sums — becomes a tidy frame of `conservation` and `hedge_ratio` checks with `expected`/`actual`/`delta`/`ok` per node per cascade. `hedge` takes a float, an explicit `{depth: cum_ratio}` dict, or a `HedgeByDepth` spec (resolved with the engine's own `resolve()`, so the expectation can't drift — pinned by test). Runs clean on post-`apply_pins`/`redistribute`/`concentrate` frames too, which makes it the one-liner to run after every edit. One summary warning on failure; silence means verified.

### What's New in v0.29.0 — feasibility noise, silenced; real problems, one filter ([#45](https://github.com/shreyasrkarwa/Analytics/issues/45))

Pinning every sibling of a partition to sum exactly to the envelope is a plan, not a problem — but the report used to flag each pin `feasible=False` with a warning, identically to a genuine can't-fit. The report now says *why* money went unabsorbed: `unabsorbed_reason` is `'no_siblings'` (root pin — changing the total is the point), `'all_blocked'` (your own pins/excludes/freezes emptied the absorber set), or `'floors_at_zero'` (genuine). An `intentional` flag marks the first two, and only genuine floors warn — everything else is data, not noise. Triage collapses to `report[~report.intentional & ~report.feasible]`. `feasible` itself is unchanged for backward compatibility.

### What's New in v0.28.0 — the weights record, complete; shares, direct ([#50](https://github.com/shreyasrkarwa/Analytics/issues/50), [#38](https://github.com/shreyasrkarwa/Analytics/issues/38))

First, the part that already existed: `cascade_many`'s **second return value** (`weights_long`) has been the resolved, normalized per-group weights since v0.7.0 — built from the exact slates each cascade ran with, so re-invoking your callables to rebuild it (and risking drift) was never necessary. What #50 rightly exposed: gates weren't in it, legacy default-attainment combos were silently absent, and it didn't say *how* each slate was chosen. Now every combo appears (legacy ones as an explicit `_Attainment` row), gate specs get `role='gate'` rows with threshold/mode, and each row carries `weights_source` + a per-metric `degenerate` fallback flag.

Second, #38's per-node question: every output now has a **`share_of_parent`** column — the effective base-layer share each node received at its sibling split (root = 1.0, sums to 1 per sibling group, gated nodes 0). "Why did this node get this share?" is now two columns: compare `share_of_parent` against `<metric>_subtree / parent's <metric>_subtree` (v0.26.0), with `weights_long` explaining the blend. No more dividing quota shares by seat shares in a scratch cell.

### What's New in v0.27.0 — auditable batch runs ([#20](https://github.com/shreyasrkarwa/Analytics/issues/20), [#19](https://github.com/shreyasrkarwa/Analytics/issues/19))

Every `cascade_many` output now carries `attrs['combo_report']` — one record per combination: `skipped` + `reason`, `targets_matched`, `rows_produced`, `n_gated_nodes`, `gate_relaxed`, `unallocated_total`, `weights_source` (fixed / policy / suggested_global / suggested_per_group / default_attainment), `direction_mismatches`, `degenerate_fallback`. `pd.DataFrame(q.attrs['combo_report'])` and the batch is auditable at a glance — assembled from state the loop always had and used to discard. And the per_group direction-mismatch warning flood is gone: unless you explicitly set `warn_on_direction_mismatch`, you get **one** batch-level summary ("`'kw'` in 7/12 combinations") with per-combo detail in the report; explicit `True`/`False` keep the old behaviors, and the report column is populated regardless.

### What's New in v0.26.0 — subtree metric rollups: the "why" column ([#17](https://github.com/shreyasrkarwa/Analytics/issues/17), [#49](https://github.com/shreyasrkarwa/Analytics/issues/49))

Consumers kept hand-rolling ~30-line rollups to see *why* a node got its quota. Now:

```python
out = rollup_metrics(quotas_long, ['knowledge_workers', 'cloud_seats'])   # <metric>_subtree per node
out['coverage'] = out['pipeline_subtree'] / out['base_quota']              # #17's use case, one line

quotas, _ = cascade_many(..., metadata_cols=['knowledge_workers'], attach_metrics=True)  # #49's flag
```

Aggregation runs over descendant *leaves* (so `agg='mean'`/`'max'` are leaf aggregates, never means of means), each cascade is isolated via the `cascade_row_keys` stamp, carried columns stay leaf-grain, and wrong keys raise (the #40 guard) instead of corrupting. Bonus fix the guard caught: `cascade_levels` outputs now inherit root key columns at every depth and carry the stamp, so rollups (and pins) work on them without `row_keys`.

### What's New in v0.25.1 — #29 and #44 were already fixed; now it's provable ([#29](https://github.com/shreyasrkarwa/Analytics/issues/29), [#44](https://github.com/shreyasrkarwa/Analytics/issues/44))

Both asks are `concentrate()` (v0.24.0), now pinned by regression tests and named in its docs: "route 100% of a parent's quota to a single child" — the survivor gets the full pool with zero leak (the leak in #29 was the pre-pins-era engine) and carries the hedged pool automatically; and "concentrate a group onto one team" — a hyphenated destination like `CENTRAL6-MIGRATION` is detected as a manager from the graph, so its reps carry the money and the team-vs-rep footgun can't happen. No behavior changes.

### What's New in v0.25.0 — pins that match nothing no longer abort the batch ([issue #48](https://github.com/shreyasrkarwa/Analytics/issues/48))

In a batch run some (node, scope) combinations legitimately have no rows — a product a region doesn't sell, a team with no reps in a quarter. `apply_pins(..., on_missing='skip')` now drops such pins into the feasibility report instead of raising (new columns: `skipped`, `reason='node_absent'|'empty_scope'`); `'warn'` adds one summary warning naming them; the default stays `'error'`. Same philosophy as `cascade_many`'s `on_error` — dropped intent is data, not an abort. Skipped pins are dropped *entirely*: no protection-set membership, so their nodes still absorb and rescale normally (pinned by test — a vacuous pin leaves the frame bit-identical). Scope-column typos still raise regardless. Your `_pin_has_rows()` guard loop can go.

### What's New in v0.24.0 — `concentrate()`: the inverse of `redistribute` ([issue #47](https://github.com/shreyasrkarwa/Analytics/issues/47))

"All of CENTRAL's Migration lands on the CENTRAL6-MIGRATION team; every other CENTRAL team = 0" is one call:

```python
edited, report = concentrate(quotas_long, 'CENTRAL6-MIGRATION',
                             scope={'st1_sales_type': 'Migration'})
# or collapse only some siblings:
edited, report = concentrate(quotas_long, 'CENTRAL6-MIGRATION', from_nodes=['C1', 'C2'], ...)
```

Same construction as `redistribute` — thin sugar that writes the pins (destination → group total, sources → $0) — so source subtrees zero and the destination subtree grows at every depth with its internal mix preserved, the parent conserves, hedge ratios hold per row, other scopes stay untouched. No per-rep pins, no `exclude`, no hand-computed group total (the ~40-line workaround collapses to one line). The report tags each node destination/source/bystander with an `exact` flag; unlisted siblings are verified to stay at baseline to the cent. One ordering subtlety handled internally: sources zero *before* the destination pin lands, so a bystander buffer inflates-then-sheds and never transits $0 (which would have equal-split its reps).

### What's New in v0.23.0 — `redistribute()`: move a region's quota to its siblings ([issue #43](https://github.com/shreyasrkarwa/Analytics/issues/43))

"MM_AMER_EAST gets zero Migration; move it to CENTRAL/WEST proportionally" is now one call:

```python
edited, report = redistribute(quotas_long, 'EAST', scope={'st1_sales_type': 'Migration'})
# custom split:
edited, report = redistribute(quotas_long, 'EAST', weights={'CENTRAL': .7, 'WEST': .3},
                              scope={'st1_sales_type': 'Migration'})
```

It's thin sugar over `apply_pins` — it writes the pins for you (source → $0, destinations → baseline + share × source), so everything the pin engine guarantees comes along: the source subtree zeroes and destination subtrees grow *at every depth*, parents conserve, each row's hedged value re-derives from its own hedge ratio, other scopes stay untouched, frozen nodes never move. `weights=` takes `'proportional'` (default), `'equal'`, or a dict; recipients must be siblings (cross-parent moves are `route_targets`' job — the error says so). The returned report tags every involved node source/destination/bystander with an `exact` flag — bystanders are *verified* to land back at baseline to the cent, not assumed. The scoped-region-Pins workaround still works; this just spells it.

### What's New in v0.22.1 — remainder pins are just pin composition ([issue #42](https://github.com/shreyasrkarwa/Analytics/issues/42))

"Pin some children exactly; the unpinned sibling takes the rest" needs no special mode — it's a pin list:

```python
# remainder to unpinned siblings, parent conserved as-is
apply_pins(quotas, [Pin('T1', 500_000), Pin('T2', 1_500_000)])

# the requested Pin(parent, total, children={...}, remainder='auto')
apply_pins(quotas, [Pin('EMEA', 5_000_000), Pin('T1', 500_000), Pin('T2', 1_500_000)])
```

This is exact, not approximate: pinned nodes never absorb for other pins, proportional absorption preserves the unpinned siblings' ratios (the issue #37 identity), and depth-ordered application lands the parent pin first regardless of list order. The composition became reliable with the v0.20.0/v0.22.0 pin fixes — the remainder-team + envelope + `exclude` workaround is obsolete. If pinned children exceed the parent's pin, free siblings floor at $0 and the gap is reported (`subtree_shortfall` + unabsorbed), never hidden. All three patterns are pinned by regression tests.

### What's New in v0.22.0 — pin list order never matters ([issue #41](https://github.com/shreyasrkarwa/Analytics/issues/41))

The reported symptom — a leaf pin overwritten by a later manager-pin rescale — was already fixed by v0.20.0's protection-aware rescale (verified across all 24 orderings of the filer's scenario: every pin held, every parent conserved). What remained: *absorber* rows could still differ by list order. `apply_pins` now applies pins in canonical depth order — shallowest pinned node first, managers before leaves, stable within a depth — so the entire output frame is identical for any ordering of `pins`, and leaf-pin allocations are computed against post-manager-rescale baselines. The feasibility report stays in the caller's pin order. The managers-before-leaves workaround is now simply what the package does internally.

### What's New in v0.21.0 — pins find the cascade key themselves ([issue #40](https://github.com/shreyasrkarwa/Analytics/issues/40))

Confirmed as reported — and worse: with per-node columns (e.g. `metadata_cols`) in the frame, `apply_pins` without `row_keys` inferred a poisoned cascade key, so manager pins silently became `pin_type='leaf'` (descendants never followed the pin) *and* sibling absorption mis-grouped. Two fixes, matching both halves of the issue's Ask: `cascade_many` outputs now carry `.attrs['cascade_row_keys']` (exact group keys + sub-target columns), which `apply_pins` uses automatically — no `row_keys` needed on batch outputs; and a hard orphan guard replaces silent corruption — if any row's parent exists in the frame but not under the same cascade key (impossible when keys are right), `apply_pins` raises, naming the poison columns and suggesting the corrected `row_keys`. The explicit-`row_keys` workaround keeps working; wrong explicit keys now error too. NaN key values are normalized so they no longer split cascades.

### What's New in v0.20.0 — protection-aware pin rescaling ([issue #39](https://github.com/shreyasrkarwa/Analytics/issues/39))

The reported symptom — hedged-layer manager pins not propagating the cross-level buffer — does not reproduce: `apply_pins` scales descendant *base* values and re-derives cascaded from each row's own hedge *ratio*, so a `basis='cascaded'` team pin under `HedgeByDepth` already rolls reps up to pinned × cross-level hedge (now pinned by test). The investigation did surface three real defects in the subtree rescale, fixed here: a later manager pin used to trample an earlier pin on one of its descendants (pin order mattered — almost certainly the wrong rollup actually observed); `freeze_nodes` inside a pinned or absorbing subtree were scaled despite the "never modified" contract; and `exclude` didn't protect descendants of a manager pin.

The rescale is now protection-aware: descendants pinned by another pin, frozen, or excluded keep their values while free siblings stretch to fill — proportional to base, exactly the old rescale when nothing is protected. Absorption is weighted by free capacity, so absorbers never push deltas onto protected rows. If protected values alone exceed a pin, free rows floor at $0 and the feasibility report's new `subtree_shortfall` column says so, with a warning.

### What's New in v0.19.2 — carrying metric values into quotas_long_df ([issue #16](https://github.com/shreyasrkarwa/Analytics/issues/16))

No new code — new clarity. The capability #16 requested (`carry_metric_cols`) has existed since v0.8.0 under a name that hid it: **`metadata_cols` carries ANY hierarchy column onto leaf rows, metric values included.** List `knowledge_workers` in `metadata_cols` and every leaf row of quotas_long_df carries its value — even while that same column drives the cascade via an explicit `MetricSpec` (carried columns are excluded from *auto* ingestion only; cascade numbers are identical either way, now pinned by test). No re-join against the source frame needed.

The related footgun is **key identity**: hierarchy leaves are identified by the deepest taxonomy column (plus group_keys present in `hierarchy_df`); cascade rows are identified by group_keys + sub-target columns. Columns that live only in the target frame (a sales type, a fiscal quarter) are not valid join keys for leaf-grain data — joining on them KeyErrors. Documented in `cascade_many`.

### What's New in v0.19.1 — metric-grain guardrails ([issue #36](https://github.com/shreyasrkarwa/Analytics/issues/36))

Metric columns must be at **leaf grain** — an account- or region-level value repeated onto every leaf row double-counts under sum rollups and collapses sibling shares to equal splits, silently. The cascader now detects the signature (values identical across ≥90% of leaf-sibling groups) and warns, naming the column, for both cascade metrics and gates. Booleans and all-zero cases are exempt, it's once-per-metric, and it's warning-only. Auto-deduping via a `grain`/`dedup_key` was deliberately not added — the correct collapse (e.g. MAX per account → SUM per rep) is source-data semantics that belongs in your feed query; see the new "Metric Grain" section under Key Concepts.

### What's New in v0.19.0 — mixed weight strategies per combination ([issue #35](https://github.com/shreyasrkarwa/Analytics/issues/35))

`cascade_many`'s `metrics=` now accepts a callable evaluated per combination — fix some groups' slates verbatim while others use suggested weights, in one call:

```python
DC_ONLY = [MetricSpec('dc_seats', direction='proportional', weight=1.0)]
quotas, weights = cascade_many(
    hierarchy_df, target_df, group_keys=['st1_sales_type', 'regional'], ...,
    metrics=lambda g: DC_ONLY if g['st1_sales_type'] == 'Migration' else None,
    suggest_config=dict(target_column=..., candidate_metrics=[...]),
    weights_mode='per_group',
)   # Migration: guaranteed pure DC-seat share · everything else: suggested
```

`None` without a `suggest_config` falls to the legacy path; bad policies flow through `on_error`; `weights_long` shows what each combination actually used; equivalence with the two-call workaround is pinned by test. (For single cascades nothing changed — fixed specs were always honored verbatim; see v0.17.0.)

### What's New in v0.18.1 — the pin absorption policy, stated and pinned ([issue #37](https://github.com/shreyasrkarwa/Analytics/issues/37))

Docs release. When you pin children (`new_ic_overrides`), the remainder flows to the **non-pinned siblings proportional to their baseline (un-pinned) cascade** — that's not a mode to request, it's an algebraic identity of renormalized shares, now stated in the docstring and locked by a regression test (the DACH scenario: $50M with two teams pinned on base totals; non-pinned siblings match baseline-proportional to the penny, base conserves at every depth, hedged derives per node). The rest of #37's proposed API already existed: `pins=` → `new_ic_overrides` (any level, v0.13.0), `pin_basis` → `override_basis` (default `'base'`), cross-combo totals → `apply_pins` (v0.16.0).

### What's New in v0.18.0 — level-by-level cascading ([issue #30](https://github.com/shreyasrkarwa/Analytics/issues/30))

When each level needs different behavior — split regions by knowledge workers but territories by seats, hedge only the front line, pin at one level — `cascade_levels` chains one-level cascades with per-transition kwargs, threading each level's base output into the next level's targets:

```python
result = cascade_levels(
    hierarchy_df, regional_targets,
    taxonomy=['regional', 'node_3_region', 'node_4_team', 'node_5_rep_no'],
    target_col='nn_acv_target',
    level_kwargs=[
        dict(metrics=KW_SPECS),                                  # d0 -> d1
        dict(metrics=KW_SPECS, hedge_multiplier=1.05),           # d1 -> d2
        dict(metrics=SEAT_SPECS, gate_metrics=DC_GATE),          # d2 -> d3
    ])
```

Base conservation holds per parent at every level, each transition hedges only its own step, quarters/keys thread through, dropped targets carry a `level` tag — and with uniform kwargs the result equals a single full-tree `cascade_many` (pinned test). The one-level primitive itself needs no new API: `cascade_many(df, targets, group_keys=[parent_col], taxonomy=[parent_col, child_col])`.

### What's New in v0.17.0 — the proportional-split front door ([issue #34](https://github.com/shreyasrkarwa/Analytics/issues/34))

Deterministic capacity-based allocation now has a named entry point: `cascader.cascade_proportional(root, target, metric='dc_seats')` — "this team holds 30% of the DC seats → 30% of the quota," no correlation, no target column, any slice size. Blends: `metrics={'dc_seats': 1.0, 'cloud_seats': 0.5}`. All cascade options (gates, `HedgeByDepth`, pins…) pass through. To be clear about what this is: sugar over the behavior that was always the default — fixed-weight `MetricSpec`s are used as-is, and `suggest_weights` has never been a required stage. See "Deterministic proportional splits" under Key Concepts.

### What's New in v0.16.0 — pin exact totals across cascades ([#22](https://github.com/shreyasrkarwa/Analytics/issues/22) · [#31](https://github.com/shreyasrkarwa/Analytics/issues/31) · [#24](https://github.com/shreyasrkarwa/Analytics/issues/24))

"This territory carries exactly $2.6M in total across all products and quarters" is now one call on the batch output:

```python
from b2b_revenue_forecasting import Pin, apply_pins

edited, feasibility = apply_pins(
    quotas_long,
    pins=[Pin('AMER_EAST_East1_4', 2_600_000),           # leaf, total across combos
          Pin('LATAM', 10_500_000),                       # manager subtree total
          Pin('UKI1_1', 400_000, scope={'fiscal_quarter': 1},
              exclude=['UKI1_3'])],                       # Q1 only, protect UKI1_3
    freeze_nodes=['East1_1'],                             # never absorbs, never changes
)
```

The pinned node keeps its baseline mix across combos; siblings absorb each cascade's delta proportionally (subtrees rescale, parents conserve, floors at $0 — never negative); infeasible pins are flagged in the feasibility report with the unabsorbed amount; `is_pinned`/`pin_type` mark provenance; and everything runs on the base layer with hedged values derived from each row's own ratio. For per-cascade pins, hedging basis, and post-edit re-hedging, see v0.13.0's `new_ic_overrides`/`override_basis`/`rehedge`.

### What's New in v0.15.0 — conditional gating per combination ([issue #14](https://github.com/shreyasrkarwa/Analytics/issues/14))

`cascade_many`'s `gate_metrics` now accepts a callable evaluated per combination with its group-key dict — so a DC-seat gate can apply to Migration only, never Expansion, in one call:

```python
DC_GATE = [MetricSpec('dc_seats', columns=['dc_seats'])]
quotas, weights = cascade_many(
    hierarchy_df, target_df, group_keys=['st1_sales_type', 'regional'], ...,
    gate_metrics=lambda g: DC_GATE if g['st1_sales_type'] == 'Migration' else None,
)
# mapping style: gate_metrics=lambda g: BY_TYPE.get(g['st1_sales_type'])
```

Exactly equivalent to the old split-and-concat workaround (pinned by test), with failing policies handled by the `on_error`/dropped-targets machinery. Also fixed: `attrs['dropped_targets']` is now stored as records so `pd.concat` on cascade outputs works again.

### What's New in v0.14.0 — no target left behind ([#26](https://github.com/shreyasrkarwa/Analytics/issues/26) · [#25](https://github.com/shreyasrkarwa/Analytics/issues/25) · [#32](https://github.com/shreyasrkarwa/Analytics/issues/32))

Targets with no matching hierarchy branch (Government + EMEA with no Government subtree) are now first-class: `cascade_many(return_dropped=True)` returns them as a frame with a `reason` column (also always on `quotas_long.attrs['dropped_targets']`), and the new `route_targets()` places that money on named recipients anywhere in the tree:

```python
quotas, weights, dropped = cascade_many(..., return_dropped=True)
routed = route_targets(
    dropped, quotas,
    recipients=['UKI1_2', 'UKI2_1', 'NORD1_3'],   # named Enterprise_EMEA reps
    target_col='nn_acv_target',
    recipient_keys={'regional': 'Enterprise_EMEA'},
    split='base_quota',                            # proportional to capacity
)
full_plan = pd.concat([quotas, routed], ignore_index=True)
```

Routing happens on the base layer, hedged values derive from each recipient's own ratio, ancestor rollups keep every depth reconciled, and routed rows carry the original segment tags plus `routed=True`. Conditional exclusions ("this rep never carries Cloud") are just two calls with different filters.

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

## 🧩 Recipes

### Family / edition gate-bridge (issue #27)

"A Migration DC-seat gate must let a Cloud edition inherit its DC counterpart's entitlement per rep." The mapping is yours; the gate is ours:

```python
FAMILY = {'Jira Cloud': ['Jira DC'],
          'Teamwork Collection': ['Jira DC', 'Confluence DC']}   # multi-counterpart is fine

seats = hierarchy_df.set_index(['rep', 'product'])['dc_seats']
hierarchy_df['family_dc_seats'] = hierarchy_df.apply(
    lambda r: sum(seats.get((r['rep'], cp), 0)        # sum vs max: YOUR catalog decision
                  for cp in FAMILY.get(r['product'], [])), axis=1)

quotas, weights = cascade_many(
    hierarchy_df, targets, group_keys=['product'], ...,
    gate_metrics=lambda g: ([MetricSpec('family_dc_seats',
                                        columns=['family_dc_seats'],
                                        gate_threshold=1.0)]
                            if g['product'] in FAMILY else None))
```

Reps inherit entitlement through any counterpart; combos outside the map stay ungated; all gate modes / fallbacks / `gating_report()` apply as usual. Pinned by `tests/test_family_bridge_recipe.py`.

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

### Metric Grain

Metric columns must be at **leaf grain**: one value describing that rep/territory alone. Non-leaf values are always computed as leaf-sums, so an ancestor-level number repeated onto child rows (an account's seats copied to every product row, a region's seats copied to every team) double-counts on the way up and makes siblings identical — collapsing their shares to an equal split. Resolve grain in the feed query (e.g. `MAX per (rep, account)` then `SUM per rep`); since v0.19.1 the cascader warns when a metric looks repeated from a coarser grain.

### Deterministic Proportional Splits (No Statistics)

The most common allocation — "split the target proportional to a metric" — needs no correlation and no `suggest_weights`. Fixed-weight `MetricSpec`s passed to `cascade_quota(metrics=...)` are used exactly as given; the suggester is an *optional* helper for when you want data-driven weight magnitudes. The one-liner (v0.17.0):

```python
quotas = cascader.cascade_proportional('Enterprise_EMEA', 1_000_000, metric='dc_seats')
# 30% of the DC seats -> 30% of the quota. Blend: metrics={'dc_seats': 1.0, 'cloud_seats': 0.5}
```

Deterministic, explainable, and correct at any slice size — including the tiny n≤2 slices where correlation is undefined.

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
