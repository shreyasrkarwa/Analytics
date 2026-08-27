# Changelog

All notable changes to `b2b_revenue_forecasting` are documented here.
This project loosely follows [Semantic Versioning](https://semver.org/).

## [0.43.0] — 2026-08

Closes [issue #66](https://github.com/shreyasrkarwa/Analytics/issues/66)
— headline corrected, a worse behavior found underneath it. "Money
silently disappears when a sibling set has zero metric" is DISPROVEN
with receipts: the engine equal-splits such sets, depth-0 lands at
exactly Σ targets, reconcile() is clean, and the whole-tree #6
warning fires (pinned as tests, including reconcile(targets=) from
v0.39.0 proving no dropped combos). What the review DID find: the
brand-new-IC carve-out (rule 'all_metrics_zero') misfires on
zero-metric slices — a rep with zero dc_seats for Migration is not a
new hire, yet got a FULL equal share, out-earning reps WITH signal;
and in cascade_levels the "leaves" of a transition are TEAMS, so a
zero-metric team took 50% of a region (500/500 where the data said
1M/0). Conserved, but badly misplaced — very plausibly the filer's
real ~0.69% mix drift, with 65% of their MM_AMER rows at zero seats.

### Fixed / Changed
- **cascade_levels: non-final transitions default new_ic_rule to
  'none'** (BREAKING where zero-metric intermediate nodes exist —
  deliberately: 1M/0 is what the data says, 500/500 was the bug).
  Opt back in per transition via level_kwargs.

### Added
- **`new_ic_rule='none'`**: explicit off-switch for the brand-new
  carve-out — for cascades where zero metric means "no business
  here", not "new hire". At leaf level the carve-out fires BEFORE
  the split policy, so combine 'none' with on_zero_metric when zero
  rows should get $0 (documented).
- **`on_zero_metric='equal'|'error'|'fallback'`** +
  **`metric_fallback=[...]`** on cascade_quota / cascade_many /
  cascade_levels: policy for a sibling set with NO signal. 'equal'
  is the (unchanged) default, now explicit; 'error' raises naming
  the parent (surfacing per combo via on_error/combo_report);
  'fallback' tries the chain in order — first column with signal
  wins, 'equal' terminates — the COALESCE the filer hand-wrote in
  SQL, but recorded.
- **combo_report accounting**: `zero_metric_parents`,
  `zero_metric_fallbacks`, `carveout_nodes` — the allocation basis
  is never invisible in the output again (the filer's valid
  complaint, fixed).
- Tests: `tests/test_zero_metric_policy.py`.

## [0.42.0] — 2026-08

Closes [issue #64](https://github.com/shreyasrkarwa/Analytics/issues/64)
— with corrections. The reported wall-clocks do not reproduce on this
code: at 107,730 rows (larger than the filer's 65K, same 4-depth /
~945-node / 114-combo shape) HEAD measures apply_pins with 400 pins
at ~24s (reported: abandoned after minutes), enforce_identities at
~5s (reported: 57 min), route_targets x3 at 0.03s (reported: 25+
min), scoped redistribute at ~4s (reported: 96s) — with linear
scaling verified at 1K/6K/24K/107K. And the "row order is
load-bearing" claim is DISPROVEN: six random shuffles x every mode
(default / scale_pins / rebalance / anchor='leaves' / apply_pins with
equal-dollar tied pins) give 0.0 delta, now pinned as tests. The
likely real mechanism behind the filer's $40K incident: pd.concat
DROPS df.attrs, so a hand-rolled stitch loses the cascade_row_keys
stamp and downstream calls fall back to key INFERENCE — which can
resolve different columns and genuinely change grouping.

### Added
- **`subset=` on apply_pins and enforce_identities** (#64): seed node
  ids; the run is confined to their full subtrees (plus, for pins,
  each pin's absorption domain — its parent's full subtree — added
  mechanically, so the slice can never truncate a family). The
  library does the slice-and-stitch itself: original index, row
  order, and attrs all preserved — the three traps of the hand-rolled
  version, structurally removed. A pin outside the seeds' subtrees
  raises. Output proven bit-identical to the full-frame run.
- **Perf**: cascade-key tuples now build from column arrays instead
  of df.apply(axis=1) (every keyed primitive benefits), and
  apply_pins resolves pin rows through one node_id group index
  instead of a full-frame boolean mask per pin.
- Tests: `tests/test_subset_fastpath.py` — equivalence, closure,
  guards, attrs survival, and the shuffle-invariance receipts.

## [0.41.0] — 2026-08

Closes [issue #63](https://github.com/shreyasrkarwa/Analytics/issues/63).
Overlapping pins on the same node — a grand total plus scoped
sales-type pins, or a total plus a scoped zero — were silently
resolved last-writer-wins, and the feasibility report made it WORSE:
each pin's achieved_total was captured at application time, so a
defeated pin still read feasible=True while the frame said otherwise
(reproduced: 900K pin, 750K delivered, four feasible=True rows).

### Added / Changed
- **`apply_pins(..., on_conflict='error'|'warn'|'allow'|
  'narrower_wins')`** (#63). A conflict is a same-node MATCHED-ROW-SET
  intersection, detected before anything applies. Default **'error'**
  (BREAKING): raises listing each family's pins — total, scope, row
  count, relation (subset / identical / partial overlap) — in the
  issue's requested format. Cross-node parent/child pins are NOT
  conflicts: that composition is deliberate (#39 protection, the #42
  remainder recipe) and its arithmetic is covered by the #55
  envelope check — the filer's case 2 was already loud on HEAD
  (overshoot_report + subtree_shortfall warnings, pinned as receipts).
- **`'narrower_wins'`**: scoped pins stand; the broadest pin
  constrains only its REMAINDER rows with total − Σ(narrower totals).
  Requires clean nesting (strict subsets of one broadest pin,
  mutually disjoint, single basis, non-negative remainder, no
  fully-covered unplaced remainder) — violations raise naming the
  numbers. Proven equal to the hand-built explicit-pin composition;
  the filer's case 3 now delivers the FULL total with the zeroed
  slice excluded, instead of silently under-delivering.
- **Final-frame audit (all modes)**: every pin's achieved_total is
  recomputed over its ORIGINAL row set on the final frame, and
  feasible goes False when it misses — a defeated pin can never
  report feasible=True again. New report columns: `conflict`
  (family description) and `adjusted_total` (narrower_wins
  remainder).
- Tests: `tests/test_pin_conflicts.py` — all three field cases,
  narrower_wins equivalence + five guards, #42 recipe untouched.

## [0.40.0] — 2026-08

Closes [issue #67](https://github.com/shreyasrkarwa/Analytics/issues/67)
and [issue #62](https://github.com/shreyasrkarwa/Analytics/issues/62)
(the latter disproven with receipts).

### Fixed
- **Zero-baseline hedge-ratio poisoning (#67)**: a row with base==0
  cannot supply a cascaded/base ratio; every editing primitive
  silently fell back to 1.0, so pinning a zeroed slice set
  base = cascaded (or cascaded = base), breaking the depth's hedge
  identity — and enforce_identities(anchor='leaves') then propagated
  the wrong base into the parent, an error three levels from its
  cause (reproduced: $37K on one team). Now the ratio is DERIVED at
  the shared choke point used by apply_pins, enforce_identities and
  resplit_by_metric (and therefore every helper built on them):
  explicit `hedge=` wins, else the mean ratio of same-(cascade,
  parent) siblings with base>0, else same-(cascade, depth) rows,
  else 1.0. One summary warning fires only when such a row actually
  RECEIVES money, naming the rows and the ratio source; gated 0/0
  rows that stay 0 are silent.

### Added
- **`apply_pins(..., hedge=float | {depth: cum_ratio} | HedgeByDepth)`**
  (#67): authoritative ratio source for zero-baseline rows, resolved
  exactly like reconcile (HedgeByDepth per cascade via the real
  resolve()). Never re-hedges rows with base>0. All four sources
  proven equivalent on depth-uniform hedges in tests.
- **#62 receipts pinned as tests**: unscoped `Pin(node, total)` was
  claimed to re-partition a node to a stale "baseline shape". It does
  not, and never did on this code: allocation is
  row × (total / current_total) — the uniform multiplier the issue
  requests as `distribute='scale'` — read from the CURRENT frame
  (70/30 → 770/330, their own "want" line). Their observed production
  drift reproduces exactly as OVERLAPPING-pin composition (#63): a
  grand-total pin listed before a scoped zero under-delivers by the
  removed slice. Mix preservation is now a pinned regression test so
  the claim can never silently become true.
- Tests: `tests/test_zero_baseline_ratio.py`.

## [0.39.0] — 2026-07

Closes [issue #60](https://github.com/shreyasrkarwa/Analytics/issues/60).
First, the correction the review owed the filer: `reconcile()`'s
`tolerance=0.05` is five CENTS — absolute dollars — not 5%. A 1% team
gap ($10,000 on a $1M team) is ~200,000x the default tolerance and has
always been flagged loudly (now pinned as a test, alongside the
$0.06-flagged / $0.04-passes receipts). The `atol` mode the issue asks
for is what shipped in v0.30.0. What the issue is RIGHT about: there
was no check that depth-0 equals the input plan — the one identity
internal checks cannot see, and precisely the invariant that survives
untouched when targets get dropped (#26) or edits legitimately float
the root (`anchor='leaves'`/`'rebalance'`, `route_targets(merge=True)`).

### Added
- **`reconcile(..., targets=<frame>, target_col=...)`** (#60): emits
  `check='target_total'` rows — depth-0 (root) base vs the INPUT
  targets, one row per cascade combo plus one global row. Target
  combos with NO quota rows appear as expected=amount / actual=0, so
  dropped-target drift is finally visible in the audit. Scalar
  `target_total=<float>` gives the global row only.
- **`check='depth_conservation'`** rows: the aggregate ledger view
  auditors hand-write ("d0 == d1 == d2"), per cascade and depth —
  exact in jagged trees because depth d compares against the sum of
  depth d-1 nodes that HAVE children (a leaf at depth 1 never
  false-alarms). Implied by per-parent conservation, but first-class
  so the audit reads like the ledger.
- **`on_fail='warn'|'raise'`**: 'raise' throws a ValueError naming the
  offending (check, node, combo, delta) rows — the "fails loudly"
  assert without hand-writing it. Default 'warn' unchanged.
- Docs now state plainly that `tolerance` is absolute dollars.
- Tests: `tests/test_conservation_audit.py` — the tolerance receipts,
  the floated-root drift caught with internals clean, dropped-target
  visibility, jagged-tree exactness, raise mode, guards.

## [0.38.0] — 2026-07

Closes [issue #61](https://github.com/shreyasrkarwa/Analytics/issues/61).
The `rollup=True` + concat pattern our own docstring recommends leaves
node-grain duplicates: every routed ancestor appears twice (once from
the tree, once from the routed overlay), so any group-by, audit, or
persist keyed on node_id sees two rows for ENTERPRISE_EMEA at depth 0.
The filer's 20-line groupby collapse — with its `routed = max(...)`
flag gymnastics — was the tell that the composition was missing.

### Added
- **`route_targets(..., merge=True)`** (#61): returns the FULL quotas
  frame with routed dollars FOLDED into the matching tree rows instead
  of the routed overlay. Each routed row first ADOPTS the
  recipient-side identity (the `recipient_keys` columns are
  overwritten with the recipient values — that is what makes a
  Government orphan land ON the Commercial rows it was routed to);
  rows are then matched on (node_id + cascade keys), resolved exactly
  as everywhere else (explicit `row_keys` → `attrs['cascade_row_keys']`
  → inference). Matched rows sum `base_quota`/`cascaded_quota`
  (cascaded via each node's own ratio, so `reconcile()` is clean on
  the merged frame by construction — rollup adds the same delta up
  the chain); unmatched rows are appended with a warning naming them.
- **Provenance survives the fold**: `routed=True` on every touched
  row, `routed_base_quota` holds the exact dollars added,
  `routed_from` records the orphan identity that was adopted away
  (e.g. `'segment=Government'`, accumulating across multiple orphans),
  and `attrs['route_report']` carries a per-routed-row fold record
  (node, keys, dollars, matched).
- **Derived columns recomputed** on touched rows: `hedge_buffer`,
  `overassignment_pct`, and `share_of_parent` (whole frame, same
  machinery as `adjust_many`).
- `merge=True` requires `rollup=True` (folding leaves alone would
  break the parent/child identities the fold exists to protect);
  `merge=False` (default) is byte-identical to the old behavior — the
  additive-overlay reading stays available when you WANT the orphan
  segment visible as separate rows.
- Tests: `tests/test_route_merge.py` — the footgun reproduced then
  folded away, equivalence with the filer's hand collapse, reconcile
  clean, provenance, multi-orphan accumulation, append path, guards.

## [0.37.0] — 2026-07

Closes [issue #58](https://github.com/shreyasrkarwa/Analytics/issues/58)
and [issue #59](https://github.com/shreyasrkarwa/Analytics/issues/59).
#59 is the important one, and we own it plainly: v0.35.0's
`on_overshoot='scale_pins'` was per-combo and one-directional, so
when aggregate pins were correct in total but combos deliberately
over/undershot (exactly what concentrate/resplit_by_metric compose
into), it removed the overshoot WITHOUT restoring the undershoot —
silent money loss with shipped APIs. Reproduced exactly, fixed, and
the failure shapes are now pinned as tests so they can't return.

### Added
- **`enforce_identities(..., anchor='root'|'leaves')`** (#58):
  'leaves' is the bottom-up dual — children stand, every parent is
  DERIVED as the exact sum of its children up to the root (which
  floats). Pins never touched; conservation by construction;
  on_overshoot ignored. The filer's 30-line rebuild, shipped.
- **`on_overshoot='rebalance'`** (#59): processed bottom-up ACROSS
  combos — when a node's aggregate matches its children's aggregate
  (within tolerance), its per-combo values FLOAT to the child sums.
  The aggregate is conserved, which means AGGREGATE PINS on that
  node stay exact (a Pin is an aggregate total by definition — the
  mode is philosophically aligned, not a compromise). Only a
  genuinely-off aggregate falls back to per-combo scale-down, now
  with proportional subtree propagation. The DACH3 shape: zero loss,
  reconcile clean.
- **Audit-grade scaling output** (#58's complaint): the report gains
  a `factor` column on scaled rows, and the warning names
  node@combo: factor pairs instead of "scaled N nodes".
- `apply_pins(on_overshoot='rebalance')` passthrough.
- `tests/test_anchor_rebalance.py` — 5 tests: the #59 netting shape
  (rebalance conserves AND reconciles; scale_pins' exact loss and
  allow's broken identities pinned as documentation), leaves-anchor
  float-up with exact pins, off-aggregate fallback with factors,
  clean no-ops for both anchors, apply_pins passthrough.

### Unchanged
- 'scale_pins'/'error'/'allow' semantics are untouched (back-compat;
  full suite green) — but scale_pins' docstring now carries an
  explicit CAUTION pointing at 'rebalance' for the per-combo
  concentration case.

## [0.36.1] — 2026-07

Docs/metadata only — no behavior changes.

### Docs
- README intro rewritten around the three design principles (two
  layers / one ratio contract; nothing silent; edits are algebra) and
  the hardening record (60 field-filed issues, 40+ test suites, CI on
  3.9–3.12).
- Features section expanded from the original five classes to the
  full v0.36 surface, grouped: Build & cascade, Batch planning, Plan
  editing (the pin engine), Explainability & validation, Forecast
  hygiene.
- PyPI metadata refreshed: richer one-line description, Development
  Status :: 4 - Beta, Python 3.11/3.12 classifiers, extended
  keywords.

## [0.36.0] — 2026-07

Closes [issue #57](https://github.com/shreyasrkarwa/Analytics/issues/57):
stakeholder edits — partial-percent moves among named siblings and
metric-based re-splits. Review verdict: both exactly expressible as
pin compositions today (verified: conserving, reconcile-clean), but
the boilerplate is real — baseline reads + hand math for fraction
moves, and children x cascades scoped pins for re-splits. Same
call as #43/#47: thin sugar over an engine that already guarantees
correctness.

### Added
- **`reallocate(quotas_long, sources, recipients=, fraction=1.0,
  weights=, scope=, freeze_nodes=, row_keys=)`** — the generalized
  sibling move: multiple sources, a fraction in (0, 1] (each source
  keeps 1 - fraction of its baseline), recipients split the moved
  total by 'proportional' / 'equal' / dict. "Cut 75% of these reps'
  Cloud quota, move it 60/40 to those two" is one call, pinned
  equivalent to the hand-built 4-pin composition.
  `redistribute(x)` == `reallocate([x], fraction=1.0)` (pinned).
- **`resplit_by_metric(quotas_long, node, metric, scope=,
  freeze_nodes=, row_keys=)`** — re-split a node's children per
  cascade proportional to their descendant-leaf sums of a carried
  metric column ("re-split this team's Migration by dc_seats").
  Frozen children hold; free children split the remaining budget by
  metric shares (equal at zero pool, warned); subtrees rescale;
  hedged re-derives per row; share_of_parent recomputed; reconcile()
  stays clean. Pinned equivalent to the children-x-cascades scoped
  pin composition. NOTE: deliberately does NOT honor is_pinned on
  children — a re-split is an overwrite; freeze_nodes is the opt-out.
- Both return (edited, report) in the family style (roles/exact for
  reallocate; per-(cascade, child) metric_share/old/new/exact for
  resplit).
- `tests/test_reallocate_resplit.py` — 5 tests: both equivalence
  anchors, the redistribute identity, equal+freeze+validation,
  frozen-hold re-split, metadata_cols error hints.

## [0.35.0] — 2026-07

Closes [issue #54](https://github.com/shreyasrkarwa/Analytics/issues/54)
and [issue #55](https://github.com/shreyasrkarwa/Analytics/issues/55).
The review found #55 was WORSE than filed: individually-valid pins
that collectively overshoot a parent's slice were classified
'all_blocked' + intentional=True by v0.29.0 and produced ZERO
warnings — a genuinely inconsistent frame, silently. That blind spot
is closed.

### Added
- **`enforce_identities(quotas_long, on_overshoot='scale_pins' |
  'error' | 'allow', tolerance=, freeze_nodes=, row_keys=)`** (#54) —
  reconcile() that FIXES: top-down per cascade, every parent's base
  is the hard budget; pinned children (is_pinned provenance +
  freeze_nodes) hold when the budget allows, free children scale to
  fill (equal split at $0 baseline), and on overshoot the pins scale
  down proportionally ('scale_pins'), raise ('error') or stay
  ('allow'). A scaled pin's subtree re-reconciles at the next depth
  automatically. Pins are NEVER scaled up implicitly (undershoot with
  no free children is left + reported). No hedge= parameter,
  deliberately: cascaded re-derives from each row's own ratio (#21),
  restoring hedged identities automatically — pinned by tests
  requiring reconcile() to pass afterward. share_of_parent
  recomputed; clean frames return bit-identical. Returns
  (edited, report) per broken (cascade, parent): gap_before, action,
  pins_scaled, resolved.
- **`apply_pins(..., on_overshoot='allow')`** (#55): after all pins
  land, every (cascade, parent) identity is checked and
  `edited.attrs['overshoot_report']` records the gaps per (parent,
  combo) — the cross-pin signal no per-pin row can see. 'allow'
  (default, backward compatible) additionally emits ONE summary
  warning when gaps exist; 'scale_pins' delegates to
  enforce_identities and recomputes the feasibility report
  (overshoot_scaled=True + updated achieved_total, feasible=False for
  scaled pins); 'error' raises naming the slices.
- `tests/test_enforce_identities.py` — 5 tests: the #55 repro
  surfaced + warned, scale_pins fits + reconciles + reports honestly,
  error policy, standalone enforce (free-rescale, frozen-hold, clean
  no-op), undershoot-left semantics.

## [0.34.0] — 2026-07

Closes [issue #56](https://github.com/shreyasrkarwa/Analytics/issues/56):
cascade_levels dropped every diagnostic its per-transition
cascade_many calls produced — weights_long was literally caught in
`_` and discarded one line after being computed. Pure assembly fix,
no new machinery (the #20 pattern).

### Added
- cascade_levels outputs now carry **`attrs['weights_long']`**,
  **`attrs['combo_report']`** and **`attrs['id_map']`** — the
  per-transition records from each underlying cascade_many call,
  tagged with `transition` (index) and `level` (parent column,
  matching the dropped_targets convention). Reconstruct with
  pd.DataFrame(result.attrs['weights_long']).
- v0.33.0's per-row provenance flows through automatically: a
  callable transition's weights records carry the sub-target columns
  and weights_source='policy'/'mixed'.
- Kept as attrs records (not new return values): the return signature
  stays non-combinatorial, and records are pd.concat-safe — the
  established discipline.
- `tests/test_levels_diagnostics.py` — 4 tests: per-transition
  slates + combo records tagged, callable-transition provenance,
  skipped-combo reasons + collision id_map, concat safety.

## [0.33.0] — 2026-07

Closes [issue #51](https://github.com/shreyasrkarwa/Analytics/issues/51)
(bug, confirmed), [issue #52](https://github.com/shreyasrkarwa/Analytics/issues/52)
(disproven, pinned) and [issue #53](https://github.com/shreyasrkarwa/Analytics/issues/53)
(satisfied by the #51 fix). #51 was the worst kind of bug — silently
money-wrong: metrics/gate callables received only the group keys, so
per-sub-target routing (Migration -> dc_seats) silently fell through
to the wrong branch and the output looked plausible. Reproduced
exactly before fixing.

### Fixed (#51)
- **Callables now receive the FULL cascade identity**: when
  `metrics`/`gate_metrics` is a callable, it is evaluated PER TARGET
  ROW with `{**group_keys, **sub_target_columns}`. The hierarchy is
  still built once per combination — "prepare once / cascade many" is
  intact; only slate resolution moved to row grain. A None return
  still falls through to the combo-level suggested/global/legacy
  slate, resolved lazily exactly as before.
- **Atomic on_error contract preserved**: row slates are resolved for
  ALL rows BEFORE any cascading, so a callable error on a later
  sub-target row still skips the whole combination with no partial
  frames (pinned by test).
- **cascade_levels threads the root taxonomy key forward**, so
  transition callables see {parent key, root key, sub-target columns}
  — previously just the bare parent column.
- Backward compatible: extra keys in `g` are purely additive;
  group-key-only callables produce bit-identical cascades (pinned).

### Changed
- `weights_long`: when per-row policies are in play, records are
  tagged with the sub-target columns — per-row provenance, no drift.
- `combo_report.weights_source`: `'mixed'` when a policy decided some
  rows and fell through for others.

### #52 — disproven, with receipts
- No case normalization exists anywhere in the package (and never
  did): group and sub-target values arrive at callables VERBATIM,
  now pinned by a mixed-case regression test ('MiGrAtIoN' in,
  'MiGrAtIoN' received). The 'MIGRATION' observed in the field came
  from the consumer's own upper-casing feed. The verbatim guarantee
  is now documented.

### #53 — no new parameter needed
- With full identity in `g`, the documented mapping pattern
  (`lambda g: BY_TYPE.get(g['st1_sales_type'])`) IS the requested
  `metrics_by=`; equivalence with manual split-and-concat is pinned.

### Added
- `tests/test_callable_identity.py` — 6 tests: the Migration/dc
  incident routed correctly + split-and-concat equivalence,
  cascade_levels identity contents, verbatim mixed case, per-row
  weights records + 'mixed' source, atomic skip, backward compat.

## [0.32.0] — 2026-07

Closes [issue #15](https://github.com/shreyasrkarwa/Analytics/issues/15)
— the oldest open gap (filed Jul 6): PipelineAdjuster required a
single SalesHierarchy + quota dict, so the pipeline-health feature
was unusable from the batch API the package steers everyone toward.
Of the issue's three proposals we shipped (a), the batch entry point;
(b) a pipeline_adjust= config would bloat cascade_many's signature
for a post-processing concern, and (c) returning internal
per-combination hierarchies would leak objects and break the
tidy-frame contract.

### Added
- **`adjust_many(quotas_long, pipeline_cols, mode='flag_only' |
  'redistribute', coverage_thresholds=, max_adjustment_pct=,
  locked_nodes=, row_keys=)`** — per cascade, the graph and quota
  dict PipelineAdjuster expects are REBUILT from the frame (parent
  links + carried pipeline columns) and the real class runs: risk
  bands, ancestor-inherited thresholds, zero-sum redistribution,
  rails, locks. No duplicated logic (the reconcile()/resolve()
  pattern), pinned by an equivalence-anchor test against a
  hand-driven PipelineAdjuster.
- Output columns on every row: pipeline, coverage_ratio,
  healthy_threshold, at_risk_threshold, risk_status; redistribute
  mode also adjusts cascaded_quota (zero-sum per team, PipelineAdjuster's
  contract), re-derives base_quota from each row's own hedge ratio
  (#21 — reconcile() stays fully clean afterward, pinned),
  recomputes share_of_parent, and adds a quota_delta audit column.
  Managers are never changed.
- pipeline_cols accepts a summed list (the pipeline_attr contract);
  missing columns error with the metadata_cols pointer (#16).
- `tests/test_adjust_many.py` — 5 tests: equivalence anchor
  (diagnose AND redistribute), flag_only no-op, redistribute
  invariants (zero-sum / 20% rail / managers / reconcile-clean /
  share recompute), locked + inherited thresholds, missing-column
  hint.

## [0.31.1] — 2026-07

Closes [issue #27](https://github.com/shreyasrkarwa/Analytics/issues/27)
as a documented, regression-pinned RECIPE rather than a new API —
deliberately (the #36 precedent). A ProductFamily/edition_map utility
would either hardcode one company's catalog (Jira Cloud<->Jira DC,
Teamwork -> Jira DC OR Confluence DC, HELA<->ELA — knowledge that
changes with the product line) or be a parameter-heavy wrapper around
one pandas lookup; and the genuinely ambiguous choices — WHICH
counterpart(s), and whether multi-counterpart entitlement sums or
maxes — are the consumer's to make. Source-data semantics stay in the
feed layer.

### Docs
- README "Recipes" section: the family gate-bridge — a 6-line
  consumer-owned bridge column feeding the v0.15.0 conditional-gate
  callable, multi-counterpart included. Everything the package DOES
  own (gate modes, fallbacks, gating_report, per-combo policies)
  applies unchanged.
- `gate_metrics` docstring points at the recipe.

### Added
- `tests/test_family_bridge_recipe.py` — the recipe pinned end to
  end: Cloud rep inherits its DC counterpart's entitlement, rep
  without any is gated at $0, multi-counterpart inheritance passes,
  non-family combos stay ungated.

## [0.31.0] — 2026-07

Closes [issue #18](https://github.com/shreyasrkarwa/Analytics/issues/18):
suffixed ids with no way back to originals. v0.8.0 (#7) built the
foundation (id_map, original_id, original-keyed joins); #18's review
found three real batch-level holes, all fixed. Guess-stripping
suffixes was never the answer — now it's officially obsolete.

### Fixed
- **The concat NaN hole**: in a batch where only SOME combinations had
  renames, clean combos' rows got original_id=NaN after concat instead
  of the documented "the id itself". cascade_many now normalizes both
  original_* columns batch-wide, so self-mapping holds everywhere.

### Added
- **`original_parent`** column (alongside original_id, emitted whenever
  any rename happened): the parent id mapped back through id_map — so
  children of a renamed MANAGER re-join to source data without string
  surgery.
- **`attrs['id_map']`** on cascade_many outputs: the per-combination
  {sanitized -> original} mapping as records
  ({...group keys, 'sanitized', 'original'}), attrs-concat safe.
- Docs: the suffix format ('<id>__<level_column>') is documented as
  stable — and explicitly documented as something you should never
  parse; use original_id / original_parent / attrs['id_map'].
- Clean runs stay uncluttered: no renames -> no original_* columns,
  empty attrs['id_map'].
- `tests/test_rename_mapping.py` — 4 tests: batch-wide self-mapping
  (the NaN hole), renamed-manager children map back, attrs records,
  single-cascade path, clean-run hygiene.

## [0.30.0] — 2026-07

Closes [issue #46](https://github.com/shreyasrkarwa/Analytics/issues/46):
after every run, consumers hand-wrote the same scratch-cell checks —
d0==d1 sums, d2==base x 1.05, d3==base x 1.155, per-parent
conservation. (These checks caught several regressions during OUR
development too — #39's trilogy. They deserved to be one call.)

### Added
- **`reconcile(quotas_long, hedge=None, tolerance=0.05,
  ratio_tolerance=1e-4, row_keys=None)`** — a tidy frame of checks:
  `conservation` (every parent's base vs its children's sum, per
  cascade, dollar tolerance) and `hedge_ratio` (each node's actual
  cascaded/base vs the expected CUMULATIVE ratio, relative tolerance;
  base<=0 rows skipped — gated nodes have no ratio). Columns: cascade
  keys, node_id, parent, depth, check, expected, actual, delta, ok.
  ONE summary warning when anything fails; silent when clean.
- `hedge` accepts a float (f**depth), an explicit {depth: cum_ratio}
  dict (the literal hand-written identity list), or a HedgeByDepth
  spec — resolved per cascade with the REAL HedgeByDepth.resolve() on
  a graph rebuilt from the frame's parent links, so the expectation
  cannot drift from the engine (pinned by an engine-anchor test).
- Post-edit frames reconcile clean: apply_pins/redistribute/
  concentrate preserve per-row ratios (#21) and conserve parents —
  now verifiable in one line after every edit.
- Cascade identity via attrs['cascade_row_keys'] + the #40 orphan
  guard, now shared machinery (`_cascade_key_context`) with
  rollup_metrics.
- `tests/test_reconcile.py` — 5 tests: engine anchor (the issue's
  exact 1.10/1.05 spec), float + dict forms, corruption caught
  (conservation + ratio flags, one warning), post-pins clean, gated
  rows skipped.

## [0.29.0] — 2026-07

Closes [issue #45](https://github.com/shreyasrkarwa/Analytics/issues/45):
intentional full-partition pins (every sibling pinned to sum exactly
to the envelope), root pins, and exclude-all removals flooded the
feasibility report with feasible=False + a warning each —
indistinguishable from genuine infeasibility, burying real problems.
Even our own test suite emitted 24+ of these for intentional root
pins.

### Added
- **`unabsorbed_reason`** column in the apply_pins report:
  'no_siblings' (nothing to absorb exists — root pin),
  'all_blocked' (the caller's own pins/excludes/freezes emptied the
  absorber set), or 'floors_at_zero' (free capacity existed but hit
  $0 — genuine). Worst-across-rows wins per pin; genuine trumps
  intentional.
- **`intentional`** column: True when unabsorbed money is entirely
  explained by the caller's construction (no_siblings/all_blocked,
  no subtree shortfall). Real problems are now one filter:
  `report[~report.intentional & ~report.feasible]`.

### Changed
- The "could not be absorbed" warning fires ONLY for
  'floors_at_zero'. Intentional cases land in the report silently —
  data, not noise (the same principle as #26/#48). The
  subtree_shortfall warning is unchanged (always genuine). `feasible`
  semantics are unchanged for backward compatibility — it still means
  "landed with full absorption".
- `tests/test_feasibility_intent.py` — 6 tests: full partition
  (conserved to the cent, silent), root pin, exclude-all, genuine
  floors (warns), frozen-mass shortfall stays genuine, columns on
  clean/skipped rows.

## [0.28.0] — 2026-07

Closes [issue #50](https://github.com/shreyasrkarwa/Analytics/issues/50)
and [issue #38](https://github.com/shreyasrkarwa/Analytics/issues/38).
Review verdict on #50: the requested export has existed since v0.7.0 —
cascade_many's SECOND return value (weights_long) is built from the
exact slates passed to each cascade, so there is nothing to re-derive
and nothing to drift. But the ask exposed three real holes in it, all
fixed here.

### Changed — weights_long is now the complete authoritative record
- **Gate slates included** (#50's "gated flag"): one row per gate spec
  per combo, role='gate', with gate_threshold and gate_mode; blend
  rows carry role='blend'.
- **Legacy combos no longer silently absent**: combinations on the
  default '_Attainment' path get an explicit provenance row
  (metric='_Attainment', weights_source='default_attainment').
- New columns: weights_source ('fixed' | 'policy' |
  'suggested_global' | 'suggested_per_group' | 'default_attainment')
  and per-metric `degenerate` (True where the v0.12.0 suggest fallback
  fired for that metric).

### Added — per-node shares (#38)
- **`share_of_parent`** column in quotas_to_dataframe (hence every
  single-cascade and batch output): the effective share the cascade
  applied at each node's sibling split, on the base layer. Root = 1.0,
  sums to 1 per sibling group, gated nodes show 0. Combined with
  v0.26.0's `<metric>_subtree` columns, "why did this node get this
  share?" is a two-column comparison — no more reverse-engineering
  quota_share / seat_share by hand.
- `tests/test_weights_explainability.py` — 4 tests: roles + gates +
  provenance, legacy row + degenerate flags, share_of_parent
  invariants (root/sums/gated, single + batch), and the decomposition
  identity (share_of_parent == metric-subtree share for a pure
  single-metric cascade).

## [0.27.0] — 2026-07

Closes [issue #20](https://github.com/shreyasrkarwa/Analytics/issues/20)
and [issue #19](https://github.com/shreyasrkarwa/Analytics/issues/19):
batch runs were hard to audit — the row-level flags exist, but nothing
summarized WHICH combinations were skipped/relaxed/stranded, and
per_group weight suggestion flooded the output with one direction-
mismatch warning per (group x metric).

### Added
- **`attrs['combo_report']`** (#20) on cascade_many outputs: one
  record per group-key combination — skipped + reason,
  targets_matched, rows_produced, n_gated_nodes, gate_relaxed,
  unallocated_total, weights_source ('fixed' | 'policy' |
  'suggested_global' | 'suggested_per_group' | 'default_attainment'),
  direction_mismatches and degenerate_fallback. Stored as records
  (attrs-concat safe); reconstruct with
  pd.DataFrame(q.attrs['combo_report']). All assembled from state the
  loop already had and discarded — no new machinery.
- **Direction-mismatch summarization** (#19), per_group mode only:
  unless the caller EXPLICITLY sets warn_on_direction_mismatch in
  suggest_config, per-combo warnings are silenced and ONE batch-level
  summary is emitted ("'metric' in N/M combinations"), pointing at the
  report. Explicit True keeps today's per-group warnings verbatim;
  explicit False silences everything. The report column is populated
  in all three cases — the signal is data, not noise.
- `tests/test_combo_report.py` — 5 tests: shape + skipped reasons,
  gate bookkeeping (strand_at_root unallocated vs redistribute
  relaxed), weights_source variants, warning-summary trichotomy,
  attrs concat safety.

## [0.26.0] — 2026-07

Closes [issue #17](https://github.com/shreyasrkarwa/Analytics/issues/17)
and [issue #49](https://github.com/shreyasrkarwa/Analytics/issues/49)
(the same request, filed 4 days apart): the package rolls leaf metrics
up the tree internally everywhere (gates, blend shares, route_targets)
but never surfaced it, so consumers hand-rolled ~30-line
depth-descending rollups to answer "why did this node get its quota?".

### Added
- **`rollup_metrics(quotas_long, metrics, agg='sum', row_keys=None)`**
  (#17): one `<metric>_subtree` column per metric — leaf rows carry
  their own value, managers the subtree aggregate, per cascade.
  Aggregation runs over DESCENDANT LEAVES (mean/max/min are leaf
  aggregates, never a mean of means); 'sum' matches the cascader's
  own rollup semantics. Carried source columns stay leaf-grain (NaN
  on managers — the v0.19.2 contract is untouched). Wrong cascade
  keys cannot corrupt silently: the #40 orphan guard raises, naming
  the poison columns.
- **`cascade_many(..., attach_metrics=True | [cols])`** (#49): the
  requested flag — rolls up every numeric metadata_cols column (or an
  explicit list) as a post-step through rollup_metrics; one code path.
- **cascade_levels outputs now work too**: deeper transitions inherit
  the root-target key columns down the parent chain (they were NaN
  below the first transition), and the output carries
  `.attrs['cascade_row_keys']` like cascade_many's — found by the
  orphan guard during development, not by a user.
- `tests/test_rollup_metrics.py` — 5 tests: per-depth sums with
  quarter isolation + root anchor, attach_metrics equivalence + list
  form, leaf-grain mean/max, cascade_levels output, validation
  (missing/non-numeric/poisoned keys).

## [0.25.1] — 2026-07

Closes [issue #29](https://github.com/shreyasrkarwa/Analytics/issues/29)
and [issue #44](https://github.com/shreyasrkarwa/Analytics/issues/44) —
both already resolved by `concentrate()` (v0.24.0); this release pins
the exact scenarios and makes them findable in the docs.

### Verified (now regression tests)
- **#29 lone-survivor routing**: `concentrate(q, 'UKI2',
  scope={'segment': 'Government'})` gives the survivor the FULL parent
  pool (no leak — the pre-v0.13/v0.16 leak the issue describes is
  long gone), zeroes the siblings' subtrees, and the survivor carries
  the hedged pool automatically via per-row ratio re-derivation — no
  pre-computed hedged pool, no sibling enumeration.
- **#44 team destination**: a hyphenated id like 'CENTRAL6-MIGRATION'
  is detected as a MANAGER from the graph (pin_type='subtree'), so
  the region total lands on its reps and the team-vs-rep footgun
  cannot happen; every other team's reps zero; region conserved per
  combo.

### Docs
- concentrate() docstring now names both phrasings ("route 100% of a
  parent to a single child", "concentrate a group onto one
  team/subtree").

## [0.25.0] — 2026-07

Closes [issue #48](https://github.com/shreyasrkarwa/Analytics/issues/48):
one pin matching zero rows (a product a region doesn't sell, a team
with no reps in a quarter) aborted the entire apply_pins batch,
forcing callers to pre-filter pins with hand-rolled scope matching.
Mirrors the established `on_error='skip'` pattern from
cascade_many/cascade_levels: dropped intent is data, not an abort.

### Added
- **`apply_pins(..., on_missing='error'|'skip'|'warn')`** (default
  'error' — behavior unchanged). 'skip' drops zero-row pins and
  records them in the feasibility report; 'warn' additionally emits
  ONE summary warning naming them. New report columns: `skipped`
  (bool) and `reason` ('node_absent' when the node id is nowhere in
  the frame; 'empty_scope' when the node exists but Pin.scope matched
  nothing). Skipped rows keep pin_node/basis/requested_total, zeros
  elsewhere, feasible=False; report stays in INPUT pin order.
- **No ghost side-effects** (the design decision the issue left
  open): a skipped pin is dropped ENTIRELY — excluded from the
  protection set and the depth ordering — so its node still absorbs
  for other pins and still rescales inside pinned subtrees. Pinned by
  test: a batch with a vacuous pin produces a frame identical to the
  batch without it.
- A missing Pin.scope COLUMN (typo) still raises regardless of
  on_missing — that's a programming error, not a missing combination.
- The default-'error' message now points at on_missing='skip'.
- `tests/test_pin_on_missing.py` — 5 tests: default unchanged, skip
  reasons + input order, warn summary, ghost-protection identity,
  all-skipped no-op + validation.

### Notes
- `redistribute`/`concentrate` deliberately keep strict validation —
  they are single planning operations, not batches; a missing node
  there is a real error.

## [0.24.0] — 2026-07

Closes [issue #47](https://github.com/shreyasrkarwa/Analytics/issues/47):
`concentrate()` — collapse siblings' scoped quota onto one sibling,
the inverse of `redistribute`. Review verdict: expressible today as
direct pins or chained redistribute (verified identical), and the
filer's ~40-line workaround is over-built (manager zero-pins zero the
whole subtree; no per-rep pins, no `exclude` needed). The real gaps —
hand-computed group total, spurious unabsorbed noise in the no-buffer
report, no single roles report — are what the helper closes.

### Added
- **`concentrate(quotas_long, to_node, from_nodes=, scope=,
  freeze_nodes=, row_keys=)`** — pins `to_node` to the summed scoped
  baseline of the group and zeroes the sources (default: all unfrozen
  siblings), via the same quiet-run + verify machinery as
  redistribute. Full pin contract: per-depth reshaping (destination's
  internal mix preserved), parent conservation, per-row hedge
  re-derivation, scope isolation, freeze honored. Returns
  (edited_df, report) with destination/source/bystander roles and an
  `exact` flag; bystanders (unlisted siblings) are verified to stay at
  baseline to the cent.
- Sources must be siblings; cross-parent pulls error with a
  route_targets pointer.
- `tests/test_concentrate.py` — 5 tests: three-spelling equivalence
  (helper == direct pins == chained redistribute), subset + bystander,
  freeze defaults, inverse-composition anchor (team level identical;
  rep-level equal-split after transiting $0, as documented),
  validation suite.

### Fixed (design detail worth knowing)
- Inside `concentrate`, sources are zeroed BEFORE the destination pin
  lands (same depth, stable order): with a bystander buffer present,
  the buffer inflates then sheds and never floors at $0 — so its
  internal rep mix survives. Destination-first would transiently floor
  the buffer and equal-split its reps. Caught by test 2 during
  development.

### Internal
- Shared `_scope_mask` / `_run_pins_quietly` helpers now back both
  redistribute and concentrate (no behavior change; full suite green).

## [0.23.0] — 2026-07

Closes [issue #43](https://github.com/shreyasrkarwa/Analytics/issues/43):
cross-sibling redistribution that reshapes source AND destination at
every depth. Review verdict: the capability already existed as scoped
pin composition — `apply_pins(q, [Pin('EAST', 0, scope=...)])` zeroes
the source subtree at every depth and grows the unpinned siblings at
baseline proportions, hedge ratios preserved per row (verified, incl.
the filer's manual proportional-share computation being unnecessary).
The genuine gap was ergonomics: custom `weights=` required reading
baselines out of the frame and hand-computing destination pins.

### Added
- **`redistribute(quotas_long, from_node, to_nodes=, weights=,
  scope=, freeze_nodes=, row_keys=)`** — the requested API, built as
  thin sugar over `apply_pins` (it writes the pins for you; no new
  engine). `weights='proportional'` (default) | `'equal'` | dict
  (normalized). Inherits the full pin contract: per-depth subtree
  reshaping, parent conservation, per-row hedge re-derivation (#21),
  scope isolation, freeze honored, $0 floors. Returns
  (edited_df, report) with per-node roles
  (source/destination/bystander) and an `exact` flag — bystander rows
  VERIFY the cancellation identity (non-recipients end exactly at
  baseline) rather than assuming it.
- Recipients must be siblings of the source; cross-parent moves get a
  clear error pointing at route_targets.
- Internal absorption noise is suppressed only for redistribute's own
  conservation-neutral pin package and replaced by explicit
  verification; genuine failures (frozen mass, floors) still warn via
  the report's `exact` column.
- `tests/test_redistribute.py` — 5 tests: default == single scoped
  Pin (the identity), dict weights with bystander-exact check,
  no-buffer silence, equal split, validation suite.

## [0.22.1] — 2026-07

Closes [issue #42](https://github.com/shreyasrkarwa/Analytics/issues/42):
"no native remainder-to-unpinned-siblings pin mode." Review verdict:
the mode exists — as plain pin composition — and became reliable with
the v0.20.0/v0.22.0 pin fixes (the issue predates them). No new API:
a `children=`/`remainder_node=` parameter would be a second spelling
of a three-line pin list. Docs + pinned tests only.

### Docs
- "Remainder pins (issue #42)" recipe in the apply_pins docstring and
  README: `[Pin('T1', x), Pin('T2', y)]` sends the leftover to the
  unpinned siblings at baseline proportions with the parent conserved;
  add `Pin(parent, total)` for the requested
  `Pin(parent, total, children={...}, remainder='auto')`. Why it is
  exact: pinned nodes never absorb for other pins (#24), proportional
  absorption preserves unpinned ratios (the #37 identity), and
  depth-ordered application (#41) lands the parent first regardless
  of list order.

### Added
- `tests/test_remainder_pins.py` — 3 tests: sibling-only remainder
  with NON-canceling deltas (proportionality genuinely exercised),
  parent-total + fixed children (per-quarter conservation + internal
  mix of remainder teams preserved), and over-pinned children
  (parent-child gap exactly equals reported subtree_shortfall +
  child unabsorbed; no negatives).

## [0.22.0] — 2026-07

Closes [issue #41](https://github.com/shreyasrkarwa/Analytics/issues/41):
pin list order (managers before leaves) was load-bearing. The reported
symptom — a leaf pin overwritten by a later manager-pin subtree
rescale — was already fixed by v0.20.0's protection-aware rescale
(verified across all 24 orderings of the filer's notebook shape:
region + team + rep + zero pins, every pin held, every parent
conserved). This release removes the residual: ABSORBER rows could
still differ by list order, so two colleagues writing the same pins in
different order got different plans for the unpinned reps.

### Changed
- **Canonical application order**: apply_pins sorts pins by the DEPTH
  of the pinned node, shallowest first (managers before leaves —
  exactly the ordering the filer found load-bearing), stable within a
  depth. The output frame, absorber rows included, is now identical
  for any ordering of `pins`. Leaf-pin allocations are computed
  against post-manager-rescale baselines.
- The feasibility report is returned in the INPUT pin order (rows are
  slotted back), so positional indexing keeps working.

### Added
- `tests/test_pin_ordering.py` — 4 tests: the literal #41 symptom
  (leaf pin listed before its covering manager pin), all-24-
  permutations bit-identical frames + conservation, report-order
  contract, same-depth stability.

## [0.21.0] — 2026-07

Closes [issue #40](https://github.com/shreyasrkarwa/Analytics/issues/40):
apply_pins without `row_keys` inferred the cascade key from every
non-structural column, so per-node columns (metadata_cols) split
parent and child rows into different key tuples — manager pins
silently downgraded to pin_type='leaf' (descendants never followed the
pin) and, worse than reported, sibling absorption mis-grouped too.
Both halves of the issue's Ask are implemented.

### Added
- **Cascade-identity stamp (Ask a).** cascade_many outputs carry
  `.attrs['cascade_row_keys']` — the exact group_keys + sub-target
  columns. `apply_pins(row_keys=None)` uses it automatically, so pins
  Just Work on batch outputs even with metadata_cols present.
- **Orphan guard (Ask b).** Wrong keys are now a hard error, never
  silent corruption: if any row's parent exists in the frame but not
  under the same cascade key (impossible under correct keys — a
  cascade always contains the parent row), apply_pins raises a
  ValueError naming the per-node columns poisoning the identity and
  suggesting the corrected row_keys. Catches wrong EXPLICIT row_keys
  too. Leaf-only frames (parents absent entirely) remain allowed.
- `tests/test_pin_row_keys.py` — 6 tests: stamp contents, no-row_keys
  subtree pin with metadata, poison diagnosis, wrong explicit keys,
  the filer's workaround, leaf-only frames.

### Fixed
- Cascade-key tuples normalize NaN to None, so legitimately-NaN key
  values no longer split a cascade into per-row groups (NaN != NaN).

## [0.20.0] — 2026-07

Closes [issue #39](https://github.com/shreyasrkarwa/Analytics/issues/39):
"pins on the hedged layer don't propagate the cross-level buffer."
Review verdict: the HEADLINE claim does not reproduce — a
cascaded-basis manager pin under `HedgeByDepth` already rolls
descendants up to pinned x cross-level hedge, because `apply_pins`
scales descendant BASE values and re-derives cascaded from each row's
own hedge RATIO (now pinned by test). But probing the filer's
workaround ("separately-pinned/zeroed reps") exposed three real
defects in the subtree rescale, all fixed here.

### Fixed
- **Later subtree pins no longer trample earlier descendant pins.**
  Every pin's node is protected from every other pin's rescales, so
  pin order no longer changes where pinned nodes land (this was almost
  certainly the wrong rollup behind #39's symptom).
- **`freeze_nodes` honored inside subtrees.** The docstring promised
  frozen nodes are "never modified" — but a frozen node inside a
  pinned (or absorbing) manager's subtree was scaled. Now it holds.
- **`Pin.exclude` protects descendants of a manager pin** (the #39
  Ask), not just sibling absorbers.

### Changed
- `_scale_subtree` replaced by a protection-aware recursive rescale:
  protected children keep their values, free children scale to fill
  the remainder proportional to their base (equal split when the free
  baseline is $0 — which also means a zero-baseline subtree pin now
  DISTRIBUTES instead of sitting on the manager row). With nothing
  protected, identical to the old proportional rescale (pinned by
  test).
- Sibling absorption is weighted by FREE capacity (base minus
  protected mass inside the subtree), so absorbers never push deltas
  onto protected descendants.
- Feasibility report gains a `subtree_shortfall` column: when
  protected descendants alone exceed the pinned total, free rows floor
  at $0, `feasible=False`, and a warning names the column — never
  hidden.

### Added
- `tests/test_hedged_layer_pins.py` — 7 tests: the disproven #39
  identity, order-independence, frozen/excluded descendants,
  infeasible shortfall, absorber-subtree protection, unprotected
  equivalence.

## [0.19.2] — 2026-07

Closes [issue #16](https://github.com/shreyasrkarwa/Analytics/issues/16):
"quotas_long_df omits the metric values used in the cascade." Review
verdict: solved since v0.8.0 — `metadata_cols` (issue #7) carries ANY
hierarchy column onto leaf rows, metric values included, even the
column driving the cascade. The gap was discoverability, so this is a
docs + pinned-test release; no behavior changes.

### Added
- Regression test (`test_output_metadata.py` test 7): metric columns
  carried via `metadata_cols` are emitted on leaf rows (NaN on
  non-leaves), and the cascade numbers are identical with and without
  the carry — including when the carried column is the driving metric
  of an explicit `MetricSpec`.

### Docs
- `cascade_many(metadata_cols=...)` docstring now states it carries
  metric columns too ("despite the name"), that explicit MetricSpecs
  still resolve carried columns, and documents the two KEY IDENTITIES:
  hierarchy leaf = deepest taxonomy column (+ group_keys present in
  hierarchy_df); cascade row = group_keys + sub-target columns.
  Target-frame-only columns are not join keys for leaf-grain data.
- README "What's New in v0.19.2" mapping the requested
  `carry_metric_cols` API onto `metadata_cols`.

## [0.19.1] — 2026-07

Closes [issue #36](https://github.com/shreyasrkarwa/Analytics/issues/36):
guidance + guardrails for metric-column GRAIN. An ancestor-level value
repeated onto every leaf row double-counts under SUM rollups and
collapses sibling shares to equal splits — silently (two field
incidents).

### Added
- **Grain-mismatch detector**: `cascade_quota` (and therefore every
  batch/level driver) warns when an active metric's or gate's values
  are identical across >= 90% of leaf-sibling groups — the signature
  of a coarser-grain column. Conservative by design: needs >= 2 groups
  of >= 2 leaves, boolean/0-1 metrics exempt, all-zero groups exempt
  (the zero-signal warning owns those), once per metric per cascader,
  warning-only (legitimately uniform books still cascade).

### Docs
- README "Metric Grain" section + `from_dataframe(metrics_cols=...)`
  note: metric columns must be at LEAF grain; non-leaf values are
  leaf-sums.
- The proposed `grain`/`dedup_key` auto-dedup is deliberately NOT
  implemented: collapsing a repeated value requires knowing which
  aggregation is meaningful (e.g. MAX per account, then SUM per rep) —
  source-data semantics that belong in the feed query. Guessing would
  create a new class of silently-wrong numbers; the library detects
  and warns instead.

## [0.19.0] — 2026-07

Implements [issue #35](https://github.com/shreyasrkarwa/Analytics/issues/35):
mixed weight strategies in the batch path. Review notes: the
`cascade_quota` half of the ticket ("honor already-weighted specs
verbatim" / `weighting='fixed'`) has always been the only behavior —
verified and documented in v0.17.0 (issue #34). What was genuinely
missing: per-COMBINATION strategy in `cascade_many`, where a static
`metrics=` list and `suggest_config=` were mutually exclusive for the
whole batch.

### Added
- **`cascade_many(metrics=...)` accepts a callable** — evaluated per
  combination with the group-key dict; returning a list of MetricSpecs
  fixes that combination's slate VERBATIM, returning None falls
  through to `suggest_config` (global or per_group) or, without one,
  to the legacy '_Attainment' path (matching
  `cascade_quota(metrics=None)`). Mirrors the v0.15.0 gate_metrics
  callable:
  `metrics=lambda g: DC_ONLY if g['st1_sales_type'] == 'Migration'
  else None`.
- A callable may coexist with `suggest_config`; a STATIC list still
  may not (unchanged validation). Bad returns / raising policies flow
  through `on_error` + the dropped-targets frame. `weights_long`
  records whichever slate each combination actually used. Exact
  equivalence with the split-and-concat two-call workaround is pinned
  by test; `cascade_levels` inherits the capability via level_kwargs.

## [0.18.1] — 2026-07

Closes [issue #37](https://github.com/shreyasrkarwa/Analytics/issues/37)
as already-resolved, with verification. Every element of the proposed
API maps onto shipped features: `pins=` → `new_ic_overrides`
(any-level pins since v0.13.0), `pin_basis='base'` →
`override_basis='base'` (the default since v0.13.0), aggregate
cross-combo totals → `apply_pins` (v0.16.0). The claimed-missing
`remainder='baseline_proportional_nonpinned'` policy is the ONLY
behavior: because each child's baseline equals target x its composite
share, renormalizing shares among the non-pinned children is
algebraically identical to splitting the remainder proportional to
their baseline values.

### Docs / Tests
- The absorption policy is now stated explicitly on the
  `new_ic_overrides` docstring and in the README.
- The issue's exact DACH scenario ($50M, DACH1 pinned $12.5M base,
  DACH2 $12M) is a pinned regression test: non-pinned siblings match
  baseline-proportional to $0.0000, base conserves at every depth,
  hedged values derive from each node's ratio.
- No code behavior changes.

## [0.18.0] — 2026-07

Implements [issue #30](https://github.com/shreyasrkarwa/Analytics/issues/30):
a supported multi-level / recursive cascade driver. Review notes:
two of the four motivations already shipped — per-level hedges
(v0.11.0 `HedgeByDepth`) and level-specific pins (v0.13.0/v0.16.0) —
and the requested `cascade_level` primitive already exists as
`cascade_many(df, targets, group_keys=[parent_col],
taxonomy=[parent_col, child_col])`, where every parent is its own
root. What was genuinely missing: chaining levels with DIFFERENT
kwargs per transition (especially different metric blends per level).

### Added
- **`cascade_levels(hierarchy_df, root_targets, taxonomy, target_col,
  level_kwargs=[...], on_error='skip', return_dropped=False)`** —
  cascades one level at a time, threading each level's BASE output
  into the next level's targets, with per-transition kwargs (metrics,
  gate_metrics, hedge_multiplier, new_ic_overrides, suggest_config,
  ...). Each transition hedges only its own step; base conservation
  holds per parent at every level; intermediate levels aggregate
  metric columns per (parent, child) by SUM (matching the cascader's
  leaf rollups); sub-target key columns (fiscal_quarter, ...) thread
  through; dropped targets aggregate across levels with a `level`
  tag. Output: one tidy frame per node at its level (`level`,
  `depth`, is_leaf at the deepest level, audit columns intact).
- **Pinned equivalence anchor:** with uniform kwargs, the chained
  base layer equals a single full-tree `cascade_many` — number for
  number.

### Notes
- Requires globally unique child ids (qualified naming); ambiguity
  raises. Jagged hierarchies (missing level values) are excluded per
  transition with a warning — use the full-tree cascade for those.
- Replaces the consumer's three hand-rolled per-level engines
  (~300 lines).

## [0.17.0] — 2026-07

Closes [issue #34](https://github.com/shreyasrkarwa/Analytics/issues/34):
a deterministic "proportional-to-metric" mode. Verification first: the
mechanism has ALWAYS existed — fixed-weight `MetricSpec`s passed to
`cascade_quota(metrics=...)` are used as-is; `suggest_weights` is an
optional helper, never a required stage, and no statistics run unless
you call it. The requested `weighting='fixed'` flag would toggle
behavior that is already the only behavior. What was missing was the
discoverable front door.

### Added
- **`QuotaCascader.cascade_proportional(root, target, metric='dc_seats')`**
  — the issue's Option B, verbatim: split the target proportional to
  one metric (or a fixed blend via
  `metrics={'dc_seats': 1.0, 'cloud_seats': 0.5}`), deterministic and
  explainable at any slice size including n=1/2. Pure sugar: builds
  the fixed-weight specs and delegates to `cascade_quota`, so gates,
  `HedgeByDepth`, pins, `gate_fallback`, the base layer, and
  `gating_report()` all work unchanged. Column resolution follows the
  v0.7.1 order (Qi_<name> convention, then the plain column).
  `direction='inverse'` supported; mixed directions -> build specs
  directly.

### Docs
- `suggest_weights` docstring and the README now state explicitly that
  the suggester is optional, with a "Deterministic proportional
  splits" section. (The #33 immunity noted in the issue also already
  ships: since v0.12.0 degenerate slices keep declared weights.)

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
