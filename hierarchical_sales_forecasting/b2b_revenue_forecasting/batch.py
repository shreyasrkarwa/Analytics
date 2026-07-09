"""
Batch / multi-combination cascading (issue #4).

Real planning runs cascade many targets across many segments — every
(sales_type, product, regional) combination, for every quarter. Before
v0.7.0 every consumer hand-rolled the same loop: filter the hierarchy
per combo, build a SalesHierarchy, suggest weights, cascade, tag the
outputs with the combo keys, concat. `cascade_many` centralizes that
loop — and with it all the correctness work the package already does
(value coercion, self-loop/collision handling, DAG validation, root
never gated, per-depth reconciliation of the base layer).
"""
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from b2b_revenue_forecasting.hierarchy import SalesHierarchy
from b2b_revenue_forecasting.metric_spec import MetricSpec
from b2b_revenue_forecasting.quota_cascader import QuotaCascader

_WEIGHTS_MODES = ("global", "per_group")
_ON_ERROR = ("skip", "raise")


def cascade_many(
    hierarchy_df: pd.DataFrame,
    target_df: pd.DataFrame,
    group_keys: List[str],
    target_col: str,
    taxonomy: List[str],
    metrics: Optional[List[MetricSpec]] = None,
    gate_metrics: Optional[Union[
        List[MetricSpec],
        Callable[[Dict[str, Any]], Optional[List[MetricSpec]]],
    ]] = None,
    hedge_multiplier: Union[float, Dict[str, float]] = 1.0,
    suggest_config: Optional[Dict[str, Any]] = None,
    weights_mode: str = "global",
    level_names: Optional[List[str]] = None,
    brand_new_col: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
    on_error: str = "skip",
    on_collision: str = "suffix",
    verbose: bool = False,
    return_dropped: bool = False,
    **cascade_kwargs: Any,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame],
           Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Cascade many targets across many segment combinations in one call.

    For each unique combination of `group_keys` found in `target_df`:
      1. `hierarchy_df` is filtered to that combination (on whichever
         group_keys exist as hierarchy_df columns),
      2. a SalesHierarchy is built ONCE (with all validation/coercion),
      3. weights are resolved (fixed `metrics`, or suggested via
         `suggest_config` — see `weights_mode`),
      4. EVERY matching `target_df` row is cascaded against that prepared
         hierarchy — so extra target_df columns (e.g. fiscal_quarter)
         act as sub-targets and reuse the prepared group
         ("prepare once / cascade many").

    Parameters
    ----------
    hierarchy_df : pd.DataFrame
        Long frame with taxonomy columns + metric/gate columns, one row
        per leaf (rep/territory). May also carry the group_key columns
        used to slice it (e.g. st1_sales_type, base_product_r4f).
    target_df : pd.DataFrame
        One row per cascade to run: group_key columns + `target_col`.
        Any OTHER columns (e.g. fiscal_quarter) are treated as
        sub-target identifiers — passed through to the output so each
        cascade's rows are fully tagged.
    group_keys : list[str]
        Columns defining a combination. Every group_key must exist in
        target_df; those that also exist in hierarchy_df are used to
        filter it. The taxonomy root column may be one of them (typical:
        ["st1_sales_type", "base_product_r4f", "regional"]).
    target_col : str
        Column in target_df holding the dollar target for that cascade.
    taxonomy : list[str]
        Hierarchy columns from root to leaf, passed to
        SalesHierarchy.from_dataframe(path_cols=...).
    metrics : list[MetricSpec], optional
        Fixed metric slate used for every combination. Mutually
        exclusive with `suggest_config`.
    gate_metrics : list[MetricSpec] | callable, optional
        Gate metrics, passed straight to cascade_quota. All
        gate_fallback semantics (v0.5.0) apply; override the fallback
        via **cascade_kwargs (e.g. gate_fallback="strand_at_root").

        Conditional gating (issue #14, v0.15.0): pass a CALLABLE that
        receives the combination's group-key dict and returns the gate
        list for that combination (or None for "no gates"). Resolved
        once per combination — gates only where the policy says so::

            gate_metrics=lambda g: (
                [MetricSpec('dc_seats', columns=['dc_seats'])]
                if g['st1_sales_type'] == 'Migration' else None)

        A mapping-style policy is the same thing via dict.get::

            BY_TYPE = {'Migration': [MetricSpec('dc_seats',
                                                columns=['dc_seats'])]}
            gate_metrics=lambda g: BY_TYPE.get(g['st1_sales_type'])

        If the callable raises for a combination, that combination is
        handled per `on_error` (skip + dropped-targets frame, or raise).
    hedge_multiplier : float | dict | HedgeByDepth
        Passed to cascade_quota. A HedgeByDepth spec (issue #13) is
        resolved against EACH combination's freshly built hierarchy —
        the only way to express a per-level hedge here, since node ids
        are never visible to the caller. E.g.
        HedgeByDepth(from_leaves={1: 1.10, 2: 1.05}). The un-hedged
        base layer is always included in the output (base_quota
        column).
    suggest_config : dict, optional
        Enables data-driven weights via MetricSpec.suggest_weights.
        Shape: {"target_column": str, "candidate_metrics": [...],
                ...any other suggest_weights kwargs}.
        Mutually exclusive with `metrics`.
    weights_mode : str
        "global" (default) — suggest once on the FULL hierarchy_df and
        reuse for every combination. "per_group" — re-suggest on each
        combination's slice (weights adapt to the segment; slower).
        Ignored when `metrics` is given.
    level_names : list[str], optional
        Human-readable names per depth (see quotas_to_dataframe).
        Defaults to `taxonomy`.
    brand_new_col : str, optional
        Passed to from_dataframe for each slice; combine with
        new_ic_attr='_is_brand_new' in **cascade_kwargs.
    metadata_cols : list[str], optional
        Descriptive columns (rep name, segment, geo, ...) carried
        through to the leaf rows of quotas_long_df (issue #7). They are
        excluded from metric aggregation.
    on_error : str
        "skip" (default) — a failing combination emits a warning and is
        excluded from the outputs; a summary warning lists all failures
        at the end. "raise" — re-raise the first failure immediately.
    return_dropped : bool
        When True, a THIRD frame is returned containing every target_df
        row that was dropped (its combination failed / had no hierarchy
        branch), with all original columns plus a `reason` column —
        so dropped money is data, not log noise (issue #26). The same
        rows are ALWAYS attached as
        quotas_long.attrs['dropped_targets'] (as a list of record
        dicts — reconstruct with pd.DataFrame(...)), regardless of
        this flag. Feed the frame to route_targets() to place the
        money on chosen recipients (issues #25/#32).
    on_collision : str
        Passed to from_dataframe (v0.6.0 duplicate-level policy).
    verbose : bool
        Passed to cascade_quota (default False here — batch runs would
        otherwise print one weights table per combination).
    **cascade_kwargs
        Any other cascade_quota argument (new_ic_attr, new_ic_rule,
        new_ic_overrides, gate_fallback, ...).

    Returns
    -------
    (quotas_long_df, weights_long_df)
        quotas_long_df — one row per (cascade, node): group_keys +
            sub-target columns + node_id, parent, depth, level, is_leaf,
            cascaded_quota, base_quota (un-hedged), hedge_buffer, and
            is_gated / gate_relaxed / is_unallocated when applicable.
        weights_long_df — one row per (combination, metric): group_keys
            + the normalized-weights table actually used. Empty when
            cascading via the legacy path (no metrics at all).
    """
    # ---- Argument validation -----------------------------------------
    if weights_mode not in _WEIGHTS_MODES:
        raise ValueError(f"weights_mode must be one of {_WEIGHTS_MODES}, "
                         f"got '{weights_mode}'.")
    if on_error not in _ON_ERROR:
        raise ValueError(f"on_error must be one of {_ON_ERROR}, "
                         f"got '{on_error}'.")
    if metrics is not None and suggest_config is not None:
        raise ValueError(
            "Pass either metrics=[MetricSpec, ...] (fixed slate) OR "
            "suggest_config={...} (data-driven weights) — not both."
        )
    missing_keys = [k for k in group_keys if k not in target_df.columns]
    if missing_keys:
        raise ValueError(f"group_keys {missing_keys} not found in target_df "
                         f"columns: {list(target_df.columns)}")
    if target_col not in target_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in target_df.")
    missing_tax = [c for c in taxonomy if c not in hierarchy_df.columns]
    if missing_tax:
        raise ValueError(f"taxonomy columns {missing_tax} not found in "
                         f"hierarchy_df.")

    if level_names is None:
        level_names = list(taxonomy)

    # group_keys that can actually slice hierarchy_df
    filter_keys = [k for k in group_keys if k in hierarchy_df.columns]

    def _suggest(df_slice: pd.DataFrame) -> List[MetricSpec]:
        cfg = dict(suggest_config)
        suggested, _report = MetricSpec.suggest_weights(
            df_slice,
            target_column=cfg.pop("target_column"),
            candidate_metrics=cfg.pop("candidate_metrics"),
            **cfg,
        )
        return suggested

    # Global weights resolved once, if applicable
    global_metrics: Optional[List[MetricSpec]] = metrics
    if suggest_config is not None and weights_mode == "global":
        global_metrics = _suggest(hierarchy_df)

    quota_frames: List[pd.DataFrame] = []
    weight_frames: List[pd.DataFrame] = []
    failures: List[Tuple[tuple, str]] = []
    dropped_frames: List[pd.DataFrame] = []

    passthrough_cols = [c for c in target_df.columns
                        if c not in group_keys and c != target_col]

    # ---- Iterate combinations (prepare once ... ) ----------------------
    for combo_vals, combo_targets in target_df.groupby(group_keys, sort=False):
        if not isinstance(combo_vals, tuple):
            combo_vals = (combo_vals,)
        combo_dict = dict(zip(group_keys, combo_vals))
        try:
            # 1. Filter the hierarchy to this combination
            df_slice = hierarchy_df
            for k in filter_keys:
                df_slice = df_slice[df_slice[k] == combo_dict[k]]
            if df_slice.empty:
                raise ValueError(
                    f"hierarchy_df has no rows for combination {combo_dict} "
                    f"(filtered on {filter_keys})."
                )

            # 2. Identify a single root for the slice
            roots = df_slice[taxonomy[0]].dropna().unique()
            if len(roots) != 1:
                raise ValueError(
                    f"Combination {combo_dict} yields {len(roots)} distinct "
                    f"'{taxonomy[0]}' roots ({list(roots)[:5]}); expected "
                    f"exactly 1. Add '{taxonomy[0]}' to group_keys or "
                    f"pre-split the data."
                )
            root = str(roots[0])

            # 3. Build the hierarchy ONCE for this combination
            h = SalesHierarchy()
            metric_cols = [c for c in hierarchy_df.columns
                           if c not in taxonomy and c not in group_keys
                           and c != target_col
                           and c not in (metadata_cols or [])]
            h.from_dataframe(df_slice, path_cols=taxonomy,
                             metrics_cols=metric_cols,
                             brand_new_col=brand_new_col,
                             on_collision=on_collision,
                             metadata_cols=metadata_cols)
            cascader = QuotaCascader(h)

            # 4. Resolve weights for this combination
            if suggest_config is not None and weights_mode == "per_group":
                combo_metrics = _suggest(df_slice)
            else:
                combo_metrics = global_metrics

            # 4b. Resolve gates for this combination (issue #14) —
            # a callable policy is evaluated against the group-key dict.
            if callable(gate_metrics):
                combo_gates = gate_metrics(dict(combo_dict))
                if combo_gates is not None and not (
                        isinstance(combo_gates, list)
                        and all(isinstance(g, MetricSpec)
                                for g in combo_gates)):
                    raise ValueError(
                        f"gate_metrics callable must return a list of "
                        f"MetricSpec or None for combination {combo_dict}, "
                        f"got {type(combo_gates).__name__}."
                    )
            else:
                combo_gates = gate_metrics

            # Record the weights actually used (once per combination)
            if combo_metrics:
                wdf = MetricSpec.normalized_weights(combo_metrics)
                for k, v in combo_dict.items():
                    wdf[k] = v
                weight_frames.append(wdf)

            # 5. Cascade every sub-target row against the prepared group
            for _, trow in combo_targets.iterrows():
                target = float(trow[target_col])
                quotas = cascader.cascade_quota(
                    root, target,
                    hedge_multiplier=hedge_multiplier,
                    metrics=combo_metrics,
                    gate_metrics=combo_gates,
                    verbose=verbose,
                    **cascade_kwargs,
                )
                qdf = cascader.quotas_to_dataframe(
                    quotas, level_names=level_names,
                    unhedged_quotas="auto",
                    metadata_cols=metadata_cols,
                )
                qdf = qdf.rename(columns={"unhedged_quota": "base_quota"})
                # Tag with group keys + sub-target identifiers + the target
                for k, v in combo_dict.items():
                    qdf[k] = v
                for c in passthrough_cols:
                    qdf[c] = trow[c]
                qdf[target_col] = target
                quota_frames.append(qdf)

        except Exception as exc:  # noqa: BLE001 — reported per policy
            if on_error == "raise":
                raise
            failures.append((combo_vals, f"{type(exc).__name__}: {exc}"))
            # Issue #26: dropped money must be data, not log noise —
            # keep the full original target rows with the reason.
            dropped_frames.append(
                combo_targets.assign(reason=f"{type(exc).__name__}: {exc}"))
            warnings.warn(
                f"cascade_many: combination {combo_dict} skipped — "
                f"{type(exc).__name__}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    if failures:
        warnings.warn(
            f"cascade_many finished with {len(failures)} of "
            f"{target_df.groupby(group_keys).ngroups} combination(s) "
            f"skipped: {[dict(zip(group_keys, f[0])) for f in failures[:10]]}",
            UserWarning,
            stacklevel=2,
        )

    # ---- Assemble tidy long outputs ------------------------------------
    id_cols = group_keys + passthrough_cols + [target_col]
    if quota_frames:
        quotas_long = pd.concat(quota_frames, ignore_index=True)
        lead = [c for c in id_cols if c in quotas_long.columns]
        rest = [c for c in quotas_long.columns if c not in lead]
        quotas_long = quotas_long[lead + rest]
    else:
        quotas_long = pd.DataFrame()

    if weight_frames:
        weights_long = pd.concat(weight_frames, ignore_index=True)
        lead = [c for c in group_keys if c in weights_long.columns]
        rest = [c for c in weights_long.columns if c not in lead]
        weights_long = weights_long[lead + rest]
    else:
        weights_long = pd.DataFrame()

    # ---- Dropped-targets frame (issue #26) ------------------------------
    # Always available on the output's attrs; opt into an explicit third
    # return value with return_dropped=True.
    if dropped_frames:
        dropped_long = pd.concat(dropped_frames, ignore_index=True)
    else:
        dropped_long = pd.DataFrame(
            columns=list(target_df.columns) + ["reason"])
    # Stored as RECORDS, not a DataFrame: pandas compares .attrs with ==
    # during pd.concat, and a DataFrame value there makes concatenating
    # two cascade outputs raise "truth value is ambiguous". Reconstruct
    # with pd.DataFrame(quotas_long.attrs['dropped_targets']).
    quotas_long.attrs["dropped_targets"] = dropped_long.to_dict("records")

    if return_dropped:
        return quotas_long, weights_long, dropped_long
    return quotas_long, weights_long


def cascade_levels(
    hierarchy_df: pd.DataFrame,
    root_targets: pd.DataFrame,
    taxonomy: List[str],
    target_col: str,
    level_kwargs: Optional[List[Dict[str, Any]]] = None,
    on_error: str = "skip",
    return_dropped: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Multi-level cascade driver (issue #30): cascade ONE LEVEL AT A TIME,
    threading level N's output into level N+1's targets, with different
    kwargs per transition — different metric blends, gates, hedges,
    pins, even suggest_config, at every level.

        result = cascade_levels(
            hierarchy_df, regional_targets,
            taxonomy=['regional', 'node_3_region', 'node_4_team',
                      'node_5_rep_no'],
            target_col='nn_acv_target',
            level_kwargs=[
                dict(metrics=KW_SPECS),                      # d0 -> d1
                dict(metrics=KW_SPECS, hedge_multiplier=1.05),  # d1 -> d2
                dict(metrics=SEAT_SPECS,                     # d2 -> d3
                     gate_metrics=DC_GATE,
                     new_ic_overrides={'East1_4': 250_000}),
            ])

    Each transition is a cascade_many run with a two-column taxonomy
    (every parent is its own root/combination) — the same "one-level
    primitive" you can also call directly:
    ``cascade_many(df, targets, group_keys=[parent_col],
    taxonomy=[parent_col, child_col], ...)``.

    Mechanics
    ---------
    - The BASE layer threads forward: level N children's base_quota
      becomes level N+1's targets, so each transition hedges only its
      own step (no hedge compounding across transitions) and base
      conservation holds per parent at every level.
    - For non-leaf transitions, metric columns are aggregated per
      (parent, child) by SUM — identical to the cascader's leaf-summed
      rollups. Non-numeric columns are dropped from intermediate
      levels.
    - Extra columns in root_targets (e.g. fiscal_quarter) are
      sub-target keys: they ride through every level and keep cascades
      separated, exactly as in cascade_many.
    - Combinations that fail at any transition follow `on_error`; their
      target rows are collected (with a `level` column) into
      result.attrs['dropped_targets'] (records) and, with
      return_dropped=True, returned as a second frame.

    Requirements
    ------------
    - Child ids must be unique to one parent across the frame (usual
      qualified naming: 'Enterprise_EMEA_T1'). Ambiguous ids raise.
    - Rows with a missing level value are excluded from that transition
      (with a warning) — for jagged hierarchies use the full-tree
      cascade_many instead.

    Returns
    -------
    A tidy frame with one row per node at ITS level: `level` (taxonomy
    column name), `depth` (level index), node_id, parent, is_leaf
    (True only at the deepest level), base_quota, cascaded_quota, the
    audit columns produced by that transition, and your key columns.
    """
    if len(taxonomy) < 2:
        raise ValueError("taxonomy must have at least 2 levels.")
    missing = [c for c in taxonomy if c not in hierarchy_df.columns]
    if missing:
        raise ValueError(f"taxonomy columns {missing} not in hierarchy_df.")
    if taxonomy[0] not in root_targets.columns:
        raise ValueError(f"root_targets must contain the root column "
                         f"'{taxonomy[0]}'.")
    if target_col not in root_targets.columns:
        raise ValueError(f"target_col '{target_col}' not in root_targets.")
    n_transitions = len(taxonomy) - 1
    if level_kwargs is None:
        level_kwargs = [{} for _ in range(n_transitions)]
    if len(level_kwargs) != n_transitions:
        raise ValueError(
            f"level_kwargs must have one dict per transition "
            f"({n_transitions} for this taxonomy), got {len(level_kwargs)}."
        )

    # Child ids must belong to exactly one parent (qualified naming)
    for parent_col, child_col in zip(taxonomy, taxonomy[1:]):
        pairs = hierarchy_df[[parent_col, child_col]].dropna()
        amb = (pairs.groupby(child_col)[parent_col].nunique())
        amb = amb[amb > 1]
        if len(amb):
            raise ValueError(
                f"'{child_col}' values appear under multiple "
                f"'{parent_col}' parents (e.g. {list(amb.index[:3])}) — "
                f"level chaining needs globally unique child ids. Qualify "
                f"the names or use the full-tree cascade_many."
            )

    key_cols = [c for c in root_targets.columns
                if c != taxonomy[0] and c != target_col]
    metric_cols = [c for c in hierarchy_df.columns if c not in taxonomy]

    pieces: List[pd.DataFrame] = []
    dropped_pieces: List[pd.DataFrame] = []
    targets = root_targets.copy()

    for i, (parent_col, child_col) in enumerate(zip(taxonomy, taxonomy[1:])):
        # Two-level slice with metrics aggregated per (parent, child) —
        # SUM matches the cascader's leaf-rollup semantics.
        pair_df = hierarchy_df[[parent_col, child_col] + metric_cols]
        n_missing = int(pair_df[[parent_col, child_col]].isna().any(axis=1).sum())
        if n_missing:
            warnings.warn(
                f"cascade_levels: {n_missing} row(s) with a missing "
                f"'{parent_col}'/'{child_col}' value are excluded from "
                f"transition {i} — jagged hierarchies need the full-tree "
                f"cascade_many.",
                UserWarning, stacklevel=2,
            )
            pair_df = pair_df.dropna(subset=[parent_col, child_col])
        agg = (pair_df.groupby([parent_col, child_col], sort=False)
               .sum(numeric_only=True).reset_index())

        step = cascade_many(
            agg, targets,
            group_keys=[parent_col],
            target_col=target_col,
            taxonomy=[parent_col, child_col],
            on_error=on_error,
            return_dropped=True,
            **level_kwargs[i],
        )
        quotas_i, _, dropped_i = step
        if len(dropped_i):
            dropped_pieces.append(dropped_i.assign(level=parent_col))

        if quotas_i.empty:
            break

        if i == 0:
            roots = quotas_i[quotas_i["depth"] == 0].copy()
            roots["level"] = taxonomy[0]
            roots["depth"] = 0
            roots["is_leaf"] = False
            pieces.append(roots)

        children = quotas_i[quotas_i["depth"] == 1].copy()
        children["level"] = child_col
        children["depth"] = i + 1
        children["is_leaf"] = (child_col == taxonomy[-1])
        pieces.append(children)

        # Thread the BASE layer forward as the next level's targets
        if i < n_transitions - 1:
            targets = (children[["node_id"] + key_cols + ["base_quota"]]
                       .rename(columns={"node_id": child_col,
                                        "base_quota": target_col})
                       .reset_index(drop=True))

    if pieces:
        result = pd.concat(pieces, ignore_index=True)
        lead = ["level", "depth", "node_id", "parent", "is_leaf"]
        lead = [c for c in lead if c in result.columns]
        rest = [c for c in result.columns if c not in lead]
        result = result[lead + rest]
    else:
        result = pd.DataFrame()

    dropped = (pd.concat(dropped_pieces, ignore_index=True)
               if dropped_pieces else pd.DataFrame())
    result.attrs["dropped_targets"] = (dropped.to_dict("records")
                                       if len(dropped) else [])
    if return_dropped:
        return result, dropped
    return result


def route_targets(
    targets: pd.DataFrame,
    quotas_long: pd.DataFrame,
    recipients: List[str],
    target_col: str,
    recipient_keys: Optional[Dict[str, Any]] = None,
    split: str = "base_quota",
    rollup: bool = True,
) -> pd.DataFrame:
    """
    Route target rows onto named recipient nodes in a DIFFERENT part of
    the tree (issues #25 / #32) — e.g. carry Government+EMEA targets on
    six named Enterprise_EMEA reps, split by capacity, added on top of
    their normal quota, tagged with the original segment.

    Composes with cascade_many: run the batch, take the dropped-targets
    frame (issue #26), route it::

        quotas, weights, dropped = cascade_many(..., return_dropped=True)
        gov = dropped[dropped.reason.str.contains('no rows')]
        routed = route_targets(
            gov, quotas,
            recipients=['UKI1_2', 'UKI2_1', 'NORD1_3'],
            target_col='nn_acv_target',
            recipient_keys={'regional': 'Enterprise_EMEA'},
        )
        full = pd.concat([quotas, routed], ignore_index=True)

    Mechanics (all on the BASE layer, per issue #21's contract):
      - each target row's amount is split across `recipients` by their
        `split` values read from quotas_long (their existing base_quota
        by default — "proportional to capacity"),
      - each routed node's cascaded_quota is DERIVED from its own
        existing hedge ratio (cascaded/base), never re-hedged,
      - with rollup=True (default), ancestor rows are emitted too (base
        summed up each recipient's parent chain, hedged via each
        ancestor's ratio) so the routed slice reconciles at every depth,
      - every routed row carries ALL columns of its originating target
        row (group keys, fiscal_quarter, segment tags, ...) plus a
        `routed=True` marker — so the money stays attributable to the
        original segment after concatenation.

    The result is ADDITIVE by construction: concatenate it to
    quotas_long and aggregate by node to see combined carrying totals.
    (For conditional exclusions — e.g. one rep must not carry Cloud
    products — call route_targets twice with different `targets` filters
    and `recipients` lists.)

    Parameters
    ----------
    targets : pd.DataFrame
        The target rows to route — typically (a filter of) the
        dropped-targets frame from cascade_many(return_dropped=True),
        but any frame with `target_col` works.
    quotas_long : pd.DataFrame
        Output of cascade_many (or a quotas_to_dataframe result with
        the base/audit columns): supplies recipients' split weights,
        hedge ratios, and parent chains.
    recipients : list[str]
        Leaf node_ids that will carry the money.
    target_col : str
        Column in `targets` holding the dollar amount per row.
    recipient_keys : Optional[Dict[str, Any]]
        Column=value filters applied to quotas_long first (e.g.
        {'regional': 'Enterprise_EMEA', 'fiscal_quarter': 1}). Required
        whenever a recipient matches MORE than one row (multiple
        combinations or sub-targets in the frame) — the weights and
        ratios must come from exactly one cascade snapshot.
    split : str
        "base_quota" (default) — proportional to the recipients'
        existing un-hedged quota. "equal" — even split. Any other value
        must be a numeric column in quotas_long (e.g. a metadata
        capacity column). If the chosen values sum to <= 0, an equal
        split is used with a warning.
    rollup : bool
        Emit ancestor rows (default True) so per-depth sums of the
        routed slice equal the routed amount.

    Returns
    -------
    pd.DataFrame — routed rows only (concatenate to quotas_long
    yourself), sorted by target row then depth.
    """
    if target_col not in targets.columns:
        raise ValueError(f"target_col '{target_col}' not found in targets "
                         f"columns: {list(targets.columns)}")
    if not recipients:
        raise ValueError("recipients must be a non-empty list of leaf "
                         "node_ids.")
    required = {"node_id", "parent", "depth", "base_quota",
                "cascaded_quota"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)} — pass the frame produced by "
                         f"cascade_many.")

    slice_df = quotas_long
    if recipient_keys:
        for col, val in recipient_keys.items():
            if col not in slice_df.columns:
                raise ValueError(f"recipient_keys column '{col}' not in "
                                 f"quotas_long.")
            slice_df = slice_df[slice_df[col] == val]

    node_rows = slice_df.set_index("node_id", drop=False)
    dup = node_rows.index[node_rows.index.duplicated()].unique().tolist()
    if any(r in dup for r in recipients) or (
            rollup and len(dup) > 0 and not node_rows.empty):
        raise ValueError(
            f"Nodes appear in multiple rows after filtering "
            f"(e.g. {dup[:5]}) — weights/ratios would be ambiguous. "
            f"Narrow recipient_keys to exactly one cascade (include the "
            f"group keys AND any sub-target column like fiscal_quarter)."
        )
    unknown = [r for r in recipients if r not in node_rows.index]
    if unknown:
        raise ValueError(f"recipients not found in quotas_long (after "
                         f"recipient_keys filter): {unknown}")

    # ---- Split weights from the recipients' existing rows --------------
    if split == "equal":
        weights = {r: 1.0 / len(recipients) for r in recipients}
    else:
        if split not in node_rows.columns:
            raise ValueError(f"split column '{split}' not in quotas_long.")
        vals = {}
        for r in recipients:
            v = node_rows.at[r, split]
            vals[r] = 0.0 if pd.isna(v) else float(v)
        total = sum(vals.values())
        if total <= 0:
            warnings.warn(
                f"route_targets: split column '{split}' sums to 0 across "
                f"the recipients — falling back to an equal split.",
                UserWarning, stacklevel=2,
            )
            weights = {r: 1.0 / len(recipients) for r in recipients}
        else:
            weights = {r: v / total for r, v in vals.items()}

    def _ratio(node_id: str) -> float:
        base = node_rows.at[node_id, "base_quota"]
        if pd.isna(base) or float(base) == 0.0:
            return 1.0
        return float(node_rows.at[node_id, "cascaded_quota"]) / float(base)

    def _ancestors(node_id: str) -> List[str]:
        chain, seen = [], set()
        current = node_rows.at[node_id, "parent"]
        while (current is not None and pd.notna(current)
               and current in node_rows.index and current not in seen):
            chain.append(current)
            seen.add(current)
            current = node_rows.at[current, "parent"]
        return chain

    structural = [c for c in ("depth", "level", "parent", "is_leaf")
                  if c in node_rows.columns]

    routed_rows = []
    for _, trow in targets.iterrows():
        amount = float(trow[target_col])
        base_add: Dict[str, float] = {}
        for r in recipients:
            base_add[r] = amount * weights[r]
        if rollup:
            for r in recipients:
                for anc in _ancestors(r):
                    base_add[anc] = base_add.get(anc, 0.0) + amount * weights[r]

        for node_id, badd in base_add.items():
            row = {c: trow[c] for c in targets.columns}
            row["node_id"] = node_id
            for c in structural:
                row[c] = node_rows.at[node_id, c]
            row["base_quota"] = round(badd, 2)
            row["cascaded_quota"] = round(badd * _ratio(node_id), 2)
            row["routed"] = True
            routed_rows.append(row)

    routed = pd.DataFrame(routed_rows)
    if not routed.empty and "depth" in routed.columns:
        routed = routed.sort_values(["depth", "node_id"],
                                    kind="stable").reset_index(drop=True)
    return routed
