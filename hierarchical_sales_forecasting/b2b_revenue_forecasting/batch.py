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
    metrics: Optional[Union[
        List[MetricSpec],
        Callable[[Dict[str, Any]], Optional[List[MetricSpec]]],
    ]] = None,
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
    attach_metrics: Union[bool, List[str]] = False,
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
    metrics : list[MetricSpec] | callable, optional
        Fixed metric slate used for every combination. A STATIC list is
        mutually exclusive with `suggest_config`.

        Mixed strategies (issue #35, v0.19.0): pass a CALLABLE that
        receives the combination's group-key dict and returns that
        combination's fixed slate — honored verbatim — or None to fall
        through. With `suggest_config` present, None means "use the
        suggested weights for this combination" (global or per_group
        per `weights_mode`); without it, None means the legacy
        '_Attainment' path, matching cascade_quota(metrics=None)::

            DC_ONLY = [MetricSpec('dc_seats', direction='proportional',
                                  weight=1.0)]
            cascade_many(...,
                metrics=lambda g: (DC_ONLY
                                   if g['st1_sales_type'] == 'Migration'
                                   else None),
                suggest_config=dict(...), weights_mode='per_group')

        Errors raised by the callable follow `on_error` (skip +
        dropped-targets frame, or raise).
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

        Direction-mismatch warnings in per_group mode (issue #19,
        v0.27.0): unless you explicitly set
        warn_on_direction_mismatch here, per-combination warnings are
        summarized into ONE batch-level warning ("metric X in N/M
        combinations"), with per-combo detail in
        attrs['combo_report']. Explicit True keeps per-group warnings;
        explicit False silences everything — the report column is
        populated either way.
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
        Columns carried through to the LEAF rows of quotas_long_df
        (issue #7) — and despite the name, that includes METRIC columns
        (issue #16): list `knowledge_workers` here and every leaf row
        carries its value, so pipeline-coverage / capacity analysis
        needs no re-join against the source frame. Carried columns are
        excluded from AUTO metric ingestion, but explicit MetricSpecs
        still resolve them (they remain node attributes) — so a column
        can drive the cascade AND ride along in the output; cascade
        numbers are identical either way (pinned by test). Non-leaf
        rows show NaN: these are leaf-grain values.

        Key identities (the #16 footgun): a HIERARCHY LEAF is
        identified by the deepest taxonomy column (plus any group_keys
        that exist in hierarchy_df); a CASCADE ROW is identified by
        group_keys + sub-target columns (e.g. fiscal_quarter). Columns
        that live only in target_df (a sales type not present in
        hierarchy_df) are NOT valid join keys for leaf-grain data.
    attach_metrics : bool | list[str]
        Explainability rollups (issue #49). True — every numeric
        metadata_cols column gets a ``<col>_subtree`` companion holding
        each node's subtree aggregate (sum over descendant leaves) for
        its cascade; or pass an explicit list of carried columns. Thin
        wrapper over rollup_metrics() (issue #17). The carried columns
        themselves stay leaf-grain (NaN on managers).
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

        The output also carries .attrs['cascade_row_keys'] (v0.21.0,
        issue #40): the exact columns identifying one cascade
        (group_keys + sub-target columns). apply_pins reads it when
        row_keys= is omitted, so pins Just Work on this frame even
        when metadata_cols add per-node columns.

        And .attrs['combo_report'] (v0.27.0, issue #20): one record per
        group-key combination — skipped + reason, targets_matched,
        rows_produced, n_gated_nodes, gate_relaxed, unallocated_total,
        weights_source ('fixed' | 'policy' | 'suggested_global' |
        'suggested_per_group' | 'default_attainment'),
        direction_mismatches (issue #19) and degenerate_fallback.
        Reconstruct with pd.DataFrame(q.attrs['combo_report']) — the
        batch run, auditable at a glance.
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
    if (metrics is not None and not callable(metrics)
            and suggest_config is not None):
        raise ValueError(
            "Pass either metrics=[MetricSpec, ...] (fixed slate) OR "
            "suggest_config={...} (data-driven weights) — not both. "
            "(A CALLABLE metrics= policy may coexist with suggest_config: "
            "combinations where it returns None use the suggested "
            "weights — issue #35.)"
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

    # Direction-mismatch policy (issue #19): with per_group suggestion,
    # a warning per (group x metric) floods the output. Unless the
    # caller EXPLICITLY set warn_on_direction_mismatch, per-combo runs
    # are silenced and one aggregated summary is emitted after the
    # loop; the per-combo mismatches always land in
    # attrs['combo_report'] (data, not noise).
    _user_set_dir_warn = (suggest_config is not None
                          and "warn_on_direction_mismatch"
                          in suggest_config)
    _summarize_dir = (suggest_config is not None
                      and weights_mode == "per_group"
                      and not _user_set_dir_warn)

    def _suggest(df_slice: pd.DataFrame,
                 silence_direction: bool = False,
                 ) -> Tuple[List[MetricSpec], Dict[str, Any]]:
        cfg = dict(suggest_config)
        if silence_direction:
            cfg["warn_on_direction_mismatch"] = False
        suggested, report = MetricSpec.suggest_weights(
            df_slice,
            target_column=cfg.pop("target_column"),
            candidate_metrics=cfg.pop("candidate_metrics"),
            **cfg,
        )
        return suggested, report

    def _report_flags(report: Dict[str, Any]) -> Tuple[List[str], bool]:
        mismatches = sorted(
            n for n, r in (report or {}).items()
            if r.get("direction_matches_data") is False)
        degenerate = any(r.get("degenerate") for r in (report or {}).values())
        return mismatches, degenerate

    # Global weights resolved once, if applicable
    metrics_policy = metrics if callable(metrics) else None
    global_metrics: Optional[List[MetricSpec]] = (
        None if callable(metrics) else metrics)
    _global_report: Dict[str, Any] = {}
    if suggest_config is not None and weights_mode == "global":
        global_metrics, _global_report = _suggest(hierarchy_df)

    quota_frames: List[pd.DataFrame] = []
    weight_frames: List[pd.DataFrame] = []
    failures: List[Tuple[tuple, str]] = []
    dropped_frames: List[pd.DataFrame] = []
    combo_records: List[Dict[str, Any]] = []   # issue #20

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

            # 4. Resolve weights for this combination. A callable metrics
            # policy (issue #35) wins when it returns a slate; None falls
            # through to suggest_config (if any) or the legacy path.
            combo_metrics = None
            policy_decided = False
            if metrics_policy is not None:
                combo_metrics = metrics_policy(dict(combo_dict))
                if combo_metrics is not None:
                    if not (isinstance(combo_metrics, list)
                            and all(isinstance(m, MetricSpec)
                                    for m in combo_metrics)):
                        raise ValueError(
                            f"metrics callable must return a list of "
                            f"MetricSpec or None for combination "
                            f"{combo_dict}, got "
                            f"{type(combo_metrics).__name__}."
                        )
                    policy_decided = True
            combo_report_flags: Tuple[List[str], bool] = ([], False)
            _rep: Dict[str, Any] = {}
            if policy_decided:
                weights_source = "policy"
            elif suggest_config is not None and weights_mode == "per_group":
                combo_metrics, _rep = _suggest(
                    df_slice, silence_direction=_summarize_dir
                    or ("warn_on_direction_mismatch" in suggest_config
                        and not suggest_config[
                            "warn_on_direction_mismatch"]))
                combo_report_flags = _report_flags(_rep)
                weights_source = "suggested_per_group"
            elif not policy_decided:
                combo_metrics = global_metrics
                if suggest_config is not None:
                    weights_source = "suggested_global"
                    combo_report_flags = _report_flags(_global_report)
                elif global_metrics is not None:
                    weights_source = "fixed"
                else:
                    weights_source = "default_attainment"

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

            # Record the weights actually used (once per combination) —
            # the AUTHORITATIVE record of what drove the run (issue
            # #50): blend slate (or the legacy default), gate slate,
            # provenance, and per-metric suggest-fallback flags. Never
            # re-derive by re-invoking your callables.
            if combo_metrics:
                wdf = MetricSpec.normalized_weights(combo_metrics)
            else:                     # legacy '_Attainment' default path
                wdf = pd.DataFrame([{
                    "metric": "_Attainment",
                    "direction": "proportional",
                    "input_weight": 1.0, "normalized_share": 1.0,
                    "active": True,
                }])
            wdf["role"] = "blend"
            wdf["gate_threshold"] = None
            wdf["gate_mode"] = None
            if combo_gates:
                gdf = pd.DataFrame([{
                    "metric": g.name, "direction": g.direction,
                    "input_weight": None, "normalized_share": None,
                    "active": True, "role": "gate",
                    "gate_threshold": g.gate_threshold,
                    "gate_mode": g.gate_mode,
                } for g in combo_gates])
                wdf = pd.concat([wdf, gdf], ignore_index=True)
            wdf["weights_source"] = weights_source
            deg_map: Dict[str, bool] = {}
            if weights_source == "suggested_per_group":
                deg_map = {n: bool(r.get("degenerate"))
                           for n, r in (_rep or {}).items()}
            elif weights_source == "suggested_global":
                deg_map = {n: bool(r.get("degenerate"))
                           for n, r in (_global_report or {}).items()}
            wdf["degenerate"] = (wdf["metric"].map(deg_map).fillna(False)
                                 if deg_map else False)
            for k, v in combo_dict.items():
                wdf[k] = v
            weight_frames.append(wdf)

            # 5. Cascade every sub-target row against the prepared group
            _rows_produced = 0
            _gated_union: set = set()
            _relaxed_any = False
            _unallocated_total = 0.0
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
                # Per-combination bookkeeping (issue #20)
                _rows_produced += len(qdf)
                _gated_union |= set(cascader.gated_nodes)
                _relaxed_any = _relaxed_any or bool(
                    cascader.gate_relaxed_nodes)
                _unallocated_total += float(cascader.unallocated or 0.0)

            combo_records.append({
                **combo_dict,
                "skipped": False, "reason": None,
                "targets_matched": int(len(combo_targets)),
                "rows_produced": _rows_produced,
                "n_gated_nodes": len(_gated_union),
                "gate_relaxed": _relaxed_any,
                "unallocated_total": round(_unallocated_total, 2),
                "weights_source": weights_source,
                "direction_mismatches": combo_report_flags[0],
                "degenerate_fallback": combo_report_flags[1],
            })

        except Exception as exc:  # noqa: BLE001 — reported per policy
            if on_error == "raise":
                raise
            failures.append((combo_vals, f"{type(exc).__name__}: {exc}"))
            combo_records.append({
                **combo_dict,
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}",
                "targets_matched": int(len(combo_targets)),
                "rows_produced": 0, "n_gated_nodes": 0,
                "gate_relaxed": False, "unallocated_total": 0.0,
                "weights_source": None, "direction_mismatches": [],
                "degenerate_fallback": False,
            })
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
    # Cascade-identity stamp (issue #40): the columns that identify ONE
    # cascade in this frame. apply_pins(row_keys=None) reads this instead
    # of inferring keys from "every non-structural column" — which broke
    # silently when per-node columns (metadata_cols) were present.
    # Stored as a plain list of strings (attrs-concat safe).
    quotas_long.attrs["cascade_row_keys"] = list(group_keys) + \
        [c for c in passthrough_cols if c in quotas_long.columns]
    # Per-combination diagnostics (issue #20). Stored as RECORDS
    # (attrs-concat safe); reconstruct with
    # pd.DataFrame(quotas_long.attrs['combo_report']).
    quotas_long.attrs["combo_report"] = combo_records

    # Aggregated direction-mismatch summary (issue #19): one warning
    # for the whole batch instead of one per (group x metric).
    if _summarize_dir:
        n_suggested = sum(1 for r in combo_records
                          if r["weights_source"] == "suggested_per_group")
        counts: Dict[str, int] = {}
        for r in combo_records:
            for m in r["direction_mismatches"]:
                counts[m] = counts.get(m, 0) + 1
        if counts:
            detail = ", ".join(f"'{m}' in {n}/{n_suggested} combinations"
                               for m, n in sorted(counts.items()))
            warnings.warn(
                f"suggest_weights direction mismatches across the batch: "
                f"{detail}. Directions were kept as declared. Per-combo "
                f"detail in quotas_long.attrs['combo_report'] "
                f"('direction_mismatches'). Set "
                f"warn_on_direction_mismatch in suggest_config to True "
                f"for per-group warnings or False to silence.",
                UserWarning, stacklevel=2,
            )

    # Explainability rollups (issue #49) — see rollup_metrics (#17).
    if attach_metrics:
        if attach_metrics is True:
            roll_cols = [c for c in (metadata_cols or [])
                         if c in quotas_long.columns
                         and pd.api.types.is_numeric_dtype(
                             quotas_long[c])]
            if not roll_cols:
                raise ValueError(
                    "attach_metrics=True needs numeric metric columns "
                    "carried via metadata_cols=[...] (v0.8.0; carries "
                    "metric values too — issue #16).")
        else:
            roll_cols = list(attach_metrics)
        stamped = dict(quotas_long.attrs)
        quotas_long = rollup_metrics(quotas_long, roll_cols)
        quotas_long.attrs.update(stamped)

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
    # Tag EVERY row with the root-target key columns (issues #17/#49):
    # deeper transitions only knew their immediate parent, so columns
    # like the region key were NaN below the first transition. Inherit
    # them down the parent chain (child-uniqueness is already
    # validated, so node_id -> value is unambiguous), which makes the
    # cascade-identity stamp below valid at every depth.
    root_keys = [c for c in root_targets.columns
                 if c != target_col and c in result.columns]
    if len(result):
        for c in root_keys:
            for _ in range(max(1, result["depth"].nunique())):
                missing = result[c].isna()
                if not missing.any():
                    break
                val_of = (result.loc[~result[c].isna()]
                          .set_index("node_id")[c].to_dict())
                result.loc[missing, c] = result.loc[missing,
                                                    "parent"].map(val_of)
    # Cascade-identity stamp (issues #40/#17): one cascade per original
    # root-target row; per-node parent columns (e.g. the intermediate
    # level columns) must never enter the key.
    result.attrs["cascade_row_keys"] = root_keys
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


_ROLLUP_AGGS = ("sum", "mean", "max", "min")


def _cascade_key_context(df, row_keys, exclude_cols, caller):
    """Shared cascade-identity machinery for frame-level tools
    (rollup_metrics / reconcile): key resolution (explicit ->
    attrs['cascade_row_keys'] -> inference) + the #40 orphan guard.
    Returns (keys, key_of, row_ix, child_ix)."""
    from b2b_revenue_forecasting.pins import _STRUCTURAL_COLS

    if row_keys is not None:
        keys = list(row_keys)
    else:
        stamped = df.attrs.get("cascade_row_keys")
        if (isinstance(stamped, (list, tuple))
                and all(isinstance(c, str) and c in df.columns
                        for c in stamped)):
            keys = list(stamped)
        else:
            keys = [c for c in df.columns
                    if c not in _STRUCTURAL_COLS
                    and c not in (exclude_cols or [])
                    and not c.endswith("_subtree")]
    key_of = (df[keys].apply(
                  lambda r: tuple(None if pd.isna(v) else v for v in r),
                  axis=1) if keys
              else pd.Series([()] * len(df), index=df.index))

    row_ix, child_ix = {}, {}
    for idx in df.index:
        k = key_of.at[idx]
        row_ix[(k, df.at[idx, "node_id"])] = idx
        child_ix.setdefault((k, df.at[idx, "parent"]), []).append(idx)

    all_node_ids = set(df["node_id"])
    orphans = [idx for idx in df.index
               if pd.notna(df.at[idx, "parent"])
               and df.at[idx, "parent"] in all_node_ids
               and (key_of.at[idx], df.at[idx, "parent"]) not in row_ix]
    if orphans:
        poison = set()
        for idx in orphans[:20]:
            parent_rows = df.index[df["node_id"] == df.at[idx, "parent"]]
            for col in keys:
                v = df.at[idx, col]
                if all(not (df.at[p, col] == v
                            or (pd.isna(df.at[p, col]) and pd.isna(v)))
                       for p in parent_rows):
                    poison.add(col)
        raise ValueError(
            f"{caller}: {len(orphans)} row(s) have a parent that exists "
            f"in the frame but NOT under the same cascade key — the "
            f"row_keys are wrong. Keys in use: {keys}. Per-node columns "
            f"poisoning the identity: {sorted(poison)}. Pass row_keys= "
            f"(group keys + sub-target columns only).")
    return keys, key_of, row_ix, child_ix


def rollup_metrics(
    quotas_long: pd.DataFrame,
    metrics: Union[str, List[str]],
    agg: str = "sum",
    row_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Roll leaf-grain metric columns UP the tree on a cascade output
    (issues #17 / #49): one new ``<metric>_subtree`` column per metric,
    holding each node's subtree aggregate for that cascade.

    "Does this team's pipeline cover its quota?" becomes::

        out = rollup_metrics(quotas_long, ['pipeline'])
        out['coverage'] = out['pipeline_subtree'] / out['base_quota']

    Semantics
    ---------
    - Aggregation runs over DESCENDANT LEAVES (frame-local: nodes with
      no children in the same cascade), so ``agg='mean'``/``'max'`` are
      well-defined — a manager's value is the mean/max of its leaf
      values, never a mean of means. ``'sum'`` (default) matches the
      cascader's own leaf-sum rollup semantics. Leaf rows carry their
      own value.
    - The source columns are left untouched: carried metric columns
      (``metadata_cols``) stay leaf-grain with NaN on managers, per the
      v0.19.2 contract. NaN leaf values are skipped (all-NaN subtree
      -> NaN).
    - Cascade identity comes from ``.attrs['cascade_row_keys']``
      (stamped by cascade_many/cascade_levels), else ``row_keys=``,
      else inference — and, as in apply_pins (#40), a broken key set
      cannot corrupt silently: if any row's parent exists in the frame
      but not under the same cascade key, this raises naming the
      per-node columns poisoning the identity.

    Parameters
    ----------
    quotas_long : pd.DataFrame
        cascade_many / cascade_levels output (or any long frame with
        node_id, parent + the metric columns on leaf rows).
    metrics : str | list[str]
        Column(s) to roll up. Must exist and be numeric.
    agg : str
        'sum' (default), 'mean', 'max', or 'min' — over descendant
        leaf values.
    row_keys : list[str], optional
        Columns identifying ONE cascade; rarely needed thanks to the
        attrs stamp.

    Returns
    -------
    pd.DataFrame — a copy with the ``<metric>_subtree`` columns added.
    """
    from b2b_revenue_forecasting.pins import _STRUCTURAL_COLS

    if isinstance(metrics, str):
        metrics = [metrics]
    if not metrics:
        raise ValueError("rollup_metrics: pass at least one metric "
                         "column.")
    if agg not in _ROLLUP_AGGS:
        raise ValueError(f"agg must be one of {_ROLLUP_AGGS}, "
                         f"got '{agg}'.")
    required = {"node_id", "parent"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)}.")
    absent = [m for m in metrics if m not in quotas_long.columns]
    if absent:
        raise ValueError(
            f"rollup_metrics: {absent} not in quotas_long. Carry metric "
            f"columns onto leaf rows with cascade_many(metadata_cols="
            f"[...]) (v0.8.0; despite the name it carries metric "
            f"values too — issue #16).")
    non_num = [m for m in metrics
               if not pd.api.types.is_numeric_dtype(quotas_long[m])]
    if non_num:
        raise ValueError(f"rollup_metrics: {non_num} are not numeric.")

    df = quotas_long.copy()
    keys, key_of, row_ix, child_ix = _cascade_key_context(
        df, row_keys, exclude_cols=list(metrics), caller="rollup_metrics")

    # Descendant-leaf row indices per row, memoized per key group
    memo: Dict[Any, List[Any]] = {}

    def _leaf_rows(idx) -> List[Any]:
        if idx in memo:
            return memo[idx]
        k = key_of.at[idx]
        kids = child_ix.get((k, df.at[idx, "node_id"]), [])
        out = ([idx] if not kids
               else [li for c in kids for li in _leaf_rows(c)])
        memo[idx] = out
        return out

    for m in metrics:
        col = f"{m}_subtree"
        df[col] = [getattr(df.loc[_leaf_rows(idx), m], agg)()
                   for idx in df.index]
    return df


def reconcile(
    quotas_long: pd.DataFrame,
    hedge: Any = None,
    tolerance: float = 0.05,
    ratio_tolerance: float = 1e-4,
    row_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    One-call validator for cascade outputs (issue #46): per-parent
    BASE-layer conservation and, when `hedge` is given, per-node hedge
    identities — the checks everyone hand-writes after a run
    (d0==d1 sums, d2==base x 1.05, d3==base x 1.155, ...).

        frame = reconcile(quotas_long,
                          hedge=HedgeByDepth(from_leaves={1: 1.10,
                                                          2: 1.05}))
        assert frame.ok.all()

    Checks (one tidy row each; `check` column says which):
      conservation — every parent's base_quota vs the sum of its
        children's base_quota, per cascade, within `tolerance` dollars.
      hedge_ratio  — each node's actual cascaded/base ratio vs the
        expected CUMULATIVE ratio from `hedge`, within
        `ratio_tolerance` (relative). Rows with base <= 0 are skipped
        (a gated node has no ratio).

    `hedge` accepts:
      float           — flat multiplier; expected = hedge ** depth.
      dict            — {depth: cumulative_ratio}, exactly the
                        hand-written identity list (missing depths
                        default to 1.0).
      HedgeByDepth    — resolved per cascade with the REAL
                        HedgeByDepth.resolve() on a graph rebuilt from
                        the frame's parent links, manager multipliers
                        compounded root-down — so the expectation
                        cannot drift from what the engine does.

    Post-edit frames reconcile too: apply_pins / redistribute /
    concentrate preserve each row's original hedge ratio (the #21
    contract) and conserve parents, so a clean edit stays clean here.

    Returns a tidy DataFrame: cascade key columns, node_id, parent,
    depth, check, expected, actual, delta, ok. Emits ONE summary
    warning when any ok=False; silent otherwise.
    """
    from b2b_revenue_forecasting.quota_cascader import HedgeByDepth

    required = {"node_id", "parent", "base_quota"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)}.")
    if hedge is not None and "cascaded_quota" not in quotas_long.columns:
        raise ValueError("hedge checks need a cascaded_quota column.")
    if hedge is not None and not isinstance(hedge, (int, float, dict,
                                                    HedgeByDepth)):
        raise ValueError("hedge must be a float, a {depth: cum_ratio} "
                         "dict, or a HedgeByDepth spec.")

    df = quotas_long
    keys, key_of, row_ix, child_ix = _cascade_key_context(
        df, row_keys, exclude_cols=[], caller="reconcile")

    groups: Dict[Any, List[Any]] = {}
    for idx in df.index:
        groups.setdefault(key_of.at[idx], []).append(idx)

    rows: List[Dict[str, Any]] = []
    for k, idxs in groups.items():
        kd = dict(zip(keys, k)) if keys else {}

        # ---- conservation: parent base == sum(children base) --------
        for idx in idxs:
            node = df.at[idx, "node_id"]
            kids = child_ix.get((k, node), [])
            if not kids:
                continue
            expected = float(df.at[idx, "base_quota"])
            actual = float(sum(df.at[c, "base_quota"] for c in kids))
            rows.append({**kd, "node_id": node,
                         "parent": df.at[idx, "parent"],
                         "depth": df.at[idx, "depth"]
                         if "depth" in df.columns else None,
                         "check": "conservation",
                         "expected": round(expected, 2),
                         "actual": round(actual, 2),
                         "delta": round(actual - expected, 2),
                         "ok": abs(actual - expected) <= tolerance})

        # ---- hedge identities ----------------------------------------
        if hedge is None:
            continue
        if isinstance(hedge, HedgeByDepth):
            import networkx as nx
            g = nx.DiGraph()
            for idx in idxs:
                g.add_node(df.at[idx, "node_id"])
            for idx in idxs:
                p_ = df.at[idx, "parent"]
                if pd.notna(p_) and (k, p_) in row_ix:
                    g.add_edge(p_, df.at[idx, "node_id"])
            mult = hedge.resolve(g)
            cum: Dict[str, float] = {}

            def _cum(node):
                if node in cum:
                    return cum[node]
                preds = [df.at[row_ix[(k, node)], "parent"]]
                p_ = preds[0]
                if pd.isna(p_) or (k, p_) not in row_ix:
                    cum[node] = 1.0
                else:
                    cum[node] = _cum(p_) * float(mult.get(p_, 1.0))
                return cum[node]

        for idx in idxs:
            node = df.at[idx, "node_id"]
            base = float(df.at[idx, "base_quota"])
            if base <= 0:
                continue                      # gated/zero: no ratio
            depth = (int(df.at[idx, "depth"])
                     if "depth" in df.columns else None)
            if isinstance(hedge, HedgeByDepth):
                expected = _cum(node)
            elif isinstance(hedge, dict):
                expected = float(hedge.get(depth, 1.0))
            else:
                expected = float(hedge) ** (depth or 0)
            actual = float(df.at[idx, "cascaded_quota"]) / base
            ok = (abs(actual / expected - 1.0) <= ratio_tolerance
                  if expected != 0 else actual == 0)
            rows.append({**kd, "node_id": node,
                         "parent": df.at[idx, "parent"],
                         "depth": depth, "check": "hedge_ratio",
                         "expected": round(expected, 6),
                         "actual": round(actual, 6),
                         "delta": round(actual - expected, 6),
                         "ok": ok})

    frame = pd.DataFrame(rows)
    n_bad = int((~frame["ok"]).sum()) if len(frame) else 0
    if n_bad:
        warnings.warn(
            f"reconcile: {n_bad} check(s) failed (of {len(frame)}) — "
            f"filter the returned frame with ~frame.ok for the "
            f"violations.",
            UserWarning, stacklevel=2)
    return frame
