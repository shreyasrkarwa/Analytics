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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    gate_metrics: Optional[List[MetricSpec]] = None,
    hedge_multiplier: Union[float, Dict[str, float]] = 1.0,
    suggest_config: Optional[Dict[str, Any]] = None,
    weights_mode: str = "global",
    level_names: Optional[List[str]] = None,
    brand_new_col: Optional[str] = None,
    metadata_cols: Optional[List[str]] = None,
    on_error: str = "skip",
    on_collision: str = "suffix",
    verbose: bool = False,
    **cascade_kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    gate_metrics : list[MetricSpec], optional
        Gate metrics, passed straight to cascade_quota. All
        gate_fallback semantics (v0.5.0) apply; override the fallback
        via **cascade_kwargs (e.g. gate_fallback="strand_at_root").
    hedge_multiplier : float | dict
        Passed to cascade_quota. The un-hedged base layer is always
        included in the output (base_quota column).
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
                    gate_metrics=gate_metrics,
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

    return quotas_long, weights_long
