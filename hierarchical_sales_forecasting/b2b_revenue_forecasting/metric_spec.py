"""
MetricSpec: a single metric's contribution to multi-metric quota cascading.

A MetricSpec tells the QuotaCascader how to fold one historical signal
(e.g., NetNewACV, CloudSeats, ExpansionSpent, or ANY metric the analyst
chooses to track) into the share-of-parent calculation when distributing
a target down the org tree.

Each spec carries:
  - name         : a friendly identifier — any string, used as the suffix
                   when building default column names "Q1_<name>", ...
  - direction    : "proportional" (more of this metric -> more quota) or
                   "inverse"      (more of this metric -> less quota).
                   ** This is always a USER decision — the package never
                   overrides direction based on data. Domain knowledge
                   trumps statistical sign. **
  - weight       : relative importance vs other metrics (any non-negative
                   float; weights are normalized to sum=1 at cascade time)
  - lookback     : number of historical quarters to aggregate
  - columns      : explicit list of dataframe column names; if None, we
                   default to [f"Q{i}_{name}" for i in 1..lookback]
  - aggregation  : "sum" | "mean" | "last" — how to fold lookback quarters
  - impute_zeros : if True, zero values within the lookback are imputed
                   with the node's own non-zero average (helps partial-
                   history ICs). Auto-disabled at runtime when values look
                   boolean (all in {0, 1}) — see _aggregate_node_metric.

The helper MetricSpec.suggest_weights() inspects historical data and
proposes a WEIGHT for each user-provided MetricSpec by correlating each
metric with a designated target column. Direction is preserved exactly as
the user declared it — magnitude of the Pearson correlation becomes the
suggested weight. If the sign of the correlation contradicts the user's
direction (e.g., the user said "inverse" but the data shows a positive
correlation), the suggester emits a warning so the user can re-examine
their assumption — but the user's direction is NOT overridden.

For exploratory use, MetricSpec.suggest_directions_and_weights() does
infer both direction and weight from data; this is meant for sanity
checks, not production planning.

Supports ANY metric name (free-form strings) and ANY numeric data type
including booleans (True/False are aggregated as 1/0).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union, Any
import warnings
import pandas as pd
import numpy as np

# Allowed values, kept as constants so callers can introspect them.
DIRECTIONS = ("proportional", "inverse")
AGGREGATIONS = ("sum", "mean", "last")
# Gate PASS predicates (issue #9). A node is GATED when it fails its
# gate's predicate against the aggregated (leaf-summed) value:
#   "gt"     pass iff value >  gate_threshold   (default; pre-0.9.0 behavior)
#   "ge"     pass iff value >= gate_threshold
#   "lt"     pass iff value <  gate_threshold   (gate values that are TOO HIGH)
#   "le"     pass iff value <= gate_threshold
#   "truthy" pass iff bool(value)               (threshold ignored)
GATE_MODES = ("gt", "ge", "lt", "le", "truthy")


@dataclass
class MetricSpec:
    """
    Specification for one metric's role in multi-metric cascading.

    Parameters
    ----------
    name : str
        Friendly identifier for the metric (any string — e.g. "NetNewACV",
        "CloudSeats", "Renewals_Caught_Up", "Has_Cert_Coverage"). Used as
        the suffix for default column naming: "Q1_<name>", "Q2_<name>", ...
    direction : str
        "proportional" — higher values pull MORE quota toward the node.
        "inverse"      — higher values pull LESS quota toward the node.
        Always a user decision; never inferred without explicit opt-in.
    weight : float
        Non-negative importance weight. Weights across all metrics are
        normalized to sum=1 at cascade time, so absolute scale doesn't
        matter — only the ratio between metrics.

        How weights become influence (issue #11)
        ----------------------------------------
        Normalization happens across ACTIVE metrics only (weight > 0);
        inactive metrics contribute exactly 0. A metric's real influence
        is weight / sum(active weights), NOT its raw value:

            raw weights [1.0, 0.5, 0.0]  ->  shares [66.7%, 33.3%, 0%]
            raw weights [1.0, 0.98, 0.4, 0.067]
                -> 0.067 / 2.447 = 2.7% influence (not 6.7%!)

        Inspect the actual shares with MetricSpec.normalized_weights(specs)
        or cascader.weights_report (auto-printed before every verbose
        multi-metric cascade).
    lookback : int
        Number of historical quarters to aggregate. Default 4.
    columns : Optional[List[str]]
        Explicit list of dataframe / node-attribute column names for this
        metric. If None, defaults to [f"Q{i}_{name}" for i in 1..lookback].
        Use this when your column naming doesn't follow the Qi_<name>
        convention (e.g., a single LTM column).

        The name/columns contract (issue #6)
        ------------------------------------
        `name` is the metric's identity; `columns` is where its VALUES
        live. `resolved_columns()` is what the cascader actually reads:
          1. `columns` if you set it — always wins, end to end
             (including on specs returned by suggest_weights).
          2. Otherwise the Qi_<name> convention
             (Q1_<name> ... Q<lookback>_<name>).
          3. Since v0.7.1 the CASCADER adds a runtime fallback: if
             `columns` is unset and none of the Qi_<name> attributes
             exist on a leaf, it reads the attribute named exactly
             `<name>`. So a spec named "knowledge_workers" finds a
             plain `knowledge_workers` column with no extra code.
        Note: the `column=` field accepted by suggest_weights is the
        CORRELATION column (often an analysis-only helper like
        "CloudSeats_4Q_sum") and is intentionally never copied into
        `columns`.
    aggregation : str
        How to fold the lookback columns: "sum" (default), "mean", or "last".
    impute_zeros : bool
        If True (default), zero values within the lookback are imputed with
        the node's own non-zero average — mirroring the partial-history
        handling in the legacy single-metric path. Set False for metrics
        where zero is a meaningful value (rates, counts that legitimately
        can be zero, etc.). At runtime this is auto-disabled for any node-
        metric whose values are all booleans / 0-or-1, regardless of this
        setting.
    """
    name: str
    direction: str = "proportional"
    weight: float = 1.0
    lookback: int = 4
    columns: Optional[List[str]] = None
    aggregation: str = "sum"
    impute_zeros: bool = True
    # Only meaningful when this spec is passed to cascade_quota(gate_metrics=...).
    #
    # The exact gate predicate (issue #9): a node PASSES the gate iff
    #
    #     gate_mode == "gt":     aggregated_value >  gate_threshold
    #     gate_mode == "ge":     aggregated_value >= gate_threshold
    #     gate_mode == "lt":     aggregated_value <  gate_threshold
    #     gate_mode == "le":     aggregated_value <= gate_threshold
    #     gate_mode == "truthy": bool(aggregated_value)   (threshold ignored)
    #
    # where aggregated_value is the metric rolled up from leaves (sum for
    # non-leaf nodes). Nodes that FAIL are gated: excluded from the cascade
    # with quota = 0 (subject to cascade_quota's gate_fallback).
    #
    # Defaults (gate_mode="gt", gate_threshold=0.0) reproduce the original
    # "must have at least some of this" semantics exactly ("0 unmigrated
    # seats => 0 migration quota"). Examples:
    #   at least 5 seats:            gate_threshold=5, gate_mode="ge"
    #   boolean entitlement flag:    gate_mode="truthy"
    #   exclude churn-heavy (>100):  gate_threshold=100, gate_mode="le"
    #
    # Note for "lt"/"le" gates: leaf sums GROW as you go up the tree, so a
    # parent can fail while its children pass — the v0.5.0 gate_fallback
    # machinery handles any resulting fully-gated levels.
    gate_threshold: float = 0.0
    gate_mode: str = "gt"

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"MetricSpec.direction must be one of {DIRECTIONS}, "
                f"got '{self.direction}'"
            )
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"MetricSpec.aggregation must be one of {AGGREGATIONS}, "
                f"got '{self.aggregation}'"
            )
        if self.weight < 0:
            raise ValueError(
                f"MetricSpec.weight must be non-negative, got {self.weight}"
            )
        if self.lookback < 1:
            raise ValueError(
                f"MetricSpec.lookback must be >= 1, got {self.lookback}"
            )
        if self.gate_mode not in GATE_MODES:
            raise ValueError(
                f"MetricSpec.gate_mode must be one of {GATE_MODES}, "
                f"got '{self.gate_mode}'"
            )

    def resolved_columns(self) -> List[str]:
        """
        Return the column names this spec reads from: `columns` when set,
        else the Qi_<name> convention (Q1_<name> ... Q<lookback>_<name>).

        The cascader additionally falls back to the attribute named
        exactly `<name>` when `columns` is unset and none of these
        conventional columns exist on a leaf (issue #6) — see the class
        docstring for the full name/columns contract.
        """
        if self.columns is not None:
            return list(self.columns)
        return [f"Q{i}_{self.name}" for i in range(1, self.lookback + 1)]

    # ------------------------------------------------------------------
    # Normalized-weights view (for explaining the cascade to stakeholders)
    # ------------------------------------------------------------------
    @classmethod
    def normalized_weights(cls, specs: List["MetricSpec"]) -> pd.DataFrame:
        """
        Return a DataFrame showing each metric's input weight, the
        normalized share it actually contributes to the cascade, its
        direction, and whether it's active (weight > 0).

        The cascader normalizes weights to sum=1 across active metrics at
        cascade time — so a slate of [1.0, 0.98, 0.067, 0.0, 0.3] does NOT
        mean "0.067 is 6.7% of the influence." It means "0.067 / 2.347 =
        2.9% of the influence." This helper makes that visible so analysts
        can explain to stakeholders exactly how much each signal pulls the
        allocation.

        Columns of the returned DataFrame:
          metric           — MetricSpec.name
          direction        — 'proportional' or 'inverse'
          input_weight     — what the user / suggester provided
          normalized_share — actual share of cascade influence (sums to 1
                             across active rows)
          active           — True iff input_weight > 0
        """
        active = [s for s in specs if s.weight > 0]
        total = sum(s.weight for s in active) if active else 1.0
        active_names = {s.name for s in active}

        rows = []
        for s in specs:
            is_active = s.name in active_names
            rows.append({
                "metric": s.name,
                "direction": s.direction,
                "input_weight": float(s.weight),
                "normalized_share": (s.weight / total) if is_active else 0.0,
                "active": is_active,
            })
        return pd.DataFrame(rows)

    @classmethod
    def format_normalized_weights(cls, specs: List["MetricSpec"],
                                  title: str = "Multi-metric cascade — normalized weights:") -> str:
        """
        Pretty-print the normalized-weights table as a string. Mirrors what
        QuotaCascader.cascade_quota(verbose=True) emits before the cascade
        runs. Useful when you want the same view in a notebook or report.
        """
        report = cls.normalized_weights(specs)
        if report.empty:
            return f"{title}\n  (no metrics declared)"

        # Column widths
        name_w = max(report["metric"].astype(str).map(len).max(), len("metric"))
        dir_w = max(report["direction"].astype(str).map(len).max(), len("direction"))

        lines = [title]
        lines.append(
            f"  {'metric':<{name_w}}  {'direction':<{dir_w}}  "
            f"{'input_weight':>12}  {'normalized_share':>17}  active"
        )
        for _, r in report.iterrows():
            active_str = "yes" if r["active"] else "no"
            share = r["normalized_share"]
            share_str = f"{share*100:>15.2f}%" if r["active"] else f"{'—':>16}"
            lines.append(
                f"  {r['metric']:<{name_w}}  {r['direction']:<{dir_w}}  "
                f"{r['input_weight']:>12.3f}  {share_str:>17}  {active_str}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helper used by both weight suggesters
    # ------------------------------------------------------------------
    @classmethod
    def _normalize_candidate(
        cls,
        entry: Union[Dict[str, Any], "MetricSpec"],
        default_lookback: int,
    ) -> Tuple["MetricSpec", str, Optional[str]]:
        """
        Convert one candidate-metric entry into:
          (MetricSpec skeleton, correlation column, user-provided direction or None)

        Bare strings are NOT accepted — direction is required.
        """
        if isinstance(entry, MetricSpec):
            # User has already constructed a full spec — direction is whatever
            # they set on it (defaulted to 'proportional' if they didn't).
            return entry, _correlation_column_for(entry), entry.direction

        if isinstance(entry, dict):
            user_direction = entry.get("direction")
            if user_direction is None:
                raise ValueError(
                    f"candidate_metrics entry for '{entry.get('name', '?')}' "
                    f"is missing the required 'direction' field. Pass "
                    f"direction='proportional' or direction='inverse'. "
                    f"(Use suggest_directions_and_weights() instead if you "
                    f"intentionally want the package to guess direction "
                    f"from the data.)"
                )
            spec_skeleton = MetricSpec(
                name=entry["name"],
                direction=user_direction,
                lookback=entry.get("lookback", default_lookback),
                columns=entry.get("columns"),
                aggregation=entry.get("aggregation", "sum"),
                impute_zeros=entry.get("impute_zeros", True),
            )
            corr_col = entry.get("column") or _correlation_column_for(spec_skeleton)
            return spec_skeleton, corr_col, user_direction

        raise TypeError(
            f"candidate_metrics entries must be MetricSpec or dict, "
            f"got {type(entry).__name__}: {entry!r}"
        )

    # ------------------------------------------------------------------
    # Weight suggestion (PRIMARY API — direction is user-provided)
    # ------------------------------------------------------------------
    @classmethod
    def suggest_weights(
        cls,
        df: pd.DataFrame,
        target_column: str,
        candidate_metrics: List[Union[Dict[str, Any], "MetricSpec"]],
        default_lookback: int = 4,
        min_correlation: float = 0.05,
        warn_on_direction_mismatch: bool = True,
    ) -> Tuple[List["MetricSpec"], Dict[str, Dict[str, Any]]]:
        """
        Suggest WEIGHTS (only) for user-declared MetricSpecs by correlating
        each metric with a designated target column.

        Direction MUST be set on every candidate (either via a MetricSpec
        instance with .direction, or via a dict with a 'direction' key).
        Direction is a domain decision and is preserved exactly. If the sign
        of the correlation contradicts the user's direction, a warning is
        emitted (unless warn_on_direction_mismatch=False) — but the user's
        direction is never overridden.

        Parameters
        ----------
        df : pd.DataFrame
            Historical data, one row per IC (or aggregated unit). Must
            contain target_column and a column for each candidate metric.
        target_column : str
            The metric being cascaded (or any reference signal you want to
            weight other metrics against). E.g., "NetNewACV_4Q_sum".
        candidate_metrics : list[MetricSpec | dict]
            Each entry must carry an explicit direction. Dict shape:
              {"name": str, "direction": "proportional" | "inverse",
               "column": <correlation column>, "lookback": int (optional),
               "columns": [...] (optional), "aggregation": str (optional),
               "impute_zeros": bool (optional)}
            If 'column' is omitted, the function infers it from
            'columns'[0] when len==1, else from 'name'.
        default_lookback : int
            Lookback used for any MetricSpec that doesn't carry one.
        min_correlation : float
            Metrics with |correlation| < min_correlation get weight 0 — they
            don't influence cascading unless the user overrides.
        warn_on_direction_mismatch : bool
            Emit a UserWarning when the sign of the data correlation
            contradicts the user-declared direction. Default True.

        Returns
        -------
        (suggestions, report)
            suggestions : list[MetricSpec] — direction preserved, weight set
                          from |correlation|, ready to pass to
                          cascade_quota(metrics=...).
            report      : dict[name -> {correlation, n_observations,
                          direction (user's), weight, rationale,
                          direction_matches_data: bool}]
        """
        if target_column not in df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in dataframe. "
                f"Available columns: {list(df.columns)}"
            )

        suggestions: List[MetricSpec] = []
        report: Dict[str, Dict[str, Any]] = {}
        target_series = pd.to_numeric(df[target_column], errors="coerce")

        for entry in candidate_metrics:
            spec, corr_col, user_direction = cls._normalize_candidate(
                entry, default_lookback
            )

            corr, n, status = _pairwise_correlation(df, target_series, corr_col)

            if status != "ok":
                spec.weight = 0.0  # respect direction; just zero out influence
                report[spec.name] = {
                    "correlation": corr,
                    "n_observations": n,
                    "direction": user_direction,
                    "weight": 0.0,
                    "direction_matches_data": None,
                    "rationale": status,
                }
                suggestions.append(spec)
                continue

            magnitude = abs(corr)
            weight = magnitude if magnitude >= min_correlation else 0.0
            data_direction = "proportional" if corr >= 0 else "inverse"
            direction_matches = (data_direction == user_direction)

            if not direction_matches and warn_on_direction_mismatch and weight > 0:
                warnings.warn(
                    f"MetricSpec '{spec.name}': user-declared direction is "
                    f"'{user_direction}', but the data shows correlation "
                    f"{corr:+.3f} with '{target_column}' (suggests "
                    f"'{data_direction}'). Direction kept as user requested. "
                    f"Pass warn_on_direction_mismatch=False to silence.",
                    UserWarning,
                    stacklevel=2,
                )

            if weight == 0.0:
                rationale = (
                    f"|correlation| = {magnitude:.3f} is below "
                    f"min_correlation={min_correlation} — weight set to 0."
                )
            else:
                rationale = (
                    f"correlation with '{target_column}' = {corr:+.3f} "
                    f"(n={n}). User-declared direction='{user_direction}'; "
                    f"data sign {'agrees' if direction_matches else 'DISAGREES'}. "
                    f"Weight = |corr| = {weight:.3f}."
                )

            spec.weight = weight
            report[spec.name] = {
                "correlation": corr,
                "n_observations": n,
                "direction": user_direction,
                "weight": weight,
                "direction_matches_data": direction_matches,
                "rationale": rationale,
            }
            suggestions.append(spec)

        return suggestions, report

    # ------------------------------------------------------------------
    # Exploratory: infer BOTH direction and weight (sanity-check helper)
    # ------------------------------------------------------------------
    @classmethod
    def suggest_directions_and_weights(
        cls,
        df: pd.DataFrame,
        target_column: str,
        candidate_metrics: List[Union[str, Dict[str, Any], "MetricSpec"]],
        default_lookback: int = 4,
        min_correlation: float = 0.05,
    ) -> Tuple[List["MetricSpec"], Dict[str, Dict[str, Any]]]:
        """
        Exploratory helper — infer BOTH direction (from correlation sign)
        AND weight (from correlation magnitude). Use for sanity checks
        before locking in your domain-driven directions.

        Accepts the same candidate_metrics shapes as suggest_weights, plus
        bare strings (which default to direction='proportional' before
        being overridden by the data sign).

        Returns the same (suggestions, report) shape as suggest_weights.
        """
        if target_column not in df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in dataframe."
            )

        suggestions: List[MetricSpec] = []
        report: Dict[str, Dict[str, Any]] = {}
        target_series = pd.to_numeric(df[target_column], errors="coerce")

        for entry in candidate_metrics:
            # Looser normalization: bare strings ok in this exploratory path
            if isinstance(entry, str):
                spec = MetricSpec(name=entry, lookback=default_lookback)
                corr_col = entry
            else:
                # Reuse strict normalizer, but skip its direction requirement
                # by injecting a placeholder if needed.
                if isinstance(entry, dict) and entry.get("direction") is None:
                    entry = {**entry, "direction": "proportional"}
                spec, corr_col, _ = cls._normalize_candidate(entry, default_lookback)

            corr, n, status = _pairwise_correlation(df, target_series, corr_col)

            if status != "ok":
                spec.weight = 0.0
                report[spec.name] = {
                    "correlation": corr,
                    "n_observations": n,
                    "direction": spec.direction,
                    "weight": 0.0,
                    "direction_matches_data": None,
                    "rationale": status,
                }
                suggestions.append(spec)
                continue

            magnitude = abs(corr)
            weight = magnitude if magnitude >= min_correlation else 0.0
            inferred_direction = "proportional" if corr >= 0 else "inverse"

            spec.direction = inferred_direction
            spec.weight = weight
            report[spec.name] = {
                "correlation": corr,
                "n_observations": n,
                "direction": inferred_direction,
                "weight": weight,
                "direction_matches_data": True,  # inferred from data, by definition
                "rationale": (
                    f"correlation with '{target_column}' = {corr:+.3f} (n={n}). "
                    f"Inferred direction='{inferred_direction}', "
                    f"weight=|corr|={weight:.3f}."
                ),
            }
            suggestions.append(spec)

        return suggestions, report


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _correlation_column_for(spec: MetricSpec) -> str:
    """Pick the column to correlate against the target for a given spec."""
    if spec.columns and len(spec.columns) == 1:
        return spec.columns[0]
    return spec.name


def _pairwise_correlation(
    df: pd.DataFrame,
    target_series: pd.Series,
    corr_col: str,
) -> Tuple[Optional[float], int, str]:
    """
    Compute Pearson correlation between target_series and df[corr_col],
    pairwise-dropping NaN. Returns (correlation, n_observations, status).
    status is 'ok' on success or a human-readable rationale string on
    failure (column missing / insufficient data / constant series).
    """
    if corr_col not in df.columns:
        return None, 0, f"Column '{corr_col}' not found in dataframe — weight set to 0."

    candidate_series = pd.to_numeric(df[corr_col], errors="coerce")
    paired = pd.concat([target_series, candidate_series], axis=1).dropna()

    if len(paired) < 3 or paired.iloc[:, 1].nunique() < 2:
        return None, int(len(paired)), (
            f"Insufficient variation in '{corr_col}' "
            f"(n={len(paired)}) — weight set to 0."
        )

    corr = float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))
    if np.isnan(corr):
        corr = 0.0
    return corr, int(len(paired)), "ok"
