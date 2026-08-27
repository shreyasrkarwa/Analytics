import json
import datetime
import warnings
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from b2b_revenue_forecasting.metric_spec import MetricSpec
from b2b_revenue_forecasting.hierarchy import (
    HierarchyValidationError,
    coerce_metric_value,
    _UNCOERCIBLE,
)
from b2b_revenue_forecasting._dashboard_template import DASHBOARD_HTML_TEMPLATE


# Small floor (as a fraction of the sibling max) applied when computing
# inverse-direction shares, so that a single zero-valued sibling doesn't
# absorb 100% of the inverse weight via a 1/0 spike.
_INVERSE_EPS = 0.01

# Valid values for cascade_quota(gate_fallback=...)
_GATE_FALLBACKS = ("redistribute", "strand_at_root", "error")


class GateAllocationError(ValueError):
    """
    Raised by cascade_quota(gate_fallback="error") when a funded node's
    children are ALL gated, so its target cannot be placed anywhere
    below it without either relaxing the gate or stranding the money.
    """
    pass


class HedgeByDepth:
    """
    Depth-keyed hedge specification (issue #13). Pass as
    cascade_quota(hedge_multiplier=...) — or through cascade_many, where
    it is the ONLY way to express a per-level hedge, because the batch
    API builds each combination's hierarchy internally and node ids are
    never visible to the caller.

    The spec is resolved against whatever graph the cascader holds, at
    cascade time, into an ordinary per-node dict — so all downstream
    behavior (base layer, audit columns, reconciliation) is identical
    to passing that dict yourself.

    Parameters
    ----------
    from_leaves : Optional[Dict[int, float]]
        Keyed by a manager's distance to its FARTHEST descendant leaf:
        1 = deepest manager (all/some children are ICs), 2 = one level
        above, ... This is the natural basis for policies like
        "front-line managers hedge 10%, their directors 5%":
        {1: 1.10, 2: 1.05}. Leaves themselves never carry a hedge
        (hedges apply when a node distributes to its children).
    from_root : Optional[Dict[int, float]]
        Keyed by distance from the root (root = 0), i.e. the same
        numbers SalesHierarchy.node_depths() returns. In jagged
        hierarchies (ICs at different depths) this is NOT equivalent to
        from_leaves — pick the basis that matches your policy.
    default : float
        Multiplier for nodes matched by neither mapping. Default 1.0
        (no hedge).

    If a node is matched by BOTH mappings, the two multipliers COMPOSE
    (multiply) — they are independent policies. At least one mapping
    must be provided; all multipliers must be > 0.

    Example
    -------
    >>> cascade_many(..., hedge_multiplier=HedgeByDepth(
    ...     from_leaves={1: 1.10, 2: 1.05},   # deepest mgr 10%, next 5%
    ... ))
    """

    def __init__(self,
                 from_leaves: Optional[Dict[int, float]] = None,
                 from_root: Optional[Dict[int, float]] = None,
                 default: float = 1.0):
        if not from_leaves and not from_root:
            raise ValueError(
                "HedgeByDepth requires at least one of from_leaves= or "
                "from_root= (a {depth: multiplier} mapping)."
            )
        for label, mapping in (("from_leaves", from_leaves),
                               ("from_root", from_root)):
            for k, v in (mapping or {}).items():
                if not isinstance(k, int) or isinstance(k, bool) or k < 0:
                    raise ValueError(
                        f"HedgeByDepth.{label} keys must be non-negative "
                        f"ints (depths), got {k!r}."
                    )
                if not isinstance(v, (int, float)) or v <= 0:
                    raise ValueError(
                        f"HedgeByDepth.{label}[{k}] must be a positive "
                        f"multiplier, got {v!r}."
                    )
        if not isinstance(default, (int, float)) or default <= 0:
            raise ValueError(
                f"HedgeByDepth.default must be a positive multiplier, "
                f"got {default!r}."
            )
        self.from_leaves = dict(from_leaves or {})
        self.from_root = dict(from_root or {})
        self.default = float(default)

    def resolve(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Materialize this spec into a per-node {node_id: multiplier} dict
        for the given DAG. Called automatically by cascade_quota; public
        so consumers can inspect the mapping a given graph would get.
        """
        # Distance from root(s): shortest path, roots at 0 (matches
        # SalesHierarchy.node_depths()).
        root_depth: Dict[str, int] = {}
        for root in (n for n in graph.nodes if graph.in_degree(n) == 0):
            for node, d in nx.shortest_path_length(graph, source=root).items():
                if node not in root_depth or d < root_depth[node]:
                    root_depth[node] = d

        # Distance to farthest descendant leaf: leaves at 0, computed in
        # reverse topological order so children resolve before parents.
        leaf_dist: Dict[str, int] = {}
        for node in reversed(list(nx.topological_sort(graph))):
            children = list(graph.successors(node))
            leaf_dist[node] = (0 if not children
                               else 1 + max(leaf_dist[c] for c in children))

        resolved: Dict[str, float] = {}
        for node in graph.nodes:
            mult = None
            fr = self.from_root.get(root_depth.get(node))
            if fr is not None:
                mult = fr
            if graph.out_degree(node) > 0:          # managers only
                fl = self.from_leaves.get(leaf_dist[node])
                if fl is not None:
                    mult = fl if mult is None else mult * fl
            resolved[node] = self.default if mult is None else mult
        return resolved


class QuotaCascader:
    def __init__(self, hierarchy):
        """
        Initializes the cascader with a SalesHierarchy object.
        """
        # The underlying nx.DiGraph. `.graph` is the canonical accessor
        # across the whole package (SalesHierarchy.graph,
        # PipelineAdjuster.graph, QuotaCascader.graph) — issue #5.
        self.graph = hierarchy.graph
        # sanitized -> original node ids from the collision policy
        # (issue #7); used by quotas_to_dataframe's original_id column.
        self._id_map = dict(getattr(hierarchy, "id_map", {}) or {})
        # Populated after each multi-metric cascade_quota call so analysts
        # can inspect the normalized-weight contributions later (e.g., to
        # paste into a stakeholder report).
        self.weights_report = None
        # Populated after each cascade with gates so quotas_to_dataframe
        # and stakeholder reports can show WHY a node has zero quota.
        self.gated_nodes = set()
        # Base (un-hedged) quotas from the most recent cascade_quota call —
        # the same cascade run with hedge_multiplier=1.0 everywhere. The
        # invariant `sum(base_quotas at depth d) == macro_target` holds at
        # every depth when gate_fallback="redistribute" (the default).
        self.base_quotas = None
        # Diagnostics for gate fallback handling (issue #12) — populated
        # after every cascade_quota call:
        #   unallocated       — total dollars that could NOT be placed below
        #                       a funded node (only nonzero when
        #                       gate_fallback="strand_at_root")
        #   unallocated_nodes — {node_id: stranded_amount}
        #   gate_relaxed_nodes— nodes that RECEIVED quota despite being
        #                       gated, because every sibling was also gated
        #                       and gate_fallback="redistribute" relaxed the
        #                       gate as a last resort (no silent target loss)
        self.unallocated = 0.0
        self.unallocated_nodes = {}
        self.gate_relaxed_nodes = set()
        self.zero_metric_events = []               # issue #66
        self.carveout_nodes = set()                # issue #66
        # Overrides exceeding a parent's pool (issue #28): conservation is
        # mathematically impossible there, so the excess is reported
        # loudly instead of producing negative sibling quotas.
        self.overpinned = 0.0
        self.overpinned_nodes = {}
        # Inputs/outputs of the most recent cascade_quota call, kept so
        # gating_report() (issue #10) can reconcile without re-running.
        self.last_target = None
        self.last_quotas = None
        # Columns already flagged for non-numeric values — so the issue-#3
        # warning fires once per column, not once per node.
        self._non_numeric_warned = set()
        # Metrics already flagged for suspected grain mismatch (issue #36).
        self._grain_warned = set()

    def _warn_non_numeric(self, column: str, value: Any) -> None:
        """Warn (once per column) that a metric value was silently skipped
        pre-v0.6.1; now it is skipped LOUDLY (issue #3)."""
        if column in self._non_numeric_warned:
            return
        self._non_numeric_warned.add(column)
        warnings.warn(
            f"Metric column '{column}' holds non-numeric value(s) "
            f"(e.g. {value!r}) that cannot be coerced; they are treated as "
            f"MISSING and contribute nothing to cascades or gates. Clean "
            f"the data if this column should carry signal.",
            UserWarning,
            stacklevel=4,
        )

    @property
    def hierarchy(self):
        """
        Backward-compatible alias for `.graph` (issue #5). `.graph` is the
        canonical name across the package; pre-v0.7.2 code that read
        `cascader.hierarchy` keeps working.
        """
        return self.graph

    # ------------------------------------------------------------------
    # Single-metric helpers (legacy path — kept for backward compatibility)
    # ------------------------------------------------------------------
    def _calculate_node_historical_capacity(self, node_id: str,
                                            _visited: Optional[set] = None) -> float:
        """
        Recursively calculates the historical capacity of a node by summing up
        all '_Attainment' metrics of leaf nodes (ICs) underneath it.

        Supports any number of historical quarters (4, 8, 12, etc.) by dynamically
        discovering all attributes containing '_Attainment' on each IC node.

        For ICs with partial history (e.g., hired recently with some zero quarters),
        zero-valued quarters are imputed with the average of that IC's own non-zero
        quarters. This prevents underweighting reps who haven't been employed for
        the full lookback period.

        Returns 0.0 for brand-new ICs with no historical data at all — these are
        handled separately in cascade_quota() via equal-share carve-out.

        A recursion-stack guard (issue #1) turns accidental cycles /
        self-loops into a clear HierarchyValidationError instead of a
        RecursionError bottoming out inside networkx. (_visited tracks the
        CURRENT path only, so diamond-shaped DAGs — a node reachable via
        two branches — are still allowed, matching previous behavior.)
        """
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            raise HierarchyValidationError(
                f"Cycle detected while aggregating capacity at node "
                f"'{node_id}'. The hierarchy is not a DAG — run "
                f"SalesHierarchy.validate() to locate the cycle."
            )

        # If it's a leaf node (IC), return its historical capacity
        if self.graph.out_degree(node_id) == 0:
            attrs = self.graph.nodes[node_id]
            # Dynamically collect ALL attainment values (supports any number
            # of quarters). Coercion (issue #3) accepts numpy scalars and
            # numeric strings; uncoercible values warn once per column.
            attainments = []
            for k, v in attrs.items():
                if '_Attainment' not in k or v is None:
                    continue
                coerced = coerce_metric_value(v)
                if coerced is _UNCOERCIBLE:
                    self._warn_non_numeric(k, v)
                    continue
                attainments.append(float(coerced))

            if not attainments:
                return 0.0

            non_zero = [v for v in attainments if v > 0]

            if not non_zero:
                return 0.0  # Brand-new hire — handled by equal-share in cascade_quota

            # Impute zero quarters with the IC's own non-zero average
            avg_non_zero = sum(non_zero) / len(non_zero)
            imputed = [v if v > 0 else avg_non_zero for v in attainments]
            return sum(imputed)

        # Otherwise, aggregate the mathematical capacity of its children.
        # Push onto the recursion stack, recurse, then pop — so only true
        # ancestor->descendant->ancestor cycles trip the guard.
        _visited.add(node_id)
        try:
            total_capacity = 0.0
            for child in self.graph.successors(node_id):
                total_capacity += self._calculate_node_historical_capacity(
                    child, _visited)
        finally:
            _visited.discard(node_id)

        return total_capacity

    # ------------------------------------------------------------------
    # Multi-metric helpers (new path)
    # ------------------------------------------------------------------
    def _aggregate_node_metric(self, node_id: str, spec: MetricSpec,
                               _visited: Optional[set] = None) -> float:
        """
        Recursively compute one metric's aggregated value for a node, rolling
        up across the subtree below it.

        Mirrors _calculate_node_historical_capacity but is parameterized by a
        MetricSpec — so the SAME function works for NetNewACV, CloudSeats,
        ExpansionSpent, etc.

        For leaf (IC) nodes, the spec's columns are read directly off the
        node, aggregated (sum/mean/last), with optional zero-imputation for
        partial-history ICs.

        For non-leaf nodes, the metric is rolled up by SUMMING children — this
        is the natural rollup for stock/flow metrics (dollars, seats, counts).
        If you have a metric for which sum is not the right rollup (e.g., a
        rate), supply a precomputed column on parent nodes and use a leaf-only
        cascade — but for the metrics the cascader cares about (capacity-like
        signals), sum-rollup is correct.

        A recursion-stack guard (issue #1) raises a clear
        HierarchyValidationError on cycles/self-loops instead of a
        RecursionError; diamond-shaped DAGs remain allowed.
        """
        if _visited is None:
            _visited = set()
        if node_id in _visited:
            raise HierarchyValidationError(
                f"Cycle detected while aggregating metric '{spec.name}' at "
                f"node '{node_id}'. The hierarchy is not a DAG — run "
                f"SalesHierarchy.validate() to locate the cycle."
            )

        if self.graph.out_degree(node_id) == 0:
            attrs = self.graph.nodes[node_id]

            candidate_cols = spec.resolved_columns()
            # Issue #6 fallback: when no explicit columns were configured
            # and NONE of the Qi_<name> convention columns exist on this
            # leaf, read the attribute named exactly <name>. This makes
            # specs returned by suggest_weights directly usable when the
            # metric name IS the data column (the common single-column
            # case) — no more `spec.columns = [spec.name]` boilerplate.
            if (spec.columns is None
                    and spec.name in attrs
                    and not any(c in attrs for c in candidate_cols)):
                candidate_cols = [spec.name]

            raw_values = []
            for col in candidate_cols:
                v = attrs.get(col)
                if v is None:
                    continue  # genuinely absent — silent skip is correct
                # Coerce rather than type-check (issue #3): accepts int,
                # float, bool, numpy scalars (np.int64 / np.bool_ / ...),
                # and numeric/boolean strings. Values with no numeric
                # interpretation are skipped LOUDLY (warn once per column)
                # instead of silently aggregating to 0.
                coerced = coerce_metric_value(v)
                if coerced is _UNCOERCIBLE:
                    self._warn_non_numeric(col, v)
                    continue
                raw_values.append(coerced)

            if not raw_values:
                return 0.0

            # Auto-detect boolean / 0-1 sparse metrics. For those, zero is
            # a meaningful value (False / "didn't happen this quarter"),
            # not a missing-data marker — so imputation would falsely
            # inflate the node's signal. Skip imputation regardless of
            # spec.impute_zeros when the data looks boolean.
            looks_boolean = all(
                isinstance(v, bool) or v == 0 or v == 1
                for v in raw_values
            )

            values = [float(v) for v in raw_values]

            # Zero-imputation for partial-history nodes (skipped for booleans)
            if spec.impute_zeros and not looks_boolean:
                non_zero = [v for v in values if v > 0]
                if not non_zero:
                    return 0.0  # truly empty for this metric
                avg_non_zero = sum(non_zero) / len(non_zero)
                values = [v if v > 0 else avg_non_zero for v in values]

            if spec.aggregation == "sum":
                return sum(values)
            elif spec.aggregation == "mean":
                return sum(values) / len(values)
            elif spec.aggregation == "last":
                return values[-1]
            else:  # pragma: no cover — guarded by MetricSpec.__post_init__
                raise ValueError(f"Unknown aggregation: {spec.aggregation}")

        # Non-leaf: roll up across children (push/pop the recursion stack)
        _visited.add(node_id)
        try:
            total = 0.0
            for child in self.graph.successors(node_id):
                total += self._aggregate_node_metric(child, spec, _visited)
        finally:
            _visited.discard(node_id)
        return total

    def _compute_composite_shares(
        self,
        children: List[str],
        metrics: List[MetricSpec],
    ) -> Dict[str, float]:
        """
        Compute each child's share-of-parent using a weighted sum of
        per-metric normalized shares.

        For each metric m:
          - If direction == "proportional": share_m(child) = value_m(child) / sum_siblings value_m
          - If direction == "inverse":      share_m(child) = (1 / (value_m(child) + floor)) /
                                                              sum_siblings(1 / (value_m + floor))
            where floor = _INVERSE_EPS * max_sibling_value, preventing a 1/0 spike
            from monopolizing the metric.

        The final composite share for each child is:
          composite(child) = sum_m( normalized_weight_m * share_m(child) )

        Because each per-metric share-vector sums to 1 and the weights sum to 1,
        the composite vector also sums to 1 — making it directly usable to split
        a target across children.

        Returns
        -------
        dict[child_id -> share]   sums to 1.0 (within floating tolerance).
        Returns {} if no metric had usable data; caller should fall back to
        equal split in that case.
        """
        if not children:
            return {}

        # Active metrics: weight > 0 only. Otherwise the metric is essentially
        # disabled and contributes nothing.
        active = [m for m in metrics if m.weight > 0]
        if not active:
            return {}

        # Normalize weights so they sum to 1
        total_w = sum(m.weight for m in active)
        norm_weights = [m.weight / total_w for m in active]

        composite = {c: 0.0 for c in children}
        usable_metric_weight = 0.0  # weight from metrics that produced usable shares

        for spec, w in zip(active, norm_weights):
            values = {c: self._aggregate_node_metric(c, spec) for c in children}

            if spec.direction == "proportional":
                total = sum(values.values())
                if total <= 0:
                    # No siblings have any data for this metric — skip it; its
                    # weight gets redistributed implicitly via renormalization
                    # at the end.
                    continue
                shares = {c: values[c] / total for c in children}

            else:  # inverse
                # Apply floor relative to the sibling-max so a zero doesn't
                # cause a 1/0 spike. If all siblings are zero, the metric has
                # no signal here — skip it.
                max_v = max(values.values())
                if max_v <= 0:
                    continue
                floor = _INVERSE_EPS * max_v
                inv = {c: 1.0 / (values[c] + floor) for c in children}
                inv_total = sum(inv.values())
                shares = {c: inv[c] / inv_total for c in children}

            for c in children:
                composite[c] += w * shares[c]
            usable_metric_weight += w

        if usable_metric_weight == 0:
            return {}

        # If some metrics were skipped (no signal), the composite won't sum to
        # 1 — renormalize across the metrics that DID contribute.
        if abs(usable_metric_weight - 1.0) > 1e-9:
            composite = {c: v / usable_metric_weight for c, v in composite.items()}

        return composite

    def _is_brand_new_by_rule(
        self,
        node_id: str,
        metrics: List[MetricSpec],
        rule: str,
    ) -> bool:
        """
        Auto-detect whether a leaf node should be treated as brand-new,
        using one of the supported rules.

        rule == "all_metrics_zero":
            True iff EVERY metric's aggregated value for this node is zero.
        rule == "primary_metric_zero":
            True iff the FIRST metric's aggregated value for this node is
            zero. (The "primary" metric is whichever metric the user
            listed first.)
        """
        if not metrics:
            return False

        if rule == "primary_metric_zero":
            return self._aggregate_node_metric(node_id, metrics[0]) == 0.0

        # default: all_metrics_zero
        return all(
            self._aggregate_node_metric(node_id, m) == 0.0
            for m in metrics
        )

    def _node_has_brand_new_flag(self, node_id: str, attr: str) -> bool:
        """True iff the node has a truthy value under the given attribute."""
        return bool(self.graph.nodes[node_id].get(attr, False))

    def _warn_possible_grain_mismatch(self,
                                      specs: List[MetricSpec]) -> None:
        """
        Issue #36 guardrail: a metric whose values are IDENTICAL across
        (nearly) every leaf-sibling group is very likely populated at a
        coarser grain than the leaf — an ancestor-level number repeated
        onto every child row. Naive rollups then double-count it and
        sibling shares collapse to equal splits, silently.

        Heuristic (deliberately conservative):
          - only leaf-sibling groups with >= 2 members count; at least 2
            such groups must exist,
          - boolean / 0-1 metrics are exempt (legitimately constant),
          - all-zero groups are exempt (the tree-wide zero-signal
            warning owns that case),
          - warn when >= 90% of eligible groups are internally constant,
          - once per metric name per cascader; warning only — a
            legitimately uniform book still cascades normally.
        """
        sibling_groups: Dict[str, List[str]] = {}
        for n in self.graph.nodes:
            if self.graph.out_degree(n) == 0:
                for p in self.graph.predecessors(n):
                    sibling_groups.setdefault(p, []).append(n)
        groups = [ls for ls in sibling_groups.values() if len(ls) >= 2]
        if len(groups) < 2:
            return

        for spec in specs:
            if spec.name in self._grain_warned:
                continue
            per_group = [[self._aggregate_node_metric(leaf, spec)
                          for leaf in ls] for ls in groups]
            flat = [v for vs in per_group for v in vs]
            if not flat or all(v in (0.0, 1.0) for v in flat):
                continue                      # empty or boolean-ish
            eligible = [vs for vs in per_group if any(v != 0 for v in vs)]
            if len(eligible) < 2:
                continue
            constant = [vs for vs in eligible if max(vs) == min(vs)]
            if len(constant) / len(eligible) >= 0.9:
                self._grain_warned.add(spec.name)
                warnings.warn(
                    f"Metric '{spec.name}' is IDENTICAL across siblings in "
                    f"{len(constant)} of {len(eligible)} leaf groups — this "
                    f"usually means the column is populated at a coarser "
                    f"grain than the leaf (an ancestor-level value repeated "
                    f"onto every child row). Rollups will double-count it "
                    f"and sibling shares collapse to EQUAL SPLITS. Resolve "
                    f"the column to true per-leaf grain (e.g. dedup per "
                    f"account in SQL before summing per rep). If the "
                    f"uniformity is intentional, ignore this warning.",
                    UserWarning,
                    stacklevel=3,
                )

    @staticmethod
    def _passes_gate(value: float, gate: MetricSpec) -> bool:
        """
        The exact gate predicate (issue #9). A node PASSES iff:

          gate_mode "gt":     value >  gate_threshold   (default)
          gate_mode "ge":     value >= gate_threshold
          gate_mode "lt":     value <  gate_threshold
          gate_mode "le":     value <= gate_threshold
          gate_mode "truthy": bool(value)               (threshold ignored)
        """
        threshold = getattr(gate, "gate_threshold", 0.0)
        mode = getattr(gate, "gate_mode", "gt")
        if mode == "gt":
            return value > threshold
        if mode == "ge":
            return value >= threshold
        if mode == "lt":
            return value < threshold
        if mode == "le":
            return value <= threshold
        if mode == "truthy":
            return bool(value)
        raise ValueError(f"Unknown gate_mode '{mode}'")  # pragma: no cover

    def _compute_gated_set(self, gate_metrics: List[MetricSpec]) -> set:
        """
        Return the set of node_ids that fail at least one gate.

        For each gate metric, every node's aggregated value (rolled up
        from leaves via _aggregate_node_metric, same as cascade signals)
        is checked against the gate's predicate — see _passes_gate for
        the exact semantics per gate_mode. A node that fails ANY gate is
        gated (AND composition).

        Because _aggregate_node_metric sums child values for non-leaves,
        "gt"/"ge"/"truthy" gates propagate upward naturally: a non-leaf
        is gated iff its whole subtree lacks the signal. For "lt"/"le"
        gates the rollup grows with subtree size, so parents can fail
        while children pass — any resulting fully-gated level is handled
        by cascade_quota's gate_fallback.
        """
        gated = set()
        if not gate_metrics:
            return gated

        for node in self.graph.nodes:
            for gate in gate_metrics:
                value = self._aggregate_node_metric(node, gate)
                if not self._passes_gate(value, gate):
                    gated.add(node)
                    break  # AND logic: first failure is enough
        return gated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cascade_quota(
        self,
        root_node: str,
        macro_target: float,
        hedge_multiplier: Union[float, Dict[str, float], "HedgeByDepth"] = 1.0,
        new_ic_overrides: Optional[Dict[str, float]] = None,
        metrics: Optional[List[MetricSpec]] = None,
        new_ic_ids: Optional[List[str]] = None,
        new_ic_attr: Optional[str] = None,
        new_ic_rule: Optional[str] = None,
        gate_metrics: Optional[List[MetricSpec]] = None,
        gate_fallback: str = "redistribute",
        override_basis: str = "base",
        verbose: bool = True,
        on_zero_metric: str = "equal",
        metric_fallback: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Distributes the macro_target from the root_node down to all descendants.

        Two modes:

        1) LEGACY (metrics=None, default) — preserves the v0.2.x behavior
           exactly. Uses the implicit single-metric pathway that auto-discovers
           '_Attainment' attributes on IC nodes and sums them as historical
           capacity.

        2) MULTI-METRIC (metrics=[MetricSpec, ...]) — at every level, the
           share each child receives is a weighted blend of per-metric shares.
           Each MetricSpec carries its own direction (proportional / inverse),
           weight, lookback, and column mapping. See MetricSpec docstring for
           details. Suggested weights can be generated via
           MetricSpec.suggest_from_data().

        Parameters
        ----------
        root_node : str
            The top node whose quota we're distributing.
        macro_target : float
            The total target dollar amount at the root.
        hedge_multiplier : float | Dict[str, float] | HedgeByDepth
            Single float -> 5% buffer at EVERY manager level (e.g., 1.05).
            Dict -> per-node hedge mapping (e.g., {'RVP_NA_1': 1.10}).
            HedgeByDepth -> per-DEPTH policy (issue #13), resolved
            against this cascader's graph at call time; e.g.
            HedgeByDepth(from_leaves={1: 1.10, 2: 1.05}) gives the
            deepest managers 10%, the level above 5%, everyone else
            the default. Works through cascade_many, where per-node
            dicts are impossible (node ids aren't visible).
            Defaults to 1.0 (no hedge).
        new_ic_overrides : Optional[Dict[str, float]]
            Node IDs mapped to fixed quota amounts that bypass the
            algorithm (CRO-mandated pins).

            Since v0.13.0 (issue #28) pins work at ANY level: pinning a
            MANAGER fixes that subtree's total (the pin then cascades
            normally within the subtree), and pins on leaves in jagged
            hierarchies (an IC whose siblings are managers) are honored
            too — both were silently ignored before. Conservation rules:
            pins are paid first; unpinned siblings share the remainder
            with renormalized weights (so the parent total is conserved
            exactly); the brand-new equal-share carve-out is capped at
            the remaining pool. If pins alone EXCEED the parent's pool,
            conservation is impossible: siblings get $0 (never negative
            quotas), a warning fires, and the excess is reported via
            self.overpinned / self.overpinned_nodes and gating_report().

            Absorption policy (issue #37): the remainder flows to the
            NON-PINNED siblings proportional to their BASELINE
            (un-pinned) cascade. This is not a separate mode — because
            each child's baseline is target x its composite share,
            renormalizing shares among the non-pinned children is
            algebraically identical to splitting the remainder
            proportional to their baseline values (pinned to $0.0000 by
            regression test). No `remainder=` knob is needed.
        override_basis : str
            What a pin amount MEANS when hedging is in play (issue #23).
            "base" (default) — the pin is the UN-HEDGED plan number:
                base_quota == pin, and the hedged cascaded_quota is
                derived as pin x the node's compound hedge factor
                (product of its ancestors' hedges) — same treatment as
                every other node, so both layers conserve.
            "cascaded" — the pin is the EXACT final number the rep
                carries: cascaded_quota == pin, base_quota is derived
                as pin / compound hedge factor.
            Pre-v0.13.0 the same raw number was used in both layers
            (pinned reps silently received no hedge). With
            hedge_multiplier=1.0 the two bases are identical.
        metrics : Optional[List[MetricSpec]]
            If provided, switches to multi-metric cascading. If None, the
            legacy single-metric ('_Attainment') path is used.

            Weight normalization (issue #11): each spec's influence is
            weight / sum(weights of ACTIVE specs) — active meaning
            weight > 0; inactive specs contribute exactly 0. So raw
            weights [1.0, 0.5, 0.0] act as shares [66.7%, 33.3%, 0%],
            and a raw 0.067 alongside [1.0, 0.98, 0.4] is 2.7% of the
            influence, not 6.7%. The exact shares are printed before
            every verbose cascade and stored on self.weights_report;
            see MetricSpec.normalized_weights for the definition.
        new_ic_ids : Optional[List[str]]
            Explicit list of IC IDs to treat as brand-new (equal-share
            carve-out). Programmatic equivalent of new_ic_attr — useful
            when the brand-new list comes from outside the CSV.
        new_ic_attr : Optional[str]
            Name of a node attribute that flags brand-new ICs (typically
            populated via SalesHierarchy.from_dataframe(brand_new_col=...)
            from a CSV column). When provided, the cascader reads this flag
            from every leaf node — keeping all configuration in the same
            CSV the analyst already uploads. Defaults to None.
        new_ic_rule : Optional[str]
            Auto-detection rule for brand-new ICs:
              "all_metrics_zero"     — IC is new iff EVERY configured metric
                                       is zero (matches legacy intent).
              "primary_metric_zero"  — IC is new iff the FIRST metric in
                                       the list is zero. Useful when a
                                       primary signal (e.g., NetNewACV) is
                                       zero but secondary signals (cloud
                                       seats, accreditations) already exist.
        gate_metrics : Optional[List[MetricSpec]]
            Hard kill-switch metrics. Each spec's aggregated value is
            checked at every node; nodes whose value is <= the spec's
            gate_threshold (default 0.0) are EXCLUDED from the cascade
            and receive quota = 0. The excluded share is redistributed
            among non-gated siblings via the existing blend.

            Gates compose with AND: if ANY gate fails for a node, the
            node is gated. Because gate values are summed from leaves
            upward, a whole subtree (manager / director / region) is
            naturally gated when none of its leaves pass the gate —
            which is exactly the desired "no white space anywhere in
            this branch => no quota" semantics.

            CRO overrides win over gates: an IC pinned via
            new_ic_overrides gets its pinned quota even if its gate
            value is 0. (The CRO has explicitly assigned business
            judgment.)

            Useful for white-space-planning flows: e.g., cascading
            "migration NetNewACV" with a gate on "Unmigrated_Seats"
            ensures territories with nothing left to migrate get $0.

            The set of gated nodes is stored on self.gated_nodes after
            each call.
        gate_fallback : str
            What to do when a funded node's children are ALL gated, so
            its target has nowhere to go (issue #12 — "fully-gated
            subtree strands the target"). Note the ROOT is never gated
            to $0 in any mode; it always carries macro_target.

            "redistribute" (default)
                A fully-gated subtree's share first flows to its
                nearest non-gated siblings (this already happens
                naturally because gates roll up). If EVERY child of a
                funded node is gated — including the case where the
                whole tree fails the gate — the gate is relaxed at
                that level as a last resort and the target is
                distributed by the normal blend weights, so it still
                reaches ICs. Guarantees the base (un-hedged) quota
                sums to macro_target at EVERY depth; no silent target
                loss. Nodes funded this way are recorded in
                self.gate_relaxed_nodes (and flagged in
                quotas_to_dataframe).
            "strand_at_root"
                Children stay $0; the undistributable amount remains
                on the deepest non-gated ancestor and is reported via
                self.unallocated / self.unallocated_nodes (and an
                is_unallocated column in quotas_to_dataframe). Depth
                sums below that node will NOT reconcile — this is the
                explicit opt-in for "don't force money into gated
                territory."
            "error"
                Raise GateAllocationError instead, forcing the caller
                to decide.
        verbose : bool
            When True (default) in multi-metric mode, prints the normalized
            weights table before cascading so the analyst can see — and
            explain to stakeholders — exactly how much each signal
            influences allocation. Pass verbose=False to suppress (useful
            in tests / batch runs). The normalized-weights DataFrame is
            ALSO stored on self.weights_report for later inspection
            regardless of verbose.

        Brand-new detection — either-or precedence
        ------------------------------------------
        Either you tell the package WHICH ICs are brand-new (via
        new_ic_attr from CSV, or new_ic_ids from Python), OR you tell the
        package the RULE to figure it out (via new_ic_rule). Mixing both
        in the same call raises ValueError, because the explicit list and
        the rule answer the same question and disagreement would be silent.

        When none of new_ic_attr / new_ic_ids / new_ic_rule are set
        AND metrics= is provided, we default to new_ic_rule='all_metrics_zero'
        — the safest choice (matches the v0.2.x intent for the legacy path).

        Returns
        -------
        Dict[str, float]  node_id -> assigned quota (with hedging applied).

        Side effects: self.base_quotas holds the SAME cascade with
        hedge_multiplier=1.0 everywhere (computed in the same call — no
        second run needed), so `quota = base + hedge buffer` decomposes
        cleanly: pass unhedged_quotas="auto" to quotas_to_dataframe.
        self.unallocated / self.unallocated_nodes / self.gate_relaxed_nodes
        are refreshed per the gate_fallback docs above.
        """
        if new_ic_overrides is None:
            new_ic_overrides = {}
        explicit_new_ic_set = set(new_ic_ids or [])

        # ---- Either-or enforcement for brand-new IC detection -----------
        explicit_path_used = bool(new_ic_attr) or bool(new_ic_ids)
        if explicit_path_used and new_ic_rule is not None:
            raise ValueError(
                "Brand-new IC detection is either-or: pass an explicit "
                "identifier (new_ic_attr=<csv-column> OR new_ic_ids=<list>) "
                "OR pass new_ic_rule='all_metrics_zero' / 'primary_metric_zero' "
                "— not both. They answer the same question and silent "
                "disagreement would be a bug factory."
            )

        # Resolve the effective rule for the case where no explicit path is used
        effective_rule = new_ic_rule if new_ic_rule is not None else "all_metrics_zero"
        if effective_rule not in ("all_metrics_zero", "primary_metric_zero",
                                  "none"):
            raise ValueError(
                f"new_ic_rule must be 'all_metrics_zero', "
                f"'primary_metric_zero' or 'none', got '{effective_rule}'."
            )
        if on_zero_metric not in ("equal", "error", "fallback"):
            raise ValueError(
                f"on_zero_metric must be 'equal', 'error' or 'fallback', "
                f"got '{on_zero_metric}'.")
        if on_zero_metric == "fallback" and not (
                metric_fallback
                and all(isinstance(c, str) for c in metric_fallback)):
            raise ValueError(
                "on_zero_metric='fallback' needs metric_fallback=[...] — "
                "column names tried in order, optionally ending in "
                "'equal'.")

        if gate_fallback not in _GATE_FALLBACKS:
            raise ValueError(
                f"gate_fallback must be one of {_GATE_FALLBACKS}, "
                f"got '{gate_fallback}'."
            )

        # ---- Fail fast on cyclic graphs (issue #1) ----------------------
        # Hierarchies built via from_dataframe are validated at build time,
        # but a graph assembled manually with add_edge() could contain a
        # cycle — which previously surfaced as a RecursionError deep inside
        # networkx. Raise a clear, actionable error instead.
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            path = " -> ".join([cycle[0][0]] + [e[1] for e in cycle])
            raise HierarchyValidationError(
                f"cascade_quota requires a DAG, but the hierarchy contains "
                f"a cycle: {path}. Fix the reporting structure (see "
                f"SalesHierarchy.from_dataframe's on_collision parameter) "
                f"before cascading."
            )

        if override_basis not in ("base", "cascaded"):
            raise ValueError(
                f"override_basis must be 'base' or 'cascaded', "
                f"got '{override_basis}'."
            )

        # ---- Per-depth hedge resolution (issue #13) ---------------------
        # A HedgeByDepth spec is materialized into an ordinary per-node
        # dict against THIS graph, so everything downstream (base layer,
        # audit columns, reconciliation) behaves exactly as if the caller
        # had built the dict by hand. This is what makes depth-based
        # hedging work through cascade_many, which owns hierarchy
        # construction per combination.
        if isinstance(hedge_multiplier, HedgeByDepth):
            hedge_multiplier = hedge_multiplier.resolve(self.graph)

        def _real_hedge_at(n: str) -> float:
            """The ACTUAL hedge configured at node n (independent of which
            pass — hedged or base — is running). Used to compute compound
            hedge factors for override_basis conversion (issue #23)."""
            if isinstance(hedge_multiplier, dict):
                return float(hedge_multiplier.get(n, 1.0))
            return float(hedge_multiplier)

        # Choose which weight engine to use for THIS cascade.
        # In multi-metric mode we close over `metrics`; in legacy mode we use
        # the existing _Attainment-based capacity.
        use_metrics = metrics is not None and len(metrics) > 0

        # Compute + store + (optionally) print the normalized-weights view
        # so the analyst can see and explain how each metric contributes.
        if use_metrics:
            self.weights_report = MetricSpec.normalized_weights(metrics)
            if verbose:
                print(MetricSpec.format_normalized_weights(metrics))
        else:
            self.weights_report = None

        # Issue #6: a metric that reads zero signal across the WHOLE tree
        # is almost always a column-resolution mistake (wrong name, missing
        # columns=). It silently degraded allocations pre-v0.7.1; now it
        # warns loudly, naming the columns that were tried.
        if use_metrics:
            for m in metrics:
                if m.weight > 0 and self._aggregate_node_metric(root_node, m) == 0.0:
                    tried = m.resolved_columns()
                    if m.columns is None:
                        tried = tried + [m.name]
                    warnings.warn(
                        f"Metric '{m.name}' has ZERO signal across the entire "
                        f"tree under '{root_node}' — it will not influence "
                        f"this cascade. Columns tried: {tried}. If the data "
                        f"lives elsewhere, set MetricSpec(columns=[...]).",
                        UserWarning,
                        stacklevel=2,
                    )

        # Issue #36 guardrail: flag metrics that look repeated from an
        # ancestor grain (identical across ~all leaf-sibling groups).
        grain_specs = ([m for m in (metrics or []) if m.weight > 0]
                       + list(gate_metrics or []))
        if grain_specs:
            self._warn_possible_grain_mismatch(grain_specs)

        # Precompute the gated set (nodes whose any gate fails). Stored on
        # self so analysts can inspect it and so quotas_to_dataframe can
        # mark gated nodes in its is_gated column.
        self.gated_nodes = self._compute_gated_set(gate_metrics or [])
        if gate_metrics and verbose:
            print(f"Gates active: {len(gate_metrics)} "
                  f"({', '.join(g.name for g in gate_metrics)}); "
                  f"{len(self.gated_nodes)} nodes gated (will receive $0)")

        def _zero_metric_split(children, parent, track):
            """issue #66: a sibling set with NO metric signal. Policy:
            'error' raises naming the parent; 'fallback' tries the
            metric_fallback chain (first column with signal, 'equal'
            short-circuits); 'equal' (default) splits evenly. Every
            engagement is recorded (self.zero_metric_events) so the
            allocation basis is never invisible in the output."""
            if on_zero_metric == "error":
                raise GateAllocationError(
                    f"No metric signal among the children of '{parent}' "
                    f"— every child is 0 for the active slate. Use "
                    f"on_zero_metric='equal' (even split) or 'fallback' "
                    f"with metric_fallback=[...] to proceed.")
            shares, used = None, "equal"
            if on_zero_metric == "fallback":
                for col in metric_fallback:
                    if col == "equal":
                        break
                    spec = MetricSpec(col, direction="proportional",
                                      weight=1.0, columns=[col])
                    got = self._compute_composite_shares(children, [spec])
                    if got:
                        shares, used = got, col
                        break
            if shares is None:
                shares = {c: 1.0 / len(children) for c in children}
            if track:
                self.zero_metric_events.append(
                    {"parent": parent, "fallback_used": used})
            return shares

        def child_weights(children: List[str], _parent=None,
                          _track=False) -> Dict[str, float]:
            """
            Compute a normalized share dict for the given children using the
            currently-selected engine. A sibling set with no usable signal
            goes through the on_zero_metric policy (issue #66).
            """
            if use_metrics:
                shares = self._compute_composite_shares(children, metrics)
                if shares:
                    return shares
                return _zero_metric_split(children, _parent, _track)

            # Legacy path: capacity = sum of all _Attainment columns
            caps = {c: self._calculate_node_historical_capacity(c) for c in children}
            total = sum(caps.values())
            if total > 0:
                return {c: caps[c] / total for c in children}
            return _zero_metric_split(children, _parent, _track)

        def is_new_ic(node_id: str) -> bool:
            """True if this leaf should get the equal-share carve-out."""
            if use_metrics:
                # Explicit path wins when set (mutex was enforced above)
                if new_ic_attr and self._node_has_brand_new_flag(node_id, new_ic_attr):
                    return True
                if node_id in explicit_new_ic_set:
                    return True
                if explicit_path_used:
                    # Explicit identification was in play; rule is intentionally
                    # NOT consulted (the user opted into the explicit path).
                    return False
                if effective_rule == "none":
                    # issue #66: the rule misfires on per-combo zero-metric
                    # slices (a rep with no dc_seats for Migration is not a
                    # new hire) — 'none' turns the rule off entirely.
                    return False
                return self._is_brand_new_by_rule(node_id, metrics, effective_rule)
            # Legacy path: capacity == 0
            return self._calculate_node_historical_capacity(node_id) == 0.0

        def run_cascade(effective_hedge, track_diagnostics: bool,
                        hedged_run: bool = True) -> Dict[str, float]:
            """
            One full top-down distribution pass.

            The ROOT always carries macro_target — it is never zeroed by a
            gate (issue #12). Gated children get $0 and their share flows
            to non-gated siblings; when EVERY child of a funded node is
            gated, `gate_fallback` decides what happens.

            When track_diagnostics is True (the primary/hedged pass), the
            self.unallocated / self.unallocated_nodes /
            self.gate_relaxed_nodes diagnostics are refreshed.
            """
            if track_diagnostics:
                self.unallocated = 0.0
                self.unallocated_nodes = {}
                self.gate_relaxed_nodes = set()
                self.overpinned = 0.0
                self.overpinned_nodes = {}
                self.zero_metric_events = []       # issue #66
                self.carveout_nodes = set()        # issue #66

            # The root is NEVER gated to $0 — a fully-gated tree is handled
            # per gate_fallback below, not by silently dropping the target.
            quotas = {root_node: macro_target}
            # Compound REAL hedge factor per node (product of ancestors'
            # configured hedges) — used to convert pin amounts between the
            # base and cascaded layers (issue #23).
            cum_hedge = {root_node: 1.0}

            # Traverse top-down through the organization
            for node in nx.topological_sort(self.graph):
                if node not in quotas:
                    continue

                current_target = quotas[node]
                all_children = list(self.graph.successors(node))

                if not all_children:
                    continue  # Reached an IC (leaf node)

                # Gate filter — gated children get quota 0, do NOT
                # contribute to the blend, and are excluded from brand-new
                # carve-out / override logic below. (Exception: CRO overrides
                # win over gates — pinned ICs keep their pinned quota even if
                # gated. Documented as explicit business override.)
                gated_children = [c for c in all_children
                                  if c in self.gated_nodes
                                  and c not in new_ic_overrides]
                children = [c for c in all_children if c not in gated_children]

                if not children:
                    # EVERY child is gated (and none has an override) — the
                    # target would be stranded here. Apply the fallback.
                    if gate_fallback == "redistribute":
                        # Last resort: relax the gate at this level so the
                        # target still reaches ICs and depth sums reconcile.
                        children = all_children
                        gated_children = []
                        if track_diagnostics and current_target != 0:
                            self.gate_relaxed_nodes.update(all_children)
                    elif gate_fallback == "strand_at_root":
                        for c in all_children:
                            quotas[c] = 0.0
                        if track_diagnostics and current_target != 0:
                            self.unallocated += current_target
                            self.unallocated_nodes[node] = current_target
                        continue
                    else:  # "error"
                        raise GateAllocationError(
                            f"All children of '{node}' are gated — its target "
                            f"of {current_target:,.2f} cannot be distributed. "
                            f"Use gate_fallback='redistribute' to relax the "
                            f"gate as a last resort, or 'strand_at_root' to "
                            f"keep the amount at '{node}' and report it via "
                            f"self.unallocated."
                        )

                for c in gated_children:
                    quotas[c] = 0.0

                # Determine the specific hedge for this manager node
                if isinstance(effective_hedge, dict):
                    current_hedge = effective_hedge.get(node, 1.0)
                else:
                    current_hedge = effective_hedge

                # Apply the hedge/overassignment buffer for this layer of management
                target_to_distribute = current_target * current_hedge

                # Compound REAL hedge factor for this node's children —
                # basis conversion for pins (issue #23). Computed from the
                # CONFIGURED hedge regardless of which pass is running.
                child_factor = cum_hedge.get(node, 1.0) * _real_hedge_at(node)
                for c in children:
                    if c not in cum_hedge or child_factor < cum_hedge[c]:
                        cum_hedge[c] = child_factor

                # ---- Pins at ANY level (issues #28 / #23) ---------------
                pinned = [c for c in children if c in new_ic_overrides]
                unpinned = [c for c in children if c not in new_ic_overrides]

                pin_total = 0.0
                for c in pinned:
                    raw = float(new_ic_overrides[c])
                    if override_basis == "base":
                        # Pin is the un-hedged plan number; the hedged
                        # layer derives it via the compound factor.
                        value = raw * child_factor if hedged_run else raw
                    else:  # "cascaded"
                        # Pin is the exact final number; the base layer
                        # derives it by dividing the factor back out.
                        value = raw if hedged_run else raw / child_factor
                    quotas[c] = value
                    pin_total += value

                pool_left = target_to_distribute - pin_total
                if pool_left < -0.005:
                    # Pins exceed the parent's pool: conservation is
                    # impossible. Never emit negative sibling quotas —
                    # report the excess loudly instead (issue #28).
                    if track_diagnostics:
                        self.overpinned += -pool_left
                        self.overpinned_nodes[node] = -pool_left
                        warnings.warn(
                            f"Overrides under '{node}' total "
                            f"{pin_total:,.2f}, exceeding its pool of "
                            f"{target_to_distribute:,.2f} by "
                            f"{-pool_left:,.2f}. Unpinned siblings receive "
                            f"$0 (never negative quotas); children will sum "
                            f"ABOVE the parent. See cascader.overpinned_nodes "
                            f"/ gating_report().",
                            UserWarning,
                            stacklevel=3,
                        )
                    pool_left = 0.0

                if not unpinned:
                    # Every child pinned. Any leftover pool cannot flow —
                    # report it, mirroring gate stranding semantics.
                    if track_diagnostics and pool_left > 0.005:
                        self.unallocated += pool_left
                        self.unallocated_nodes[node] = pool_left
                        warnings.warn(
                            f"All children of '{node}' are pinned; "
                            f"{pool_left:,.2f} of its pool cannot be "
                            f"distributed (see cascader.unallocated_nodes).",
                            UserWarning,
                            stacklevel=3,
                        )
                    continue

                # Leaf-level semantics (brand-new carve-out) apply when all
                # UNPINNED children are ICs.
                at_leaf_level = all(self.graph.out_degree(c) == 0
                                    for c in unpinned)

                if at_leaf_level:
                    new_ics = [c for c in unpinned if is_new_ic(c)]
                    experienced_ics = [c for c in unpinned if c not in new_ics]

                    if new_ics and track_diagnostics:
                        self.carveout_nodes.update(new_ics)
                    if new_ics:
                        # Equal share for brand-new ICs — historically a
                        # share of the TOTAL pool (len(children) incl.
                        # pinned), but capped so the pool is never
                        # overdrawn (issue #28); when no experienced
                        # siblings exist, the whole remaining pool is
                        # split equally so the parent still conserves.
                        intended = target_to_distribute / len(children)
                        if experienced_ics:
                            equal_share = min(intended,
                                              pool_left / len(new_ics))
                        else:
                            equal_share = pool_left / len(new_ics)
                        for ic in new_ics:
                            quotas[ic] = equal_share

                        remaining = pool_left - equal_share * len(new_ics)
                        if experienced_ics:
                            weights = child_weights(experienced_ics, node, track_diagnostics)
                            for ic in experienced_ics:
                                quotas[ic] = remaining * weights[ic]
                    else:
                        weights = child_weights(unpinned, node, track_diagnostics)
                        for ic in unpinned:
                            quotas[ic] = pool_left * weights[ic]
                else:
                    # Standard proportional distribution among unpinned
                    # children (weights renormalize automatically).
                    weights = child_weights(unpinned, node, track_diagnostics)
                    for child in unpinned:
                        quotas[child] = pool_left * weights[child]

            return quotas

        # Primary (hedged) pass — diagnostics reflect this pass.
        quotas = run_cascade(hedge_multiplier, track_diagnostics=True,
                             hedged_run=True)

        # Base (un-hedged) pass, computed in the SAME call so
        # hedged = base + hedge buffer decomposes without a second
        # cascade_quota run. Skipped (aliased) when no hedge is in play.
        if isinstance(hedge_multiplier, (int, float)) and float(hedge_multiplier) == 1.0:
            self.base_quotas = dict(quotas)
        else:
            self.base_quotas = run_cascade(1.0, track_diagnostics=False,
                                           hedged_run=False)

        # Remembered for gating_report() (issue #10)
        self.last_target = float(macro_target)
        self.last_quotas = dict(quotas)

        if verbose and self.gate_relaxed_nodes:
            print(f"Gate fallback: {len(self.gate_relaxed_nodes)} gated node(s) "
                  f"received quota anyway because every sibling was also gated "
                  f"(gate_fallback='redistribute'); see cascader.gate_relaxed_nodes.")
        if verbose and self.unallocated:
            print(f"Gate fallback: ${self.unallocated:,.2f} could not be "
                  f"distributed below {len(self.unallocated_nodes)} node(s) "
                  f"(gate_fallback='strand_at_root'); see cascader.unallocated_nodes.")

        return quotas

    def cascade_proportional(
        self,
        root_node: str,
        macro_target: float,
        metric: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        direction: str = "proportional",
        **cascade_kwargs: Any,
    ) -> Dict[str, float]:
        """
        Deterministic "proportional-to-metric" cascade (issue #34):
        split the target across children in proportion to a chosen
        metric (or a fixed-weight blend) — no correlation, no target
        column, works at any slice size including n=1/2.

            # this team holds 30% of the DC seats -> 30% of the quota
            quotas = cascader.cascade_proportional(
                'Enterprise_EMEA', 1_000_000, metric='dc_seats')

            # fixed-weight blend (2:1 influence)
            quotas = cascader.cascade_proportional(
                'Enterprise_EMEA', 1_000_000,
                metrics={'dc_seats': 1.0, 'cloud_seats': 0.5})

        This is pure sugar over the mechanism that has ALWAYS been the
        default: fixed-weight MetricSpecs passed to
        cascade_quota(metrics=...) are used as-is — `suggest_weights`
        is an optional helper, never a required stage, and no
        statistics run unless you call it yourself. This method just
        builds the specs and delegates, so every cascade_quota option
        (gate_metrics, hedge_multiplier / HedgeByDepth,
        new_ic_overrides, gate_fallback, ...) passes through.

        Parameters
        ----------
        metric : str
            Single metric name. Column resolution follows the v0.7.1
            order: explicit columns aren't needed — the Qi_<name>
            convention is tried first, then the plain <name> column.
        metrics : Dict[str, float]
            Fixed blend: {name: weight}, weights > 0, normalized to
            sum 1 at cascade time (influence = weight / sum(weights)).
        direction : str
            'proportional' (default) or 'inverse', applied to every
            metric given here. For mixed directions, build MetricSpecs
            and call cascade_quota directly.

        Returns the same dict as cascade_quota (base layer, gating
        report, etc. all populated identically).
        """
        if (metric is None) == (metrics is None):
            raise ValueError(
                "cascade_proportional: pass exactly one of metric='name' "
                "or metrics={'name': weight, ...}."
            )
        if metric is not None:
            blend = {metric: 1.0}
        else:
            blend = dict(metrics)
        if not blend:
            raise ValueError("cascade_proportional: metrics dict is empty.")
        for name, w in blend.items():
            if not isinstance(w, (int, float)) or w <= 0:
                raise ValueError(
                    f"cascade_proportional: weight for '{name}' must be a "
                    f"positive number, got {w!r}."
                )
        specs = [MetricSpec(name, direction=direction, weight=float(w))
                 for name, w in blend.items()]
        return self.cascade_quota(root_node, macro_target, metrics=specs,
                                  **cascade_kwargs)

    # ------------------------------------------------------------------
    # CSV-export helpers (convert dict output -> analyst-ready DataFrame)
    # ------------------------------------------------------------------
    def _node_depths(self) -> Dict[str, int]:
        """Compute depth of every node from the root(s). Roots are at 0."""
        depths: Dict[str, int] = {}
        roots = [n for n in self.graph.nodes
                 if self.graph.in_degree(n) == 0]
        for root in roots:
            lengths = nx.shortest_path_length(self.graph, source=root)
            for node, d in lengths.items():
                # If a node is reachable from multiple roots, keep the
                # shallowest depth.
                if node not in depths or d < depths[node]:
                    depths[node] = d
        return depths

    def quotas_to_dataframe(
        self,
        quotas: Dict[str, float],
        level_names: Optional[List[str]] = None,
        unhedged_quotas: Optional[Union[Dict[str, float], str]] = None,
        metadata_cols: Optional[List[str]] = None,
        source_df: Optional[pd.DataFrame] = None,
        source_join_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convert a cascade_quota result dict into a tidy DataFrame ready to
        write to CSV (`.to_csv('cascaded_quotas.csv', index=False)`).

        Columns:
          node_id            — the node identifier
          parent             — the node's direct parent (None for root)
          depth              — distance from root (root = 0)
          level              — level NAME if level_names provided, else
                               omitted (e.g., 'Region', 'IC')
          is_leaf            — True iff this is an IC / leaf node
          cascaded_quota     — the assigned quota dollar amount
          share_of_parent    — the effective share this node received
                               of its parent's quota (issue #38): base
                               layer when unhedged_quotas is given,
                               else the hedged layer. Root = 1.0; sums
                               to 1 per sibling group; gated nodes 0.

        If unhedged_quotas is provided, three additional audit columns:
          unhedged_quota     — what the quota would be if no hedge / no
                               overassignment were applied at any level
                               (i.e., cascade run with hedge_multiplier=1.0)
          hedge_buffer       — cascaded_quota − unhedged_quota (the dollar
                               overassignment added by hedging)
          overassignment_pct — hedge_buffer / unhedged_quota

        Rows are sorted by (depth, node_id) so the CSV reads top-down
        from root → ICs.

        Parameters
        ----------
        quotas : Dict[str, float]
            The dict returned by cascade_quota() with hedging applied.
        level_names : Optional[List[str]]
            Same shape as the path_cols list passed to
            SalesHierarchy.from_dataframe(). When provided, each row gets
            a human-readable 'level' column (e.g., 'Global', 'Region',
            'RVP', 'Director', 'Manager', 'IC').
        unhedged_quotas : Optional[Dict[str, float]] or "auto"
            Pass "auto" (recommended since v0.5.0) to use
            self.base_quotas — the un-hedged cascade computed
            automatically during the last cascade_quota call — so no
            second run is needed:

                quotas = cascader.cascade_quota(..., hedge_multiplier=my_hedge)
                df = cascader.quotas_to_dataframe(
                    quotas, level_names=taxonomy, unhedged_quotas="auto")

            Alternatively pass an explicit dict (a second cascade_quota
            result computed with hedge_multiplier=1.0 but otherwise
            identical inputs) — the pre-v0.5.0 pattern.

        Since v0.5.0 two more audit columns appear when relevant:
          gate_relaxed   — True for nodes that received quota despite
                           being gated, because every sibling was also
                           gated (gate_fallback="redistribute" last
                           resort). Only added if any node was relaxed.
          is_unallocated — True for nodes holding target dollars that
                           could not be distributed below them
                           (gate_fallback="strand_at_root"). Only added
                           if any amount was stranded.

        Metadata carry-through (issue #7, v0.8.0)
        -----------------------------------------
        metadata_cols : Optional[List[str]]
            Node-attribute names to emit as columns (populate them via
            from_dataframe(metadata_cols=[...]) — or any metric column
            works too). Nodes without the attribute get NaN.
        source_df : Optional[pd.DataFrame]
            Original source frame to LEFT-JOIN onto leaf rows, so the
            CSV is analysis-ready without a manual merge. Requires
            source_join_col. The join uses ORIGINAL ids (see below), so
            it works even when the collision policy renamed nodes.
            Overlapping column names get a '_source' suffix.
        source_join_col : Optional[str]
            Column in source_df holding the leaf id — typically your
            deepest taxonomy column (e.g. 'node_5_rep_no').

        original_id column
            Added automatically whenever the hierarchy's collision
            policy renamed any node (SalesHierarchy.id_map non-empty):
            the pre-sanitization value for renamed nodes, and the
            node_id itself for everything else.
        """
        if source_df is not None and not source_join_col:
            raise ValueError(
                "source_df requires source_join_col (the column in "
                "source_df holding the leaf id, e.g. your deepest "
                "taxonomy column)."
            )
        if isinstance(unhedged_quotas, str):
            if unhedged_quotas != "auto":
                raise ValueError(
                    "unhedged_quotas accepts a dict or the string 'auto' "
                    f"(got '{unhedged_quotas}')."
                )
            if self.base_quotas is None:
                raise ValueError(
                    "unhedged_quotas='auto' requires a prior cascade_quota "
                    "call (self.base_quotas is not populated yet)."
                )
            unhedged_quotas = self.base_quotas
        depths = self._node_depths()
        rows = []
        for node, quota in quotas.items():
            parents = list(self.graph.predecessors(node))
            parent = parents[0] if parents else None
            depth = depths.get(node, -1)
            row = {
                "node_id": node,
                "parent": parent,
                "depth": depth,
                "is_leaf": self.graph.out_degree(node) == 0,
                "cascaded_quota": round(float(quota), 2),
            }
            if level_names and 0 <= depth < len(level_names):
                row["level"] = level_names[depth]

            # original_id / original_parent (issues #7/#18):
            # pre-sanitization values for renamed nodes, self otherwise
            if self._id_map:
                row["original_id"] = self._id_map.get(node, node)
                row["original_parent"] = (self._id_map.get(parent, parent)
                                          if parent is not None else None)

            # Metadata carry-through from node attributes (issue #7)
            if metadata_cols:
                attrs = self.graph.nodes[node]
                for mc in metadata_cols:
                    row[mc] = attrs.get(mc)

            if unhedged_quotas is not None:
                unhedged = float(unhedged_quotas.get(node, 0.0))
                buffer = float(quota) - unhedged
                pct = (buffer / unhedged) if unhedged != 0 else 0.0
                row["unhedged_quota"] = round(unhedged, 2)
                row["hedge_buffer"] = round(buffer, 2)
                row["overassignment_pct"] = round(pct, 4)

            # share_of_parent (issue #38): the effective share the
            # cascade applied at this node's sibling split, on the BASE
            # layer when available (falls back to the hedged layer).
            # Root = 1.0; parent with $0 -> NaN. Sums to 1 per sibling
            # group; a gated node shows 0 directly.
            layer = (unhedged_quotas if unhedged_quotas is not None
                     else quotas)
            if parent is None:
                row["share_of_parent"] = 1.0
            else:
                pval = float(layer.get(parent, 0.0))
                nval = float(layer.get(node, 0.0))
                row["share_of_parent"] = (round(nval / pval, 6)
                                          if pval != 0 else float("nan"))

            # Flag gated nodes so analysts can distinguish "0 because gated"
            # from "0 because no signal." Only populated when the most recent
            # cascade_quota call used gate_metrics.
            if self.gated_nodes:
                row["is_gated"] = node in self.gated_nodes
            if self.gate_relaxed_nodes:
                row["gate_relaxed"] = node in self.gate_relaxed_nodes
            if self.unallocated_nodes:
                row["is_unallocated"] = node in self.unallocated_nodes

            rows.append(row)

        df = pd.DataFrame(rows)
        # Reorder columns: depth/level lead, then identifiers, then values
        col_order = ["depth"]
        if "level" in df.columns:
            col_order.append("level")
        col_order.append("node_id")
        if "original_id" in df.columns:
            col_order.append("original_id")
        if "original_parent" in df.columns:
            col_order.append("original_parent")
        col_order += ["parent", "is_leaf", "cascaded_quota"]
        if unhedged_quotas is not None:
            col_order += ["unhedged_quota", "hedge_buffer", "overassignment_pct"]
        col_order.append("share_of_parent")   # issue #38
        for optional_col in ("is_gated", "gate_relaxed", "is_unallocated"):
            if optional_col in df.columns:
                col_order.append(optional_col)
        if metadata_cols:
            col_order += [c for c in metadata_cols if c in df.columns]
        df = df[col_order]

        # Optional left-join of the ORIGINAL source frame onto leaf rows,
        # keyed on original ids so it survives collision renames (issue #7).
        if source_df is not None:
            if source_join_col not in source_df.columns:
                raise ValueError(
                    f"source_join_col '{source_join_col}' not found in "
                    f"source_df columns: {list(source_df.columns)}"
                )
            join_key = "__join_id__"
            df[join_key] = df.apply(
                lambda r: (self._id_map.get(r["node_id"], r["node_id"])
                           if r["is_leaf"] else None),
                axis=1,
            )
            right = source_df.drop_duplicates(subset=[source_join_col]).copy()
            right[join_key] = right[source_join_col].astype(str)
            right = right.drop(columns=[source_join_col])
            df = df.merge(right, on=join_key, how="left",
                          suffixes=("", "_source"))
            df = df.drop(columns=[join_key])

        return df.sort_values(["depth", "node_id"]).reset_index(drop=True)

    def gating_report(self, tolerance: float = 0.01) -> Dict[str, Any]:
        """
        One consolidated view of the most recent cascade's gating outcome
        (issue #10) — instead of assembling it from gated_nodes /
        gate_relaxed_nodes / unallocated / reconciliation_report by hand.

        Returns
        -------
        dict with:
          target                 — the macro target passed to cascade_quota
          gated_count            — number of gated nodes (all levels)
          gated_node_ids         — sorted list of ALL gated node ids
          gated_leaf_ids         — the gated ICs only
          gate_relaxed_node_ids  — nodes funded despite being gated
                                   (gate_fallback='redistribute' last resort)
          unallocated_amount     — dollars stranded above gated levels
                                   (nonzero only with
                                   gate_fallback='strand_at_root')
          unallocated_nodes      — {node_id: stranded amount}
          leaf_quota_sum         — sum of IC quotas (hedged layer)
          leaf_base_sum          — sum of IC quotas (un-hedged base layer)
          base_gap               — target − leaf_base_sum − unallocated_amount
          reconciles             — True iff |base_gap| <= tolerance, i.e.
                                   every input dollar is either on an IC
                                   (base layer) or explicitly reported as
                                   unallocated. The issue-#12 invariant.

        Raises RuntimeError if no cascade has been run yet.
        """
        if self.last_quotas is None:
            raise RuntimeError(
                "gating_report() requires a prior cascade_quota call."
            )
        leaves = [n for n in self.last_quotas
                  if self.graph.out_degree(n) == 0]
        leaf_quota_sum = float(sum(self.last_quotas[n] for n in leaves))
        base = self.base_quotas or {}
        leaf_base_sum = float(sum(base.get(n, 0.0) for n in leaves))
        base_gap = self.last_target - leaf_base_sum - self.unallocated
        return {
            "target": self.last_target,
            "gated_count": len(self.gated_nodes),
            "gated_node_ids": sorted(self.gated_nodes),
            "gated_leaf_ids": sorted(
                n for n in self.gated_nodes
                if self.graph.has_node(n) and self.graph.out_degree(n) == 0
            ),
            "gate_relaxed_node_ids": sorted(self.gate_relaxed_nodes),
            "unallocated_amount": float(self.unallocated),
            "unallocated_nodes": dict(self.unallocated_nodes),
            "leaf_quota_sum": leaf_quota_sum,
            "leaf_base_sum": leaf_base_sum,
            "base_gap": float(base_gap),
            # Overrides exceeding a parent's pool (issue #28) — when
            # nonzero, children legitimately sum ABOVE their parent and
            # reconciles will be False.
            "overpinned_amount": float(self.overpinned),
            "overpinned_nodes": dict(self.overpinned_nodes),
            "reconciles": abs(base_gap) <= tolerance,
        }

    def hedge_ratios(self) -> Dict[str, float]:
        """
        Per-node hedge ratio (cascaded / base) from the most recent
        cascade (issue #21). 1.0 where the base is 0. This is the factor
        that converts an EDITED base_quota back into the hedged layer.
        """
        if self.last_quotas is None or self.base_quotas is None:
            raise RuntimeError(
                "hedge_ratios() requires a prior cascade_quota call."
            )
        return {
            n: (self.last_quotas[n] / self.base_quotas[n]
                if self.base_quotas.get(n) else 1.0)
            for n in self.last_quotas
        }

    def rehedge(self, edited_base: Dict[str, float]) -> Dict[str, float]:
        """
        Recompute the HEDGED layer from an edited base layer (issue #21).

        The supported editing workflow after any post-cascade adjustment
        (manual pins, reallocations, ...):

          1. Edit `base_quota` values — the un-hedged plan is the ONLY
             layer that conserves at every depth, so all pin math and
             parent rollups belong there (roll parents up as the SUM of
             their children's base).
          2. Call rehedge(edited_base) to derive the hedged layer:
             cascaded = base x that node's ORIGINAL hedge ratio.

        Summing HEDGED leaves into a parent instead double-counts the
        buffer up the tree (depth 0/1 suddenly show leaf-level
        overassignment) — that is the failure mode this helper exists to
        prevent. Nodes absent from the last cascade get ratio 1.0.

        Returns a new dict; neither input nor cached state is mutated.
        """
        ratios = self.hedge_ratios()
        return {n: float(v) * ratios.get(n, 1.0)
                for n, v in edited_base.items()}

    def reconciliation_report(
        self,
        quotas: Dict[str, float],
        target: Optional[float] = None,
        tolerance: float = 0.01,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Per-depth reconciliation: asserts the cascade conserves the target
        at every level (issue #12 acceptance criterion).

        For each depth d, sums the quotas of all nodes at that depth and
        compares against `target`. For an UN-HEDGED cascade (pass
        cascader.base_quotas, or any run with hedge_multiplier=1.0) with
        gate_fallback="redistribute", every depth must reconcile exactly.
        Hedged quotas legitimately grow with depth (compounded
        overassignment) — run this on the base layer, not the hedged one.

        Parameters
        ----------
        quotas : Dict[str, float]
            A cascade result. Typically cascader.base_quotas.
        target : Optional[float]
            The macro target. Defaults to the depth-0 (root) sum of
            `quotas`.
        tolerance : float
            Absolute dollar tolerance per depth (default 1 cent).
        strict : bool
            If True, raises AssertionError listing every non-reconciling
            depth instead of just flagging them in the DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
          depth, n_nodes, total_quota, target, delta, reconciles
        """
        depths = self._node_depths()
        by_depth: Dict[int, List[float]] = {}
        for node, q in quotas.items():
            d = depths.get(node, -1)
            if d < 0:
                continue
            by_depth.setdefault(d, []).append(float(q))

        if target is None:
            target = sum(by_depth.get(0, [0.0]))

        rows = []
        for d in sorted(by_depth):
            total = sum(by_depth[d])
            delta = total - target
            rows.append({
                "depth": d,
                "n_nodes": len(by_depth[d]),
                "total_quota": round(total, 2),
                "target": round(float(target), 2),
                "delta": round(delta, 2),
                "reconciles": abs(delta) <= tolerance,
            })
        df = pd.DataFrame(rows)

        if strict and not df["reconciles"].all():
            bad = df[~df["reconciles"]]
            details = "; ".join(
                f"depth {int(r.depth)}: total {r.total_quota:,.2f} "
                f"vs target {r.target:,.2f} (delta {r.delta:,.2f})"
                for r in bad.itertuples()
            )
            raise AssertionError(f"Cascade does not reconcile — {details}")

        return df

    def quotas_diff_to_dataframe(
        self,
        original: Dict[str, float],
        adjusted: Dict[str, float],
        level_names: Optional[List[str]] = None,
        leaf_only: bool = True,
    ) -> pd.DataFrame:
        """
        Compare two quota dicts (e.g., before vs after pipeline
        redistribution) and return a side-by-side DataFrame ready for
        CSV export.

        Columns:
          node_id, parent, depth, (level), is_leaf,
          original_quota, adjusted_quota, delta, delta_pct

        delta_pct is `delta / original_quota` (0 if original is 0).

        Parameters
        ----------
        original, adjusted : Dict[str, float]
            The two quota dicts to compare. Keys must match.
        level_names : Optional[List[str]]
            Optional human-readable level names (see quotas_to_dataframe).
        leaf_only : bool
            If True (default), only includes leaf / IC nodes — which is
            where PipelineAdjuster.adjust() actually changes anything.
            Set False to include the full hierarchy.
        """
        depths = self._node_depths()
        rows = []
        all_nodes = set(original) | set(adjusted)
        for node in all_nodes:
            is_leaf = self.graph.out_degree(node) == 0
            if leaf_only and not is_leaf:
                continue
            orig = float(original.get(node, 0.0))
            adj = float(adjusted.get(node, 0.0))
            delta = adj - orig
            pct = (delta / orig) if orig != 0 else 0.0
            parents = list(self.graph.predecessors(node))
            parent = parents[0] if parents else None
            depth = depths.get(node, -1)
            row = {
                "node_id": node,
                "parent": parent,
                "depth": depth,
                "is_leaf": is_leaf,
                "original_quota": round(orig, 2),
                "adjusted_quota": round(adj, 2),
                "delta": round(delta, 2),
                "delta_pct": round(pct, 4),
            }
            if level_names and 0 <= depth < len(level_names):
                row["level"] = level_names[depth]
            rows.append(row)

        df = pd.DataFrame(rows)
        col_order = ["depth"]
        if "level" in df.columns:
            col_order.append("level")
        col_order += ["node_id", "parent", "is_leaf",
                      "original_quota", "adjusted_quota", "delta", "delta_pct"]
        df = df[col_order]
        return df.sort_values(["depth", "node_id"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # HTML dashboard (self-contained, opens in any browser)
    # ------------------------------------------------------------------
    def to_html_dashboard(
        self,
        quotas: Dict[str, float],
        output_path: str,
        title: str = "Quota Cascade Dashboard",
        macro_target: Optional[float] = None,
        region_level_index: int = 1,
        top_n_ics: int = 20,
        top_n_redistributions: int = 12,
        unhedged_quotas: Optional[Dict[str, float]] = None,
        adjusted_quotas: Optional[Dict[str, float]] = None,
        diagnosis: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Generate a self-contained interactive HTML dashboard visualizing
        the cascade. Opens in any browser; no server required. Uses
        Chart.js loaded from CDN; cascade data is embedded inline as
        JSON.

        Charts rendered:
          - Quota by region (with unhedged base overlaid if provided)
          - Top N IC quotas (horizontal bar)
          - Top redistributions: original vs adjusted (if
            adjusted_quotas provided)
          - Pipeline coverage by risk status (if diagnosis provided)
          - Per-region summary table (cascaded / unhedged / buffer / %)

        Parameters
        ----------
        quotas : Dict[str, float]
            The dict returned by cascade_quota() (with hedging).
        output_path : str
            File path to write the HTML to. Overwrites if exists.
        title : str
            Title shown in the dashboard header.
        macro_target : Optional[float]
            The macro target the cascade was started from. Used for
            summary stats. If None, inferred from the root node's quota.
        region_level_index : int
            Depth in the hierarchy where "regions" sit. Default 1 (i.e.,
            children of the root). Set to match your taxonomy.
        top_n_ics : int
            How many ICs to show in the top-ICs chart. Default 20.
        top_n_redistributions : int
            How many ICs to show in the redistributions chart (sorted by
            absolute delta). Default 12.
        unhedged_quotas, adjusted_quotas, diagnosis : optional
            Same shapes as in quotas_to_dataframe / PipelineAdjuster.
            When provided, additional charts/columns light up.
        """
        depths = self._node_depths()

        # --- Roots & macro target inference
        roots = [n for n in self.graph.nodes
                 if self.graph.in_degree(n) == 0]
        root_quota = quotas[roots[0]] if roots and roots[0] in quotas else None
        if macro_target is None and root_quota is not None:
            # If hedging was applied at the root, the root quota already
            # has the root hedge multiplier folded in. Use the unhedged
            # root if we have it.
            if unhedged_quotas is not None and roots[0] in unhedged_quotas:
                macro_target = unhedged_quotas[roots[0]]
            else:
                macro_target = root_quota

        # --- Region rows (depth == region_level_index)
        region_rows = []
        for node, q in quotas.items():
            if depths.get(node) == region_level_index:
                row = {"region": node, "cascaded": float(q)}
                if unhedged_quotas is not None:
                    row["unhedged"] = float(unhedged_quotas.get(node, 0.0))
                else:
                    row["unhedged"] = None
                region_rows.append(row)
        region_rows.sort(key=lambda r: -r["cascaded"])

        # --- Leaves & top ICs
        leaves = [(n, float(q)) for n, q in quotas.items()
                  if self.graph.out_degree(n) == 0]
        leaves.sort(key=lambda t: -t[1])
        top_ics = [{"node_id": n, "cascaded_quota": q}
                   for n, q in leaves[:top_n_ics]]

        # --- Redistributions (if adjusted provided)
        redistributions = []
        if adjusted_quotas is not None:
            diffs = []
            for node, _ in leaves:
                orig = float(quotas.get(node, 0.0))
                adj = float(adjusted_quotas.get(node, 0.0))
                delta = adj - orig
                if abs(delta) > 0.5:  # ignore noise
                    diffs.append((node, orig, adj, delta))
            diffs.sort(key=lambda t: -abs(t[3]))
            redistributions = [
                {"node_id": n, "original": o, "adjusted": a, "delta": d}
                for n, o, a, d in diffs[:top_n_redistributions]
            ]

        # --- Risk status counts (if diagnosis provided)
        risk_counts = []
        if diagnosis is not None and "Risk_Status" in diagnosis.columns:
            counts = diagnosis["Risk_Status"].value_counts().to_dict()
            order = ["Healthy", "Moderate", "At Risk", "Critical"]
            for status in order:
                if status in counts:
                    risk_counts.append({"status": status, "count": int(counts[status])})
            for status, count in counts.items():
                if status not in order:
                    risk_counts.append({"status": status, "count": int(count)})

        # --- Summary cards
        ic_total = sum(q for _, q in leaves)
        summary_cards = [
            {"label": "Cascaded total (ICs)",
             "value": f"${ic_total:,.0f}",
             "sub": f"across {len(leaves)} ICs"},
        ]
        if unhedged_quotas is not None:
            unhedged_total = sum(unhedged_quotas.get(n, 0.0) for n, _ in leaves)
            buffer = ic_total - unhedged_total
            buffer_pct = (buffer / unhedged_total * 100) if unhedged_total > 0 else 0.0
            summary_cards.append({
                "label": "Hedge buffer (IC total)",
                "value": f"${buffer:,.0f}",
                "sub": f"{buffer_pct:.2f}% overassignment",
            })
        if adjusted_quotas is not None:
            n_moved = sum(
                1 for n, _ in leaves
                if abs(float(adjusted_quotas.get(n, 0)) - float(quotas.get(n, 0))) > 0.5
            )
            summary_cards.append({
                "label": "Redistributed ICs",
                "value": f"{n_moved}",
                "sub": "zero-sum per manager",
            })
        if diagnosis is not None and "Risk_Status" in diagnosis.columns:
            critical = int((diagnosis["Risk_Status"] == "Critical").sum())
            summary_cards.append({
                "label": "Critical-risk nodes",
                "value": f"{critical}",
                "sub": "from pipeline diagnose",
            })

        # --- Build the payload
        payload = {
            "meta": {
                "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ic_count": len(leaves),
                "macro_target": float(macro_target) if macro_target is not None else 0.0,
                "summary_cards": summary_cards,
            },
            "regions": region_rows,
            "top_ics": top_ics,
            "redistributions": redistributions,
            "risk_counts": risk_counts,
        }

        html = (
            DASHBOARD_HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__PAYLOAD__", json.dumps(payload))
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
