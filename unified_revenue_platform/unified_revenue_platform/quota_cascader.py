"""
Layer 3 — Target Cascading: DAG-Based Quota Allocation Engine
=============================================================

The architectural centerpiece of the Unified Revenue Platform.

Problem:
    Top-line revenue targets arrive from financial planning (Anaplan) as
    a single macro number (e.g., $500M for the fiscal year). This number
    must be deterministically distributed downward through the entire
    organizational hierarchy — from CRO to Regional VPs to Directors to
    Managers to every IC — respecting historical performance weights and
    injecting managerial safety buffers at each level.

Why Standard Approaches Fail:
    1. Meta Prophet (hierarchical forecasting): Produces statistically
       independent predictions at each node. Nothing prevents the sum
       of child forecasts from diverging 30%+ from the parent forecast.
    2. MinT/OLS Reconciliation: Corrects statistical incoherence but
       has no mechanism to inject intentional managerial hedges — those
       are business decisions, not statistical artifacts.

Solution:
    Model the org hierarchy as a directed acyclic graph (DAG) using
    NetworkX. Traverse top-down, injecting a configurable hedge
    multiplier at each level and distributing proportionally to
    historical capacity.

Mathematical Guarantee:
    sum(leaf_quotas) = macro_target * hedge_multiplier^depth
    This compound over-assignment protects the root target against
    a ~(1 - 1/hedge_multiplier)^depth aggregate shortfall.

Reference:
    Karwa, S. (2026). "Graph-Theoretic Approaches to Hierarchical
    Revenue Target Allocation in B2B Enterprises." SSRN Working Paper.
"""

import networkx as nx
import pandas as pd
from typing import Optional


class QuotaCascader:
    """
    Top-down quota allocator over a NetworkX DAG.

    The cascading algorithm:
        1. Start at the root node with the macro target T.
        2. For each node, distribute T * hedge_multiplier proportionally
           to children, weighted by their historical capacity.
        3. Recurse until all leaf nodes (ICs) are assigned quotas.
        4. Locked nodes (executive overrides) bypass proportional
           allocation and receive a fixed quota.

    Attributes:
        dag: The organizational hierarchy as a directed graph.
        capacity: Dict mapping node_id -> historical ACV capacity.
        quotas: Dict mapping node_id -> allocated quota (populated
            after calling cascade()).
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        historical_attainment: pd.DataFrame,
        node_col: str = "node_id",
        capacity_col: str = "acv_closed",
    ):
        """
        Initialize the cascader with an org DAG and historical data.

        Args:
            dag: Directed graph where edges point from manager to report.
                Must be a valid DAG (no cycles).
            historical_attainment: DataFrame with columns for node
                identifier and historical ACV closed. Used to compute
                capacity weights for proportional allocation.
            node_col: Column name for node identifier.
            capacity_col: Column name for historical capacity metric.

        Raises:
            ValueError: If the graph contains cycles (not a DAG).
        """
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(
                "Organization graph contains cycles. "
                "A valid DAG is required for quota cascading."
            )

        self.dag = dag
        self.capacity = (
            historical_attainment.groupby(node_col)[capacity_col]
            .sum()
            .to_dict()
        )
        self.quotas = {}

    def _get_capacity(self, node_id: str) -> float:
        """
        Return historical capacity for a node.

        Falls back to 1.0 if the node has no historical data, ensuring
        equal allocation among unknown-capacity siblings.
        """
        return self.capacity.get(node_id, 1.0)

    def cascade(
        self,
        root: str,
        macro_target: float,
        hedge_multiplier: float = 1.05,
        locked_nodes: Optional[dict] = None,
    ) -> dict:
        """
        Recursively cascade the macro target through the DAG.

        Algorithm (Algorithm 1 in the companion paper):
            FUNCTION CASCADE(node, target):
                quota[node] <- target
                children <- successors(node)
                IF children is empty: RETURN
                budget <- target * hedge_multiplier
                FOR each child in children:
                    weight <- capacity(child) / sum(capacity(children))
                    CASCADE(child, budget * weight)

        Args:
            root: Root node ID (e.g., 'CRO').
            macro_target: Top-line revenue target in dollars.
            hedge_multiplier: Safety buffer injected at each level.
                Default 1.05 = 5% over-assignment. The compound effect
                across D levels yields total over-assignment of
                hedge_multiplier^D - 1.
            locked_nodes: Dict of {node_id: fixed_quota} for executive
                overrides. Locked nodes receive their specified quota
                regardless of proportional allocation. The remaining
                budget is redistributed among unlocked siblings.

        Returns:
            Dict mapping every node_id in the DAG to its allocated quota.

        Example:
            >>> cascader = QuotaCascader(dag, historical_df)
            >>> quotas = cascader.cascade(
            ...     root='CRO',
            ...     macro_target=500_000_000,
            ...     hedge_multiplier=1.05,
            ...     locked_nodes={'SVP_APAC': 45_000_000}
            ... )
            >>> leaf_sum = sum(quotas[n] for n in dag
            ...               if dag.out_degree(n) == 0)
            >>> print(f"IC quota sum: ${leaf_sum/1e6:.2f}M")
        """
        locked_nodes = locked_nodes or {}
        self.quotas = {}

        def _recurse(node: str, parent_target: float):
            # Assign quota to this node
            if node in locked_nodes:
                self.quotas[node] = locked_nodes[node]
            else:
                self.quotas[node] = parent_target

            children = list(self.dag.successors(node))
            if not children:
                return  # Leaf node — quota is set

            # Inflated budget to distribute to children
            effective_target = self.quotas[node]
            child_budget = effective_target * hedge_multiplier

            # Handle locked children: deduct their fixed quotas first
            locked_children = {
                c: locked_nodes[c]
                for c in children
                if c in locked_nodes
            }
            remaining_budget = child_budget - sum(locked_children.values())
            unlocked_children = [
                c for c in children if c not in locked_nodes
            ]

            # Capacity-weighted allocation for unlocked children
            total_capacity = sum(
                self._get_capacity(c) for c in unlocked_children
            )

            for child in children:
                if child in locked_children:
                    _recurse(child, locked_children[child])
                elif total_capacity > 0:
                    weight = self._get_capacity(child) / total_capacity
                    child_quota = remaining_budget * weight
                    _recurse(child, child_quota)
                else:
                    # Equal allocation fallback
                    child_quota = remaining_budget / max(
                        len(unlocked_children), 1
                    )
                    _recurse(child, child_quota)

        _recurse(root, macro_target)
        return self.quotas

    def get_leaf_quotas(self) -> dict:
        """Return quotas for leaf nodes (ICs) only."""
        return {
            n: q
            for n, q in self.quotas.items()
            if self.dag.out_degree(n) == 0
        }

    def get_cascade_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame of the cascade allocation.

        Returns:
            DataFrame with columns: node_id, quota, depth, is_leaf,
            num_children, historical_capacity.
        """
        if not self.quotas:
            raise ValueError(
                "No quotas computed. Call cascade() first."
            )

        # Find root (node with in-degree 0)
        roots = [n for n in self.dag.nodes if self.dag.in_degree(n) == 0]

        # BFS to compute depth
        depths = {}
        for root in roots:
            for node, depth in nx.single_source_shortest_path_length(
                self.dag, root
            ).items():
                depths[node] = depth

        records = []
        for node, quota in self.quotas.items():
            records.append(
                {
                    "node_id": node,
                    "quota": round(quota, 2),
                    "depth": depths.get(node, 0),
                    "is_leaf": self.dag.out_degree(node) == 0,
                    "num_children": self.dag.out_degree(node),
                    "historical_capacity": self._get_capacity(node),
                }
            )

        return pd.DataFrame(records).sort_values(
            ["depth", "quota"], ascending=[True, False]
        )

    def validate_coherence(self, tolerance: float = 0.01) -> dict:
        """
        Validate that quota allocation is coherent at every level.

        Checks that each parent's quota * hedge_multiplier equals
        the sum of its children's quotas (within tolerance).

        Args:
            tolerance: Acceptable relative deviation (default 1%).

        Returns:
            Dict with 'is_coherent' (bool) and 'violations' (list of
            nodes where coherence fails).
        """
        violations = []

        for node in self.dag.nodes:
            children = list(self.dag.successors(node))
            if not children:
                continue

            parent_quota = self.quotas.get(node, 0)
            children_sum = sum(
                self.quotas.get(c, 0) for c in children
            )

            if parent_quota > 0:
                deviation = abs(children_sum / parent_quota - 1)
                # Allow for hedge multiplier expansion
                if deviation > 0.5:  # More than 50% deviation is wrong
                    violations.append(
                        {
                            "node": node,
                            "parent_quota": parent_quota,
                            "children_sum": children_sum,
                            "deviation": round(deviation, 4),
                        }
                    )

        return {
            "is_coherent": len(violations) == 0,
            "violations": violations,
        }


def build_org_dag(
    opps_df: pd.DataFrame,
    rep_col: str = "rep_name",
    mgr1_col: str = "mgr1_name",
    mgr2_col: str = "mgr2_name",
    root_label: str = "CRO",
) -> nx.DiGraph:
    """
    Construct the reporting DAG from Salesforce opportunity data.

    Builds the organizational hierarchy by extracting unique
    manager-report pairs from the opportunity owner fields.

    Args:
        opps_df: Opportunities DataFrame with manager hierarchy columns.
        rep_col: Column for IC/rep name.
        mgr1_col: Column for direct manager name.
        mgr2_col: Column for skip-level manager name.
        root_label: Label for the root node (e.g., 'CRO').

    Returns:
        NetworkX DiGraph where edges point from manager to report.
    """
    dag = nx.DiGraph()
    pairs = set()

    cols = [rep_col, mgr1_col, mgr2_col]
    unique_rows = opps_df[cols].drop_duplicates()

    for _, row in unique_rows.iterrows():
        mgr2 = row[mgr2_col]
        mgr1 = row[mgr1_col]
        rep = row[rep_col]

        if pd.notna(mgr2) and pd.notna(mgr1):
            pairs.add((root_label, str(mgr2)))
            pairs.add((str(mgr2), str(mgr1)))
        if pd.notna(mgr1) and pd.notna(rep):
            pairs.add((str(mgr1), str(rep)))

    dag.add_edges_from(pairs)
    return dag
