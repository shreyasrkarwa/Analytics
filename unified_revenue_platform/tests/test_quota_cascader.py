"""Tests for the DAG-based Quota Cascader."""

import pytest
import networkx as nx
import pandas as pd

from unified_revenue_platform.quota_cascader import QuotaCascader, build_org_dag


@pytest.fixture
def simple_dag():
    """Create a simple 3-level hierarchy: CRO -> 2 VPs -> 4 ICs."""
    dag = nx.DiGraph()
    dag.add_edges_from([
        ("CRO", "VP_AMER"),
        ("CRO", "VP_EMEA"),
        ("VP_AMER", "IC_1"),
        ("VP_AMER", "IC_2"),
        ("VP_EMEA", "IC_3"),
        ("VP_EMEA", "IC_4"),
    ])
    return dag


@pytest.fixture
def historical_df():
    """Historical attainment data for capacity-weighted allocation."""
    return pd.DataFrame({
        "node_id": ["IC_1", "IC_2", "IC_3", "IC_4"],
        "acv_closed": [100_000, 200_000, 150_000, 50_000],
    })


@pytest.fixture
def cascader(simple_dag, historical_df):
    return QuotaCascader(dag=simple_dag, historical_attainment=historical_df)


class TestQuotaCascader:
    """Test suite for QuotaCascader."""

    def test_basic_cascade(self, cascader):
        """Cascade should assign quotas to all nodes."""
        quotas = cascader.cascade(root="CRO", macro_target=1_000_000)
        assert "CRO" in quotas
        assert quotas["CRO"] == 1_000_000
        # All 6 nodes should have quotas
        assert len(quotas) == 6

    def test_leaf_quotas_sum_exceeds_target(self, cascader):
        """Leaf quota sum should exceed macro target due to hedge."""
        quotas = cascader.cascade(
            root="CRO",
            macro_target=1_000_000,
            hedge_multiplier=1.05,
        )
        leaf_sum = sum(
            q for n, q in quotas.items()
            if cascader.dag.out_degree(n) == 0
        )
        # With 2 levels of 1.05 hedging: 1M * 1.05^2 = $1,102,500
        assert leaf_sum > 1_000_000
        assert abs(leaf_sum - 1_000_000 * 1.05 ** 2) < 1.0

    def test_no_hedge(self, cascader):
        """With hedge=1.0, leaf sum should equal macro target."""
        quotas = cascader.cascade(
            root="CRO",
            macro_target=1_000_000,
            hedge_multiplier=1.0,
        )
        leaf_sum = sum(
            q for n, q in quotas.items()
            if cascader.dag.out_degree(n) == 0
        )
        assert abs(leaf_sum - 1_000_000) < 1.0

    def test_capacity_weighted_allocation(self, cascader):
        """ICs with higher historical capacity get proportionally more."""
        quotas = cascader.cascade(
            root="CRO",
            macro_target=1_000_000,
            hedge_multiplier=1.0,
        )
        # IC_2 has 2x the capacity of IC_1, so should get 2x the quota
        # within AMER (they share the same parent)
        assert quotas["IC_2"] > quotas["IC_1"]
        ratio = quotas["IC_2"] / quotas["IC_1"]
        assert abs(ratio - 2.0) < 0.01

    def test_locked_nodes(self, cascader):
        """Locked nodes should receive their fixed quota."""
        quotas = cascader.cascade(
            root="CRO",
            macro_target=1_000_000,
            hedge_multiplier=1.05,
            locked_nodes={"VP_EMEA": 300_000},
        )
        assert quotas["VP_EMEA"] == 300_000

    def test_cyclic_graph_rejected(self, historical_df):
        """A graph with cycles should raise ValueError."""
        cyclic = nx.DiGraph()
        cyclic.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(ValueError, match="cycles"):
            QuotaCascader(dag=cyclic, historical_attainment=historical_df)

    def test_get_leaf_quotas(self, cascader):
        """get_leaf_quotas should return only IC-level quotas."""
        cascader.cascade(root="CRO", macro_target=1_000_000)
        leaves = cascader.get_leaf_quotas()
        assert len(leaves) == 4
        assert all(k.startswith("IC_") for k in leaves)

    def test_cascade_summary_dataframe(self, cascader):
        """get_cascade_summary should return a well-formed DataFrame."""
        cascader.cascade(root="CRO", macro_target=1_000_000)
        summary = cascader.get_cascade_summary()
        assert len(summary) == 6
        assert "node_id" in summary.columns
        assert "quota" in summary.columns
        assert "depth" in summary.columns
        assert "is_leaf" in summary.columns

    def test_validate_coherence_passes(self, cascader):
        """A properly cascaded hierarchy should pass coherence check."""
        cascader.cascade(root="CRO", macro_target=1_000_000)
        result = cascader.validate_coherence()
        assert result["is_coherent"] is True

    def test_single_ic(self):
        """Edge case: hierarchy with a single IC."""
        dag = nx.DiGraph()
        dag.add_edge("CRO", "IC_1")
        hist = pd.DataFrame({"node_id": ["IC_1"], "acv_closed": [100_000]})
        cascader = QuotaCascader(dag=dag, historical_attainment=hist)
        quotas = cascader.cascade(root="CRO", macro_target=1_000_000, hedge_multiplier=1.05)
        assert quotas["IC_1"] == 1_000_000 * 1.05


class TestBuildOrgDag:
    """Test suite for build_org_dag helper."""

    def test_builds_valid_dag(self):
        """build_org_dag should produce a valid DAG from opportunity data."""
        opps_df = pd.DataFrame({
            "rep_name": ["Alice", "Bob", "Charlie"],
            "mgr1_name": ["Mgr_X", "Mgr_X", "Mgr_Y"],
            "mgr2_name": ["VP_AMER", "VP_AMER", "VP_EMEA"],
        })
        dag = build_org_dag(opps_df)
        assert nx.is_directed_acyclic_graph(dag)
        assert "CRO" in dag.nodes
        assert dag.has_edge("CRO", "VP_AMER")
        assert dag.has_edge("Mgr_X", "Alice")

    def test_handles_missing_managers(self):
        """Should handle rows with NaN manager fields gracefully."""
        opps_df = pd.DataFrame({
            "rep_name": ["Alice", "Bob"],
            "mgr1_name": ["Mgr_X", None],
            "mgr2_name": [None, None],
        })
        dag = build_org_dag(opps_df)
        assert nx.is_directed_acyclic_graph(dag)
