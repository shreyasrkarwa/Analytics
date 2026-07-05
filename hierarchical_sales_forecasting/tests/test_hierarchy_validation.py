"""
Tests for issue #1 — RecursionError when a hierarchy row has equal
adjacent level values (self-loops / cycles).

Covers:
  - on_collision='suffix' (default): auto-rename + warning, cascade works
  - on_collision='skip': duplicate level dropped, cascade works
  - on_collision='error': raises HierarchyValidationError
  - blank / "nan" string levels treated as missing (no shared "nan" node)
  - cross-row cycles caught by validate() with a clear error
  - cascade_quota fails fast (clear error) on manually built cyclic graphs
  - recursive aggregation helpers raise instead of RecursionError
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    HierarchyValidationError,
)

SEPARATOR = "=" * 90

TAXONOMY = ['regional', 'node_3_region', 'node_4_team', 'node_5_rep_no']
KW_METRIC = [MetricSpec('knowledge_workers', direction='proportional',
                        weight=1.0, columns=['knowledge_workers'])]


def _issue1_df():
    """The exact reproduction rows from issue #1 (team == rep -> self-loop)."""
    return pd.DataFrame([
        dict(regional='Ent_AMER', node_3_region='East', node_4_team='T1',
             node_5_rep_no='T1', knowledge_workers=100),
        dict(regional='Ent_AMER', node_3_region='East', node_4_team='T1',
             node_5_rep_no='rep2', knowledge_workers=50),
    ])


# ----------------------------------------------------------------------
# 1. Issue #1 repro — default 'suffix' policy: no crash, target lands
# ----------------------------------------------------------------------
def test_issue1_repro_suffix_default():
    print(SEPARATOR)
    print("TEST 1: Issue #1 repro — on_collision='suffix' (default) fixes the "
          "self-loop and cascades cleanly")
    print(SEPARATOR)
    h = SalesHierarchy()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        h.from_dataframe(_issue1_df(), path_cols=TAXONOMY,
                         metrics_cols=['knowledge_workers'])
    # A collision warning was emitted
    assert any('duplicate-level' in str(w.message) for w in wlog)
    # The rep was renamed deterministically; no self-loop exists
    assert 'T1__node_5_rep_no' in h.graph.nodes
    assert not h.graph.has_edge('T1', 'T1')

    c = QuotaCascader(h)
    q = c.cascade_quota('Ent_AMER', 1_000_000.0, metrics=KW_METRIC,
                        verbose=False)   # <- crashed with RecursionError pre-0.6.0
    print(f"  T1__node_5_rep_no: ${q['T1__node_5_rep_no']:,.2f} (expected ~$666,666.67)")
    print(f"  rep2:              ${q['rep2']:,.2f} (expected ~$333,333.33)")
    assert abs(q['T1__node_5_rep_no'] - 666_666.67) < 1.0
    assert abs(q['rep2'] - 333_333.33) < 1.0
    # Full reconciliation still holds
    report = c.reconciliation_report(q, target=1_000_000.0, strict=True)
    assert report['reconciles'].all()
    print("  reconciliation_report: all depths reconcile")


# ----------------------------------------------------------------------
# 2. on_collision='skip' — duplicate level dropped from the path
# ----------------------------------------------------------------------
def test_issue1_skip_policy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: on_collision='skip' — duplicate level dropped, T1 stays one node")
    print(SEPARATOR)
    h = SalesHierarchy()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        h.from_dataframe(_issue1_df(), path_cols=TAXONOMY,
                         metrics_cols=['knowledge_workers'],
                         on_collision='skip')
    assert any('dropped from the path' in str(w.message) for w in wlog)
    assert 'T1__node_5_rep_no' not in h.graph.nodes
    assert not h.graph.has_edge('T1', 'T1')
    # T1 survives as a single node with rep2 as its child; row 1's metrics
    # attached to T1 itself (the surviving deepest node of that row).
    assert h.graph.has_edge('T1', 'rep2')
    assert h.graph.nodes['T1'].get('knowledge_workers') == 100
    print("  T1 kept as one node; edge T1->rep2 intact; metrics on T1")


# ----------------------------------------------------------------------
# 3. on_collision='error' — fail fast with row context
# ----------------------------------------------------------------------
def test_issue1_error_policy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: on_collision='error' raises HierarchyValidationError")
    print(SEPARATOR)
    h = SalesHierarchy()
    try:
        h.from_dataframe(_issue1_df(), path_cols=TAXONOMY,
                         metrics_cols=['knowledge_workers'],
                         on_collision='error')
        raise AssertionError("expected HierarchyValidationError")
    except HierarchyValidationError as e:
        assert 'T1' in str(e) and 'Row 0' in str(e)
        print(f"  Raised as expected: {e}")


# ----------------------------------------------------------------------
# 4. Blank / "nan" string levels are treated as missing
# ----------------------------------------------------------------------
def test_blank_and_nan_strings_do_not_become_nodes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: blank / 'nan' string levels (keep_default_na=False) don't "
          "collapse into a shared node")
    print(SEPARATOR)
    df = pd.DataFrame([
        # jagged rows: middle levels blank or literal 'nan' strings
        dict(regional='Corp', node_3_region='', node_4_team='nan',
             node_5_rep_no='rep_a', knowledge_workers=10),
        dict(regional='Corp', node_3_region='West', node_4_team='WT1',
             node_5_rep_no='rep_b', knowledge_workers=30),
        # 'NA' is a REAL region name (North America) — must be kept as data
        dict(regional='Corp', node_3_region='NA', node_4_team='NAT1',
             node_5_rep_no='rep_c', knowledge_workers=0),
    ])
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY, metrics_cols=['knowledge_workers'])
    assert 'nan' not in h.graph.nodes and '' not in h.graph.nodes
    # 'NA' survives as a real node (keep_default_na=False use case)
    assert 'NA' in h.graph.nodes and h.graph.has_edge('Corp', 'NA')
    # rep_a hangs directly off Corp (jagged), rep_b under West/WT1
    assert h.graph.has_edge('Corp', 'rep_a')
    assert h.graph.has_edge('WT1', 'rep_b')
    c = QuotaCascader(h)
    q = c.cascade_quota('Corp', 400_000.0, metrics=KW_METRIC, verbose=False)
    print(f"  rep_a: ${q['rep_a']:,.2f} (expected $100,000)")
    print(f"  rep_b: ${q['rep_b']:,.2f} (expected $300,000)")
    assert abs(q['rep_a'] - 100_000.0) < 0.01
    assert abs(q['rep_b'] - 300_000.0) < 0.01


# ----------------------------------------------------------------------
# 5. Cross-row cycle (name reused at two different levels) — clear error
# ----------------------------------------------------------------------
def test_cross_row_cycle_caught_by_validate():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: cross-row cycle (A->B in one row, B->A in another) raises "
          "a clear HierarchyValidationError")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(L1='A', L2='B'),
        dict(L1='B', L2='A'),   # no per-row duplicate, but a cross-row cycle
    ])
    h = SalesHierarchy()
    try:
        h.from_dataframe(df, path_cols=['L1', 'L2'])
        raise AssertionError("expected HierarchyValidationError")
    except HierarchyValidationError as e:
        assert 'cycle' in str(e).lower()
        print(f"  Raised as expected: {e}")


# ----------------------------------------------------------------------
# 6. Manually built cyclic graph — cascade_quota fails fast, clearly
# ----------------------------------------------------------------------
def test_manual_cycle_fails_fast_in_cascade():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: manually built self-loop -> cascade_quota raises "
          "HierarchyValidationError (no RecursionError)")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('Root'); h.add_node('X', {'knowledge_workers': 10})
    h.add_edge('Root', 'X')
    h.add_edge('X', 'X')          # self-loop, bypassing from_dataframe
    c = QuotaCascader(h)
    try:
        c.cascade_quota('Root', 100_000.0, metrics=KW_METRIC, verbose=False)
        raise AssertionError("expected HierarchyValidationError")
    except HierarchyValidationError as e:
        assert 'cycle' in str(e).lower()
        print(f"  cascade_quota raised as expected: {e}")
    # validate() names the same problem
    try:
        h.validate()
        raise AssertionError("expected HierarchyValidationError")
    except HierarchyValidationError as e:
        print(f"  validate() raised as expected: {e}")


# ----------------------------------------------------------------------
# 7. Recursive helpers guard the recursion stack (no RecursionError)
# ----------------------------------------------------------------------
def test_aggregation_helpers_raise_cleanly_on_cycle():
    print(f"\n\n{SEPARATOR}")
    print("TEST 7: aggregation helpers raise HierarchyValidationError on a "
          "cycle instead of RecursionError")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('A'); h.add_node('B')
    h.add_edge('A', 'B'); h.add_edge('B', 'A')   # 2-cycle
    c = QuotaCascader(h)
    for fn, label in [
        (lambda: c._aggregate_node_metric('A', KW_METRIC[0]), '_aggregate_node_metric'),
        (lambda: c._calculate_node_historical_capacity('A'), '_calculate_node_historical_capacity'),
    ]:
        try:
            fn()
            raise AssertionError(f"expected HierarchyValidationError from {label}")
        except HierarchyValidationError:
            print(f"  {label}: raised HierarchyValidationError as expected")


# ----------------------------------------------------------------------
# 8. Diamond-shaped DAGs (node with two parents) still work
# ----------------------------------------------------------------------
def test_diamond_dag_still_allowed():
    print(f"\n\n{SEPARATOR}")
    print("TEST 8: diamond DAG (node reachable via two branches) does NOT "
          "trip the cycle guard")
    print(SEPARATOR)
    h = SalesHierarchy()
    for n in ['Top', 'L', 'R']:
        h.add_node(n)
    h.add_node('Leaf', {'knowledge_workers': 5})
    h.add_edge('Top', 'L'); h.add_edge('Top', 'R')
    h.add_edge('L', 'Leaf'); h.add_edge('R', 'Leaf')
    h.validate()   # must not raise
    c = QuotaCascader(h)
    # Aggregation must not raise (Leaf legitimately counted via both paths,
    # matching pre-0.6.0 behavior)
    total = c._aggregate_node_metric('Top', KW_METRIC[0])
    print(f"  Top-level aggregate: {total} (Leaf counted via both branches)")
    assert total == 10.0


if __name__ == '__main__':
    test_issue1_repro_suffix_default()
    test_issue1_skip_policy()
    test_issue1_error_policy()
    test_blank_and_nan_strings_do_not_become_nodes()
    test_cross_row_cycle_caught_by_validate()
    test_manual_cycle_fails_fast_in_cascade()
    test_aggregation_helpers_raise_cleanly_on_cycle()
    test_diamond_dag_still_allowed()

    print(f"\n\n{SEPARATOR}")
    print("ALL HIERARCHY-VALIDATION TESTS PASSED")
    print(SEPARATOR)
