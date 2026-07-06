"""
Tests for issue #10 — consolidated gating_report().

Covers:
  - normal gated cascade: ids, counts, leaf split, reconciliation numbers
  - strand_at_root: unallocated amount accounted for; still reconciles
    (target == leaf base sum + unallocated)
  - redistribute-relaxed cascade: relaxed ids reported, no unallocated
  - hedged cascade: leaf_quota_sum vs leaf_base_sum diverge correctly
  - RuntimeError before any cascade
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import SalesHierarchy, QuotaCascader, MetricSpec

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]
GATE = [MetricSpec('seats', columns=['seats'])]


def _build(seats):
    rows = [dict(regional='AMER', team=f'T{i//2 + 1}', rep=f'r{i+1}',
                 kw=100, seats=s) for i, s in enumerate(seats)]
    h = SalesHierarchy()
    h.from_dataframe(pd.DataFrame(rows), path_cols=TAXONOMY,
                     metrics_cols=['kw', 'seats'])
    return h


# ----------------------------------------------------------------------
# 1. Normal gated cascade
# ----------------------------------------------------------------------
def test_report_basic():
    print(SEPARATOR)
    print("TEST 1: gated cascade — ids, counts, and reconciliation numbers")
    print(SEPARATOR)
    h = _build([10, 0, 5, 0])          # r2, r4 gated
    c = QuotaCascader(h)
    c.cascade_quota('AMER', 1_000_000.0, metrics=KW, gate_metrics=GATE,
                    verbose=False)
    rep = c.gating_report()
    print(f"  gated: {rep['gated_node_ids']} · leaf sum: "
          f"${rep['leaf_base_sum']:,.2f} · reconciles: {rep['reconciles']}")
    assert rep['target'] == 1_000_000.0
    assert rep['gated_leaf_ids'] == ['r2', 'r4']
    assert rep['gated_count'] == 2 and rep['gated_node_ids'] == ['r2', 'r4']
    assert rep['gate_relaxed_node_ids'] == []
    assert rep['unallocated_amount'] == 0.0
    assert abs(rep['leaf_base_sum'] - 1_000_000.0) < 0.01
    assert abs(rep['base_gap']) <= 0.01 and rep['reconciles']


# ----------------------------------------------------------------------
# 2. strand_at_root — every dollar accounted for
# ----------------------------------------------------------------------
def test_report_strand_at_root():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: strand_at_root — target == leaf base sum + unallocated")
    print(SEPARATOR)
    h = _build([0, 0, 0, 0])           # fully gated tree
    c = QuotaCascader(h)
    c.cascade_quota('AMER', 500_000.0, metrics=KW, gate_metrics=GATE,
                    gate_fallback='strand_at_root', verbose=False)
    rep = c.gating_report()
    print(f"  unallocated: ${rep['unallocated_amount']:,.2f} · leaf base "
          f"sum: ${rep['leaf_base_sum']:,.2f} · reconciles: {rep['reconciles']}")
    assert rep['unallocated_amount'] == 500_000.0
    assert rep['unallocated_nodes'] == {'AMER': 500_000.0}
    assert rep['leaf_base_sum'] == 0.0
    # Every dollar visible: 500k target = 0 on ICs + 500k reported stranded
    assert rep['reconciles']


# ----------------------------------------------------------------------
# 3. redistribute over a fully-gated tree — relaxed ids surface
# ----------------------------------------------------------------------
def test_report_relaxed():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: fully-gated tree with default fallback — relaxed nodes "
          "reported, nothing unallocated")
    print(SEPARATOR)
    h = _build([0, 0, 0, 0])
    c = QuotaCascader(h)
    c.cascade_quota('AMER', 500_000.0, metrics=KW, gate_metrics=GATE,
                    verbose=False)
    rep = c.gating_report()
    print(f"  relaxed: {len(rep['gate_relaxed_node_ids'])} node(s) · "
          f"leaf base sum: ${rep['leaf_base_sum']:,.2f}")
    assert rep['gate_relaxed_node_ids']          # non-empty
    assert rep['unallocated_amount'] == 0.0
    assert abs(rep['leaf_base_sum'] - 500_000.0) < 0.01
    assert rep['reconciles']


# ----------------------------------------------------------------------
# 4. Hedged cascade — hedged vs base leaf sums diverge correctly
# ----------------------------------------------------------------------
def test_report_hedged_sums():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: hedge=1.05 over 2 levels — leaf_quota_sum = base x 1.05^2")
    print(SEPARATOR)
    h = _build([10, 10, 10, 10])
    c = QuotaCascader(h)
    c.cascade_quota('AMER', 1_000_000.0, hedge_multiplier=1.05,
                    metrics=KW, gate_metrics=GATE, verbose=False)
    rep = c.gating_report()
    print(f"  hedged leaf sum: ${rep['leaf_quota_sum']:,.2f} · base: "
          f"${rep['leaf_base_sum']:,.2f}")
    assert abs(rep['leaf_base_sum'] - 1_000_000.0) < 0.01
    assert abs(rep['leaf_quota_sum'] - 1_000_000.0 * 1.05 ** 2) < 0.01
    assert rep['reconciles']           # reconciliation is on the BASE layer


# ----------------------------------------------------------------------
# 5. RuntimeError before any cascade
# ----------------------------------------------------------------------
def test_report_requires_cascade():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: gating_report before any cascade raises RuntimeError")
    print(SEPARATOR)
    c = QuotaCascader(_build([1, 1, 1, 1]))
    try:
        c.gating_report()
        raise AssertionError('expected RuntimeError')
    except RuntimeError as e:
        print(f"  Raised as expected: {e}")


if __name__ == '__main__':
    test_report_basic()
    test_report_strand_at_root()
    test_report_relaxed()
    test_report_hedged_sums()
    test_report_requires_cascade()

    print(f"\n\n{SEPARATOR}")
    print("ALL GATING-REPORT TESTS PASSED")
    print(SEPARATOR)
