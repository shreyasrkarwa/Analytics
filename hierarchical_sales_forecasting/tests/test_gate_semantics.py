"""
Tests for issue #9 — configurable gate threshold & semantics (gate_mode).

Covers:
  - default "gt" is byte-for-byte the pre-0.9.0 behavior
  - "ge" boundary semantics ("at least N seats")
  - "lt"/"le" gate values that are TOO HIGH (churn-style gates)
  - "truthy" for boolean entitlement flags
  - invalid gate_mode raises at construction
  - lt-gate parent/child divergence is absorbed by gate_fallback
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import SalesHierarchy, QuotaCascader, MetricSpec

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]


def _build(seats):
    rows = [dict(regional='AMER', team=f'T{i//2 + 1}', rep=f'r{i+1}',
                 kw=100, seats=s) for i, s in enumerate(seats)]
    h = SalesHierarchy()
    h.from_dataframe(pd.DataFrame(rows), path_cols=TAXONOMY,
                     metrics_cols=['kw', 'seats'])
    return h


def _run(h, gate):
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW,
                        gate_metrics=[gate], verbose=False)
    return c, q


# ----------------------------------------------------------------------
# 1. Default "gt" == pre-0.9.0 behavior exactly
# ----------------------------------------------------------------------
def test_default_gt_unchanged():
    print(SEPARATOR)
    print("TEST 1: default gate_mode='gt' — gated iff value <= threshold "
          "(pre-0.9.0 behavior)")
    print(SEPARATOR)
    h = _build([0, 10, 5, 0])
    c, q = _run(h, MetricSpec('seats', columns=['seats']))
    assert q['r1'] == 0.0 and q['r4'] == 0.0
    assert q['r2'] > 0 and q['r3'] > 0
    assert {'r1', 'r4'} <= c.gated_nodes
    print(f"  r1/r4 (0 seats) gated; r2/r3 funded — unchanged")


# ----------------------------------------------------------------------
# 2. "ge" boundary — at least N seats
# ----------------------------------------------------------------------
def test_ge_at_least_n():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: gate_threshold=5, gate_mode='ge' — exactly 5 seats PASSES")
    print(SEPARATOR)
    # Teams: T1=(r1: 4, r2: 100), T2=(r3: 5 boundary, r4: 100). Passing
    # siblings ensure the boundary rep's fate is decided by the gate
    # itself, not by the all-children-gated fallback.
    h = _build([4, 100, 5, 100])
    _, q_ge = _run(h, MetricSpec('seats', columns=['seats'],
                                 gate_threshold=5, gate_mode='ge'))
    assert q_ge['r1'] == 0.0            # 4 < 5 -> gated
    assert q_ge['r3'] > 0               # 5 >= 5 -> passes (the ge point)
    # Contrast: 'gt' at the same threshold gates the boundary value
    h2 = _build([4, 100, 5, 100])
    c_gt, q_gt = _run(h2, MetricSpec('seats', columns=['seats'],
                                     gate_threshold=5, gate_mode='gt'))
    assert q_gt['r3'] == 0.0            # 5 > 5 is False -> gated
    assert 'r3' in c_gt.gated_nodes and q_gt['r4'] > 0
    print("  ge: r3 (5 seats) funded · gt: r3 gated — boundary semantics differ")


# ----------------------------------------------------------------------
# 3. "le" — gate reps with TOO MUCH of a signal (churn tickets)
# ----------------------------------------------------------------------
def test_le_gates_high_values():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: gate_threshold=100, gate_mode='le' — churn>100 gated")
    print(SEPARATOR)
    h = _build([20, 250, 40, 90])       # 'seats' plays churn tickets here
    c, q = _run(h, MetricSpec('churn', columns=['seats'],
                              gate_threshold=100, gate_mode='le'))
    assert q['r2'] == 0.0 and 'r2' in c.gated_nodes
    assert q['r1'] > 0 and q['r3'] > 0 and q['r4'] > 0
    total = q['r1'] + q['r3'] + q['r4']
    print(f"  r2 (250 tickets) gated; survivors carry ${total:,.2f}")
    assert abs(total - 1_000_000.0) < 0.01


# ----------------------------------------------------------------------
# 4. "truthy" — boolean entitlement flags
# ----------------------------------------------------------------------
def test_truthy_flags():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: gate_mode='truthy' with True/False entitlement flags")
    print(SEPARATOR)
    rows = [
        dict(regional='AMER', team='T1', rep='r1', kw=100, ent=True),
        dict(regional='AMER', team='T1', rep='r2', kw=100, ent=False),
        dict(regional='AMER', team='T2', rep='r3', kw=100, ent=True),
    ]
    h = SalesHierarchy()
    h.from_dataframe(pd.DataFrame(rows), path_cols=TAXONOMY,
                     metrics_cols=['kw', 'ent'])
    c, q = _run(h, MetricSpec('ent', columns=['ent'], gate_mode='truthy'))
    assert q['r2'] == 0.0 and 'r2' in c.gated_nodes
    # Team split by kw rollup (T1: 200 incl. the gated rep, T2: 100) —
    # standard blend semantics; r1 then absorbs ALL of T1's share.
    assert abs(q['r1'] - 666_666.67) < 1.0
    assert abs(q['r3'] - 333_333.33) < 1.0
    report = c.reconciliation_report(q, target=1_000_000.0, strict=True)
    assert report['reconciles'].all()
    print(f"  r1: ${q['r1']:,.2f} · r2: $0 (False) · r3: ${q['r3']:,.2f} — "
          f"reconciles at every depth")


# ----------------------------------------------------------------------
# 5. Invalid gate_mode raises at construction
# ----------------------------------------------------------------------
def test_invalid_mode_raises():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: invalid gate_mode raises ValueError immediately")
    print(SEPARATOR)
    try:
        MetricSpec('seats', columns=['seats'], gate_mode='above')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'gate_mode' in str(e)
        print(f"  Raised as expected: {e}")


# ----------------------------------------------------------------------
# 6. lt-gate parent/child divergence handled by gate_fallback
# ----------------------------------------------------------------------
def test_lt_gate_fallback_interaction():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: 'lt' gate — parent rollup exceeds threshold while "
          "children pass; target still reconciles")
    print(SEPARATOR)
    # Each rep has 60 tickets (< 100 -> passes) but the T1 team rollup is
    # 120 (fails). gate_fallback='redistribute' relaxes where needed so
    # nothing is stranded.
    h = _build([60, 60, 10, 10])
    c, q = _run(h, MetricSpec('churn', columns=['seats'],
                              gate_threshold=100, gate_mode='lt'))
    report = c.reconciliation_report(q, target=1_000_000.0, strict=True)
    assert report['reconciles'].all()
    print(f"  every depth reconciles to $1M "
          f"(gate_relaxed_nodes: {len(c.gate_relaxed_nodes)})")


if __name__ == '__main__':
    test_default_gt_unchanged()
    test_ge_at_least_n()
    test_le_gates_high_values()
    test_truthy_flags()
    test_invalid_mode_raises()
    test_lt_gate_fallback_interaction()

    print(f"\n\n{SEPARATOR}")
    print("ALL GATE-SEMANTICS TESTS PASSED")
    print(SEPARATOR)
