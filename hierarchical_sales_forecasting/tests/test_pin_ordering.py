"""
Tests for issue #41 — pin list order must never matter.

The reported symptom (leaf pin overwritten by a later manager-pin
subtree rescale) was fixed by v0.20.0's protection-aware rescale.
v0.22.0 finishes the job: pins are applied in canonical DEPTH order
(shallowest pinned node first, stable within a depth), so the entire
output frame — absorber rows included — is bit-identical for any list
order, and the feasibility report follows the INPUT order.
"""
import itertools
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec, Pin, apply_pins

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    """3 teams x 2 reps, 2 quarters — the filer's notebook shape."""
    hdf = pd.DataFrame([
        dict(product='cloud', regional='EMEA', team=f'T{i//2+1}',
             rep=f'r{i+1}', kw=[100, 200, 300, 400, 250, 350][i])
        for i in range(6)])
    targets = pd.DataFrame([dict(product='cloud', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['product'],
                        target_col='tgt',
                        taxonomy=['regional', 'team', 'rep'], metrics=KW)
    return q


PINS = {
    'region': Pin('EMEA', 2_200_000.0),   # depth-1 manager
    'team':   Pin('T1', 700_000.0),       # depth-2 manager
    'rep':    Pin('r3', 250_000.0),       # leaf
    'zero':   Pin('r5', 0.0),             # zeroed leaf
}


# ----------------------------------------------------------------------
# 1. The literal #41 symptom: leaf pin first, manager pin after
# ----------------------------------------------------------------------
def test_leaf_first_not_overwritten():
    print(SEPARATOR)
    print("TEST 1: leaf pin listed BEFORE the covering manager pin "
          "still holds")
    print(SEPARATOR)
    q = _quotas()
    e, _ = apply_pins(q, [Pin('r1', 90_000.0), Pin('T1', 700_000.0)])
    e = e.set_index(['node_id', 'fiscal_quarter'])
    r1 = e.loc['r1', 'base_quota'].sum()
    t1 = e.loc['T1', 'base_quota'].sum()
    print(f"  r1 total={r1:,.2f} (pinned 90,000)  T1 total={t1:,.2f} "
          f"(pinned 700,000)")
    assert abs(r1 - 90_000.0) < 0.05
    assert abs(t1 - 700_000.0) < 0.05
    for fq in (1, 2):
        assert abs(e.loc[('T1', fq), 'base_quota']
                   - e.loc[('r1', fq), 'base_quota']
                   - e.loc[('r2', fq), 'base_quota']) < 0.05


# ----------------------------------------------------------------------
# 2. All 24 orderings -> bit-identical frames, pins held, conserved
# ----------------------------------------------------------------------
def test_all_orderings_identical():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: every permutation of 4 pins (region/team/rep/zero) "
          "yields the SAME frame")
    print(SEPARATOR)
    q = _quotas()
    frames = []
    for order in itertools.permutations(PINS):
        e, _ = apply_pins(q, [PINS[k] for k in order])
        e = e.set_index(['node_id', 'fiscal_quarter']).sort_index()
        assert abs(e.loc['EMEA', 'base_quota'].sum() - 2_200_000.0) < 0.05
        assert abs(e.loc['T1', 'base_quota'].sum() - 700_000.0) < 0.05
        assert abs(e.loc['r3', 'base_quota'].sum() - 250_000.0) < 0.05
        assert abs(e.loc['r5', 'base_quota'].sum()) < 0.05
        for fq in (1, 2):   # structural conservation everywhere
            for p, kids in [('EMEA', ['T1', 'T2', 'T3']),
                            ('T1', ['r1', 'r2']), ('T2', ['r3', 'r4']),
                            ('T3', ['r5', 'r6'])]:
                assert abs(e.loc[(p, fq), 'base_quota']
                           - sum(e.loc[(c, fq), 'base_quota']
                                 for c in kids)) < 0.05
        frames.append(e['base_quota'].round(2))
    assert all(f.equals(frames[0]) for f in frames)
    print(f"  {len(frames)} orderings, one unique result "
          f"(absorber rows included)")


# ----------------------------------------------------------------------
# 3. Feasibility report follows the INPUT pin order
# ----------------------------------------------------------------------
def test_report_in_input_order():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: report rows match the caller's pin list order")
    print(SEPARATOR)
    q = _quotas()
    listed = ['zero', 'rep', 'region', 'team']    # deliberately jumbled
    _, rep = apply_pins(q, [PINS[k] for k in listed])
    got = list(rep['pin_node'])
    expected = [PINS[k].node for k in listed]
    print(f"  report order: {got}")
    assert got == expected
    # root pin has no siblings to absorb its delta -> honestly reported
    r = rep.set_index('pin_node')
    assert not bool(r.loc['EMEA', 'feasible'])
    assert abs(r.loc['EMEA', 'unabsorbed'] - 200_000.0) < 0.5
    assert r.loc[['r5', 'r3', 'T1'], 'feasible'].all()


# ----------------------------------------------------------------------
# 4. Same-depth pins keep list order (stable), values unaffected
# ----------------------------------------------------------------------
def test_same_depth_stable():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: two leaf pins in either order -> identical frames")
    print(SEPARATOR)
    q = _quotas()
    p1, p2 = Pin('r1', 90_000.0), Pin('r4', 300_000.0)
    a, _ = apply_pins(q, [p1, p2])
    b, _ = apply_pins(q, [p2, p1])
    a = a.set_index(['node_id', 'fiscal_quarter']).sort_index()
    b = b.set_index(['node_id', 'fiscal_quarter']).sort_index()
    assert a['base_quota'].round(2).equals(b['base_quota'].round(2))
    assert abs(a.loc['r1', 'base_quota'].sum() - 90_000.0) < 0.05
    assert abs(a.loc['r4', 'base_quota'].sum() - 300_000.0) < 0.05
    print("  identical")


if __name__ == '__main__':
    test_leaf_first_not_overwritten()
    test_all_orderings_identical()
    test_report_in_input_order()
    test_same_depth_stable()

    print(f"\n\n{SEPARATOR}")
    print("ALL PIN-ORDERING TESTS PASSED")
    print(SEPARATOR)
