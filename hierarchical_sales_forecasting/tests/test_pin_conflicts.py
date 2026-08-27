"""
Tests for issue #63 — overlapping-pin conflict detection.

Covers:
  - case 1 (grand-total + scoped pins, same rep): default 'error'
    raises BEFORE anything applies, naming totals/scopes/rows/relation
  - case 3 (total + scoped-zero): same detection; 'allow' proceeds
    last-writer-wins but the report is now honest (defeated pin's
    achieved_total recomputed on the FINAL frame, feasible=False)
  - 'narrower_wins': scoped pins stand, broad total constrains the
    remainder — equals the hand-built explicit-pin composition;
    reconcile clean; report carries conflict + adjusted_total
  - narrower_wins guards: partial overlap, mutually-overlapping
    subsets, mixed basis, over-subscribed remainder, fully-covered
    unplaced remainder
  - case 2 (team pin + rep pins on the subtree) is NOT a conflict:
    deliberate composition, already loud via #55 overshoot_report +
    #39 subtree_shortfall — pinned here as receipts
  - the remainder-pins recipe (#42, sibling pins, no overlap) is
    untouched by the default
  - 'warn' emits the same text once and proceeds
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    hdf = pd.DataFrame([dict(region='EMEA', team='T', rep=r, kw=100)
                        for r in ['r1', 'r2', 'r3']])
    targets = pd.DataFrame([dict(region='EMEA', st=s, tgt=600_000.0)
                            for s in ('Exp', 'Mig', 'New')])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region', 'st'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
    return q


OVERLAP = [Pin('r1', 900_000.0),
           Pin('r1', 300_000.0, scope={'st': 'Exp'}),
           Pin('r1', 250_000.0, scope={'st': 'Mig'}),
           Pin('r1', 200_000.0, scope={'st': 'New'})]


# ----------------------------------------------------------------------
# 1. Default 'error': loud, named, nothing applied
# ----------------------------------------------------------------------
def test_error_default():
    print(SEPARATOR)
    print("TEST 1: case 1 raises by default with the full family "
          "description")
    print(SEPARATOR)
    q = _quotas()
    try:
        apply_pins(q, OVERLAP)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        assert "conflict on 'r1'" in msg
        assert '900,000.00' in msg and "scope={'st': 'Exp'}" in msg
        assert '(subset)' in msg and 'narrower_wins' in msg
    # identical scopes classified too
    try:
        apply_pins(q, [Pin('r1', 100.0, scope={'st': 'Exp'}),
                       Pin('r1', 200.0, scope={'st': 'Exp'})])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert '(identical)' in str(e)
    print("  raises with totals, scopes, row counts, relations")


# ----------------------------------------------------------------------
# 2. 'allow': last-writer-wins, but the report is honest now
# ----------------------------------------------------------------------
def test_allow_honest_report():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: 'allow' proceeds; defeated pin reports achieved "
          "750K, feasible=False (was 900K/True)")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, rep = apply_pins(q, OVERLAP, on_conflict='allow')
    assert abs(e[e.node_id == 'r1']['base_quota'].sum()
               - 750_000.0) < 0.5
    r0 = rep.iloc[0]
    assert abs(r0['achieved_total'] - 750_000.0) < 0.5
    assert not bool(r0['feasible'])
    assert 'overlaps pins' in str(r0['conflict'])
    assert rep.iloc[1]['feasible']            # scoped pins delivered
    # case 3 shape: total + scoped zero, listed total-first
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e3, rep3 = apply_pins(
            q, [Pin('r1', 900_000.0),
                Pin('r1', 0.0, scope={'st': 'Mig'})],
            on_conflict='allow')
    tot = e3[e3.node_id == 'r1']['base_quota'].sum()
    assert tot < 900_000.0 - 0.5              # under-delivered…
    assert not bool(rep3.iloc[0]['feasible'])  # …but no longer silent
    print("  final-frame audit closes the misleading feasible=True")


# ----------------------------------------------------------------------
# 3. narrower_wins == hand-built explicit pins
# ----------------------------------------------------------------------
def test_narrower_wins_equivalence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: narrower_wins — remainder to the uncovered slice; "
          "equals explicit pins; reconcile clean")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, rep = apply_pins(
            q, [Pin('r1', 900_000.0),
                Pin('r1', 300_000.0, scope={'st': 'Exp'}),
                Pin('r1', 250_000.0, scope={'st': 'Mig'})],
            on_conflict='narrower_wins')
        b, _ = apply_pins(
            q, [Pin('r1', 350_000.0, scope={'st': 'New'}),
                Pin('r1', 300_000.0, scope={'st': 'Exp'}),
                Pin('r1', 250_000.0, scope={'st': 'Mig'})])
    ax = a.set_index(['node_id', 'st']).sort_index()
    bx = b.set_index(['node_id', 'st']).sort_index()
    assert ((ax['base_quota'] - bx['base_quota']).abs() < 0.05).all()
    assert abs(a[a.node_id == 'r1']['base_quota'].sum()
               - 900_000.0) < 0.5
    assert reconcile(a, hedge=1.1)['ok'].all()
    r0 = rep.iloc[0]
    assert abs(r0['adjusted_total'] - 350_000.0) < 0.5
    assert abs(r0['achieved_total'] - 900_000.0) < 0.5   # original rows
    assert r0['feasible']
    # their case-3 semantic: Migration=0 excluded from the total
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c, _ = apply_pins(q, [Pin('r1', 800_000.0),
                              Pin('r1', 0.0, scope={'st': 'Mig'})],
                          on_conflict='narrower_wins')
    assert abs(c[c.node_id == 'r1']['base_quota'].sum()
               - 800_000.0) < 0.5            # full total, Mig at 0
    cx = c.set_index(['node_id', 'st'])
    assert abs(cx.loc[('r1', 'Mig'), 'base_quota']) < 0.05
    print("  identical frames; case 3 delivers the FULL total with "
          "Mig=0")


# ----------------------------------------------------------------------
# 4. narrower_wins guards
# ----------------------------------------------------------------------
def test_narrower_wins_guards():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: guards — oversubscribed, fully-covered unplaced, "
          "identical, mixed basis")
    print(SEPARATOR)
    q = _quotas()
    for pins, needle in [
        ([Pin('r1', 400_000.0), Pin('r1', 500_000.0,
                                    scope={'st': 'Exp'})],
         'EXCEEDING'),
        (OVERLAP, 'unplaced'),               # covers all rows, 150K left
        ([Pin('r1', 100.0, scope={'st': 'Exp'}),
          Pin('r1', 200.0, scope={'st': 'Exp'})], 'strict subset'),
        ([Pin('r1', 900_000.0),
          Pin('r1', 100.0, scope={'st': 'Exp'}, basis='cascaded')],
         'mix basis'),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                apply_pins(q, pins, on_conflict='narrower_wins')
            raise AssertionError(f'expected ValueError: {needle}')
        except ValueError as e:
            assert needle in str(e), (needle, str(e)[:120])
    print("  all four raise, naming the numbers")


# ----------------------------------------------------------------------
# 5. Case 2 receipts + recipe untouched + warn mode
# ----------------------------------------------------------------------
def test_cross_node_not_conflict():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: team+rep pins are composition (loud via #55, not "
          "#63); #42 recipe untouched; 'warn' proceeds")
    print(SEPARATOR)
    q = _quotas()
    # case 2: parent + children pins, jointly infeasible -> NOT a
    # conflict error; #55/#39 machinery is the (already loud) surface
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e2, _ = apply_pins(q, [Pin('T', 901_550.0, scope={'st': 'Exp'}),
                               Pin('r1', 400_000.0, scope={'st': 'Exp'}),
                               Pin('r2', 330_000.0, scope={'st': 'Exp'}),
                               Pin('r3', 261_705.0, scope={'st': 'Exp'})])
    assert e2.attrs['overshoot_report']
    assert any('collectively break' in str(x.message) for x in w)
    assert any('protected descendants' in str(x.message) for x in w)
    # remainder recipe (#42): sibling pins, disjoint -> silent
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter('always')
        er, _ = apply_pins(q, [Pin('r1', 500_000.0),
                               Pin('r2', 300_000.0)])
    assert not [x for x in w2 if 'conflict' in str(x.message)]
    # warn mode: one warning, proceeds
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter('always')
        ew, _ = apply_pins(q, OVERLAP, on_conflict='warn')
    hits = [x for x in w3 if 'overlapping pins' in str(x.message)]
    assert len(hits) == 1
    assert abs(ew[ew.node_id == 'r1']['base_quota'].sum()
               - 750_000.0) < 0.5
    print("  composition loud via the right channel; recipe + warn ok")


if __name__ == '__main__':
    test_error_default()
    test_allow_honest_report()
    test_narrower_wins_equivalence()
    test_narrower_wins_guards()
    test_cross_node_not_conflict()

    print(f"\n\n{SEPARATOR}")
    print("ALL PIN-CONFLICT TESTS PASSED")
    print(SEPARATOR)
