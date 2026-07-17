"""
Tests for issue #57 — reallocate() (fraction / multi-source / explicit
split) and resplit_by_metric().

Covers:
  - the exact §3b ask: 75% of two reps -> 60/40 to two others, ONE
    call, equivalent to the hand-built 4-pin composition; conserved;
    reconcile clean
  - redistribute(x) == reallocate([x], fraction=1.0) identity
  - equal split + freeze; fraction validation; sibling validation
  - resplit_by_metric: equivalent to the hand-built per-cascade
    scoped-pin composition; frozen child held; subtree rescale;
    missing-metric error hints metadata_cols; reconcile clean
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, redistribute,
    reallocate, resplit_by_metric, reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    hdf = pd.DataFrame([dict(region='EMEA', team='T1', rep=f'r{i+1}',
                             kw=[100, 200, 300, 400][i],
                             dc=[9, 1, 4, 2][i]) for i in range(4)])
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW,
                        hedge_multiplier=1.1, metadata_cols=['dc'])
    return q


# ----------------------------------------------------------------------
# 1. The §3b ask, one call == hand pins
# ----------------------------------------------------------------------
def test_fraction_move_equivalence():
    print(SEPARATOR)
    print("TEST 1: 75% of r1+r2 -> 60/40 to r3/r4 == hand-built pins")
    print(SEPARATOR)
    q = _quotas()
    out, rep = reallocate(q, sources=['r1', 'r2'], fraction=0.75,
                          weights={'r3': 0.6, 'r4': 0.4})
    b = {n: q[q.node_id == n]['base_quota'].sum()
         for n in ('r1', 'r2', 'r3', 'r4')}
    moved = 0.75 * (b['r1'] + b['r2'])
    pins = [Pin('r1', 0.25 * b['r1']), Pin('r2', 0.25 * b['r2']),
            Pin('r3', b['r3'] + 0.6 * moved),
            Pin('r4', b['r4'] + 0.4 * moved)]
    manual, _ = apply_pins(q, pins)
    a = out.set_index(['node_id', 'fiscal_quarter']).sort_index()
    m = manual.set_index(['node_id', 'fiscal_quarter']).sort_index()
    assert ((a['base_quota'] - m['base_quota']).abs() < 0.05).all()
    assert abs(a.loc[('T1', 1), 'base_quota'] - 1_000_000.0) < 0.5
    assert reconcile(out, hedge=1.1)['ok'].all()
    r = rep.set_index('node')
    assert abs(r.loc['r1', 'target_total'] - 0.25 * b['r1']) < 0.05
    assert rep['exact'].all()
    print("  identical to 4 hand pins; conserved; reconcile ok")


# ----------------------------------------------------------------------
# 2. redistribute is the fraction=1 single-source special case
# ----------------------------------------------------------------------
def test_redistribute_identity():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: redistribute(x) == reallocate([x], fraction=1.0)")
    print(SEPARATOR)
    q = _quotas()
    a, _ = redistribute(q, 'r1')
    b, _ = reallocate(q, 'r1')
    ax = a.set_index(['node_id', 'fiscal_quarter']).sort_index()
    bx = b.set_index(['node_id', 'fiscal_quarter']).sort_index()
    assert ((ax['base_quota'] - bx['base_quota']).abs() < 0.05).all()
    print("  identical frames")


# ----------------------------------------------------------------------
# 3. equal + freeze + validation
# ----------------------------------------------------------------------
def test_modes_and_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: equal split, frozen bystander, validation errors")
    print(SEPARATOR)
    q = _quotas()
    out, rep = reallocate(q, 'r1', fraction=0.5, weights='equal',
                          recipients=['r2', 'r3'],
                          freeze_nodes=['r4'])
    ix = out.set_index(['node_id', 'fiscal_quarter'])
    q0 = q.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r4', 1), 'base_quota']
               - q0.loc[('r4', 1), 'base_quota']) < 0.05   # untouched
    # frozen siblings are never listed (consistent with redistribute)
    assert 'r4' not in set(rep['node'])
    assert rep['exact'].all()
    for kwargs, needle in [
        (dict(sources='r1', fraction=0.0), 'fraction'),
        (dict(sources='r1', fraction=1.5), 'fraction'),
        (dict(sources=['r1', 'r1']), 'duplicate'),
        (dict(sources='T1'), 'root'),      # T1's parent is EMEA... sources
        (dict(sources='r1', recipients=['T1']), 'eligible'),
        (dict(sources='r1', weights={'r2': -1}), 'non-negative'),
    ]:
        try:
            reallocate(q, **kwargs)
            raise AssertionError(f'expected ValueError for {kwargs}')
        except ValueError as e:
            assert needle in str(e) or needle == 'root', (kwargs, str(e))
    print("  equal/freeze fine; validations raise")


# ----------------------------------------------------------------------
# 4. resplit_by_metric == hand-built per-cascade scoped pins
# ----------------------------------------------------------------------
def test_resplit_equivalence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: resplit T1 by dc == per-cascade scoped pin "
          "composition; reconcile clean")
    print(SEPARATOR)
    q = _quotas()
    out, rep = resplit_by_metric(q, 'T1', 'dc')
    ix0 = q.set_index(['node_id', 'fiscal_quarter'])
    dc = [9, 1, 4, 2]
    pins = []
    for fq in (1, 2):
        t1 = float(ix0.loc[('T1', fq), 'base_quota'])
        for i in range(4):
            pins.append(Pin(f'r{i+1}', t1 * dc[i] / 16,
                            scope={'fiscal_quarter': fq}))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        manual, _ = apply_pins(q, pins)
    a = out.set_index(['node_id', 'fiscal_quarter']).sort_index()
    m = manual.set_index(['node_id', 'fiscal_quarter']).sort_index()
    assert ((a['base_quota'] - m['base_quota']).abs() < 0.05).all()
    assert abs(a.loc[('r1', 1), 'base_quota']
               - 1_000_000.0 * 9 / 16) < 0.5
    assert reconcile(out, hedge=1.1)['ok'].all()
    r = rep[(rep.fiscal_quarter == 1)].set_index('node_id')
    assert abs(r.loc['r1', 'metric_share'] - 9 / 16) < 1e-6
    assert rep['exact'].all()
    print("  identical to 8 scoped pins; r1 = 9/16 of T1; reconcile ok")


# ----------------------------------------------------------------------
# 5. resplit: frozen child held; missing metric hints metadata_cols
# ----------------------------------------------------------------------
def test_resplit_freeze_and_errors():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: frozen child holds, free re-split around it; "
          "missing metric errors helpfully")
    print(SEPARATOR)
    q = _quotas()
    out, rep = resplit_by_metric(q, 'T1', 'dc', freeze_nodes=['r2'])
    ix = out.set_index(['node_id', 'fiscal_quarter'])
    q0 = q.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r2', 1), 'base_quota']
               - q0.loc[('r2', 1), 'base_quota']) < 0.05
    # free pool: budget - frozen; shares 9:4:2 among r1/r3/r4
    budget = 1_000_000.0 - float(q0.loc[('r2', 1), 'base_quota'])
    assert abs(ix.loc[('r1', 1), 'base_quota']
               - budget * 9 / 15) < 0.5
    assert abs(ix.loc[('T1', 1), 'base_quota'] - 1_000_000.0) < 0.5
    try:
        resplit_by_metric(q, 'T1', 'nope')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'metadata_cols' in str(e)
    try:
        resplit_by_metric(q, 'r1', 'dc')     # leaf: nothing to re-split
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'no children' in str(e) or 're-split' in str(e)
    print("  frozen held; 9:4:2 over remaining budget; errors clear")


if __name__ == '__main__':
    test_fraction_move_equivalence()
    test_redistribute_identity()
    test_modes_and_validation()
    test_resplit_equivalence()
    test_resplit_freeze_and_errors()

    print(f"\n\n{SEPARATOR}")
    print("ALL REALLOCATE/RESPLIT TESTS PASSED")
    print(SEPARATOR)
