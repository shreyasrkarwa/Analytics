"""
Tests for issue #60 — reconcile() as the penny-level conservation audit.

Covers:
  - the RECEIPTS on the disproven claim: tolerance=0.05 is five CENTS
    (absolute dollars), not 5% — a 1% team gap is flagged loudly, a
    $0.06 gap is flagged, only <= 5 cents passes
  - target_total: the one identity internal checks can't see — a root
    floated by enforce_identities(anchor='leaves') keeps every
    internal check clean while depth-0 drifts off the input plan;
    targets= catches it per combo + globally
  - dropped-target drift: a target combo with NO quota rows appears as
    expected=amount / actual=0
  - depth_conservation: the aggregate ledger view, exact in jagged
    trees (leaf at depth 1 never false-alarms)
  - on_fail='raise' names (check, node, combo); 'warn' unchanged
  - scalar target_total= global row; targets=/target_total= exclusive
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, enforce_identities,
    reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas(targets=None):
    hdf = pd.DataFrame([dict(region='EMEA', team='T', rep=f'r{i+1}',
                             kw=[100, 300][i]) for i in range(2)])
    if targets is None:
        targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                     tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW,
                        hedge_multiplier=1.1)
    return q, targets


# ----------------------------------------------------------------------
# 1. The receipts: tolerance is 5 CENTS, not 5%
# ----------------------------------------------------------------------
def test_tolerance_is_absolute_dollars():
    print(SEPARATOR)
    print("TEST 1: default tolerance = five cents absolute; 1% gaps "
          "and 6-cent gaps flagged, 4-cent gaps pass")
    print(SEPARATOR)
    q, _ = _quotas()
    for gap, flagged in [(10_000.0, True),    # the filer's "1% gap"
                         (0.06, True), (0.04, False)]:
        q2 = q.copy()
        m = (q2.node_id == 'r1') & (q2.fiscal_quarter == 1)
        q2.loc[m, 'base_quota'] -= gap
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            f = reconcile(q2)
        assert (~f.ok).any() == flagged, gap
        assert any('failed' in str(x.message) for x in w) == flagged
    print("  $10,000 and $0.06 flagged + warned; $0.04 passes")


# ----------------------------------------------------------------------
# 2. target_total: internal checks clean, plan total drifted
# ----------------------------------------------------------------------
def test_target_total_catches_floated_root():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: anchor='leaves' floats the root — every internal "
          "check clean, targets= catches the drift")
    print(SEPARATOR)
    q, targets = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(q, [Pin('r1', 900_000.0),
                              Pin('r2', 1_500_000.0)],
                          on_overshoot='allow')      # 2.4M vs 2M plan
        f, _ = enforce_identities(e, anchor='leaves')
    # internal identities all clean
    internal = reconcile(f, hedge=1.1)
    assert internal['ok'].all()
    # ... but the plan total drifted +400K, and targets= sees it
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        audit = reconcile(f, hedge=1.1, targets=targets,
                          target_col='tgt')
    tt = audit[audit.check == 'target_total']
    assert len(tt) == 3                      # 2 combos + 1 global
    g = tt[tt[['region', 'fiscal_quarter']].isna().all(axis=1)]
    assert abs(float(g['delta'].iloc[0]) - 400_000.0) < 0.5
    assert not tt['ok'].any()                # both combos drifted too
    assert any('failed' in str(x.message) for x in w)
    print(f"  internal ok; target_total delta "
          f"+{float(g['delta'].iloc[0]):,.0f} flagged")


# ----------------------------------------------------------------------
# 3. Dropped-target drift: combo with no quota rows
# ----------------------------------------------------------------------
def test_dropped_target_visible():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: a target combo with NO quota rows shows "
          "expected=amount / actual=0")
    print(SEPARATOR)
    q, _ = _quotas()
    targets3 = pd.DataFrame([
        dict(region='EMEA', fiscal_quarter=1, tgt=1_000_000.0),
        dict(region='EMEA', fiscal_quarter=2, tgt=1_000_000.0),
        dict(region='APAC', fiscal_quarter=1, tgt=250_000.0)])  # dropped
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        audit = reconcile(q, targets=targets3, target_col='tgt')
    tt = audit[audit.check == 'target_total']
    apac = tt[tt.region == 'APAC']
    assert len(apac) == 1
    assert abs(float(apac['expected'].iloc[0]) - 250_000.0) < 0.05
    assert float(apac['actual'].iloc[0]) == 0.0
    assert not bool(apac['ok'].iloc[0])
    # EMEA combos + global-minus-apac: EMEA rows ok, global not
    assert tt[tt.region == 'EMEA']['ok'].all()
    print("  APAC: expected 250,000 / actual 0 / ok=False")


# ----------------------------------------------------------------------
# 4. depth_conservation: ledger view, jagged-tree exact
# ----------------------------------------------------------------------
def test_depth_conservation_jagged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: depth aggregate rows; a leaf at depth 1 never "
          "false-alarms")
    print(SEPARATOR)
    q, _ = _quotas()
    f = reconcile(q)
    dc = f[f.check == 'depth_conservation']
    assert len(dc) == 4                     # depths 1,2 x 2 quarters
    assert dc['ok'].all() and dc['node_id'].isna().all()
    # jagged, hand-built: A -> {B -> D/E, C} — C is a LEAF at depth 1,
    # so a naive "d1 total == d2 total" audit would scream (100 vs 60)
    qj = pd.DataFrame([
        dict(node_id='A', parent=None, depth=0, base_quota=100.0),
        dict(node_id='B', parent='A', depth=1, base_quota=60.0),
        dict(node_id='C', parent='A', depth=1, base_quota=40.0),
        dict(node_id='D', parent='B', depth=2, base_quota=30.0),
        dict(node_id='E', parent='B', depth=2, base_quota=30.0)])
    fj = reconcile(qj)
    assert fj['ok'].all()                   # naive d1==d2 would fail:
    dcj = fj[fj.check == 'depth_conservation'].set_index('depth')
    # depth 2 compares against B only (C has no children)
    assert abs(dcj.loc[2, 'expected'] - 60.0) < 0.05
    assert abs(dcj.loc[1, 'expected'] - 100.0) < 0.05
    print("  uniform + jagged both clean; depth-2 expected excludes "
          "the depth-1 leaf")


# ----------------------------------------------------------------------
# 5. on_fail='raise' + scalar target_total + validation
# ----------------------------------------------------------------------
def test_on_fail_and_scalar():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: on_fail='raise' names offenders; scalar "
          "target_total; targets XOR target_total")
    print(SEPARATOR)
    q, targets = _quotas()
    # clean + scalar total -> one extra global row, ok
    f = reconcile(q, target_total=2_000_000.0)
    tt = f[f.check == 'target_total']
    assert len(tt) == 1 and tt['ok'].all()
    # broken frame raises with names
    q2 = q.copy()
    m = (q2.node_id == 'r1') & (q2.fiscal_quarter == 1)
    q2.loc[m, 'base_quota'] -= 10_000.0
    try:
        reconcile(q2, on_fail='raise')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        assert 'conservation' in msg and 'T' in msg
        assert 'fiscal_quarter=1' in msg and '-10,000' in msg
    for kwargs in (dict(targets=targets, target_col='tgt',
                        target_total=1.0),
                   dict(targets=targets),          # no target_col
                   dict(on_fail='explode')):
        try:
            reconcile(q, **kwargs)
            raise AssertionError(f'expected ValueError for {kwargs}')
        except ValueError:
            pass
    print("  raise names check/node/combo/delta; guards raise")


if __name__ == '__main__':
    test_tolerance_is_absolute_dollars()
    test_target_total_catches_floated_root()
    test_dropped_target_visible()
    test_depth_conservation_jagged()
    test_on_fail_and_scalar()

    print(f"\n\n{SEPARATOR}")
    print("ALL CONSERVATION-AUDIT TESTS PASSED")
    print(SEPARATOR)
