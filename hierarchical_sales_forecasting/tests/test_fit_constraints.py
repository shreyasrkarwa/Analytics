"""
Tests for issue #69 — fit_constraints(): overlapping totals on one
node, solved by IPF and applied through the pin engine.

Covers:
  - the filer's EXACT EAST2_1 numbers: three sales-type row totals +
    one DC product-group total, jointly satisfied (7 sweeps), both
    cuts verified from the edited frame, reconcile clean
  - DC <-> Cloud moved WITHIN each sales type (row totals hold while
    the group total binds)
  - single-cut anchor: fit_constraints with one constraint ==
    a plain scoped pin, bit-for-bit
  - the v0.41 receipt: applying the two pin sets directly RAISES
    (no longer silent) — fit_constraints is the way to satisfy both
  - loud infeasibility: contradictory cuts raise listing achieved vs
    requested per constraint; zero-support constraint raises
  - support preservation: $0 cells stay $0
  - guards: bad specs, unknown node, unmatched scope
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, fit_constraints,
    reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
DC = ['JiraDC', 'ConfDC']
CONS = [{'scope': {'st': 'Exp'}, 'total': 2_451_246.0},
        {'scope': {'st': 'Mig'}, 'total': 200_000.0},
        {'scope': {'st': 'New'}, 'total': 1_818_789.0},
        {'scope': {'product': DC}, 'total': 163_779.42}]


def _quotas(tgt=1_500_000.0):
    hdf = pd.DataFrame([dict(region='A', team='T', rep=f'r{i+1}',
                             kw=100) for i in range(2)])
    targets = pd.DataFrame(
        [dict(region='A', st=st, product=p, tgt=tgt)
         for st in ('Exp', 'Mig', 'New')
         for p in ('JiraDC', 'JiraCloud', 'ConfDC', 'ConfCloud')])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
    return q


# ----------------------------------------------------------------------
# 1. The EAST2_1 ask, end to end
# ----------------------------------------------------------------------
def test_east21_end_to_end():
    print(SEPARATOR)
    print("TEST 1: 3 sales-type totals + 1 DC-group total, jointly "
          "exact; reconcile clean")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, cells = fit_constraints(q, 'r1', CONS)
    r1 = e[e.node_id == 'r1']
    assert abs(r1[r1.st == 'Exp']['base_quota'].sum()
               - 2_451_246.0) < 0.05
    assert abs(r1[r1.st == 'Mig']['base_quota'].sum()
               - 200_000.0) < 0.05
    assert abs(r1[r1['product'].isin(DC)]['base_quota'].sum()
               - 163_779.42) < 0.05
    assert cells['exact'].all() and len(cells) == 12
    fr = e.attrs['fit_report']
    assert all(c['exact'] for c in fr['constraints'])
    assert fr['n_sweeps'] <= 20
    assert reconcile(e, hedge=1.1)['ok'].all()
    # cross-move happened WITHIN sales types: Exp row total exact while
    # its DC cells shrank and Cloud cells grew
    exp = r1[r1.st == 'Exp'].set_index('product')['base_quota']
    assert exp['JiraDC'] < 100_000 < exp['JiraCloud']
    print(f"  {fr['n_sweeps']} sweeps; both cuts exact; DC<->Cloud "
          f"moved within rows; reconcile ok")


# ----------------------------------------------------------------------
# 2. Single-cut anchor + v0.41 receipt
# ----------------------------------------------------------------------
def test_anchor_and_receipt():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: one constraint == one scoped pin; raw two-cut pins "
          "still raise (#63 detection)")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a, _ = fit_constraints(q, 'r1',
                               [{'scope': {'st': 'Exp'},
                                 'total': 900_000.0}])
        b, _ = apply_pins(q, [Pin('r1', 900_000.0,
                                  scope={'st': 'Exp'})])
    ax = a.sort_index()['base_quota']
    bx = b.sort_index()['base_quota']
    assert ((ax - bx).abs() < 0.05).all()
    # the receipt: expressing both cuts as raw pins raises
    try:
        apply_pins(q, [Pin('r1', 2_451_246.0, scope={'st': 'Exp'}),
                       Pin('r1', 163_779.42,
                           scope={'product': 'JiraDC'})])
        raise AssertionError('expected ValueError')
    except ValueError as e2:
        assert 'conflict' in str(e2)
    print("  anchor identical; raw overlap raises as designed")


# ----------------------------------------------------------------------
# 3. Infeasibility + zero support, loud
# ----------------------------------------------------------------------
def test_infeasible_loud():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: contradictory cuts raise with per-constraint "
          "residuals; zero-support raises")
    print(SEPARATOR)
    q = _quotas()
    try:
        fit_constraints(q, 'r1', [
            {'scope': {'st': 'Exp'}, 'total': 100.0},
            {'scope': {'st': 'Mig'}, 'total': 50.0},
            {'scope': {'st': 'New'}, 'total': 50.0},
            {'scope': {'product': DC}, 'total': 9_999_999.0}])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        assert 'NOT jointly satisfiable' in msg
        assert 'residual' in msg and '9,999,999' in msg
    # zero support: zero out r1's Mig cells, then demand Mig money
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        z, _ = apply_pins(q, [Pin('r1', 0.0, scope={'st': 'Mig'})])
    try:
        fit_constraints(z, 'r1', [{'scope': {'st': 'Mig'},
                                   'total': 100_000.0}])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'nothing to scale' in str(e)
    # support preservation: fit around the zeroed cells keeps them $0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e2, _ = fit_constraints(z, 'r1',
                                [{'scope': {'st': 'Exp'},
                                  'total': 2_000_000.0},
                                 {'scope': {'product': DC},
                                  'total': 500_000.0}])
    mig = e2[(e2.node_id == 'r1') & (e2.st == 'Mig')]['base_quota']
    assert (mig.abs() < 0.05).all()
    print("  residual table raised; zero cells honest + preserved")


# ----------------------------------------------------------------------
# 4. Guards
# ----------------------------------------------------------------------
def test_guards():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: spec/node/scope guards")
    print(SEPARATOR)
    q = _quotas()
    for cons, needle in [
        ([], 'non-empty'),
        ([{'scope': 'nope', 'total': 1.0}], 'non-empty'),
        ([{'scope': {'st': 'Exp'}}], 'non-empty'),
        ([{'scope': {'nope': 1}, 'total': 1.0}], 'nope'),
        ([{'scope': {'st': 'NOPE'}, 'total': 1.0}], 'matches no'),
    ]:
        try:
            fit_constraints(q, 'r1', cons)
            raise AssertionError(f'expected ValueError: {needle}')
        except ValueError as e:
            assert needle in str(e), (needle, str(e)[:100])
    try:
        fit_constraints(q, 'NOPE', [{'scope': {'st': 'Exp'},
                                     'total': 1.0}])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'NOPE' in str(e)
    print("  all guards raise, named")


if __name__ == '__main__':
    test_east21_end_to_end()
    test_anchor_and_receipt()
    test_infeasible_loud()
    test_guards()

    print(f"\n\n{SEPARATOR}")
    print("ALL FIT-CONSTRAINTS TESTS PASSED")
    print(SEPARATOR)
