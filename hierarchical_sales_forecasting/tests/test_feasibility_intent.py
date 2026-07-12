"""
Tests for issue #45 — intentional vs genuine infeasibility in the
apply_pins report.

Covers:
  - full partition (all siblings pinned, deltas cancel):
    unabsorbed_reason='all_blocked', intentional=True, NO warning,
    parent conserved to the cent
  - root pin: 'no_siblings', intentional, silent
  - exclude-all: 'all_blocked', intentional, silent
  - genuine floors: 'floors_at_zero', intentional=False, warning fires
  - frozen-mass subtree shortfall stays genuine (warns; not
    intentional)
  - columns present on fully-absorbed and skipped rows
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec, Pin, apply_pins

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    hdf = pd.DataFrame([dict(region='EMEA', team=f'T{i//2+1}',
                             rep=f'r{i+1}', kw=[100, 200, 300, 400][i])
                        for i in range(4)])
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW)
    return q


def _run(pins, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, rep = apply_pins(_quotas(), pins, **kw)
    absorb_warns = [x for x in w
                    if 'could not be absorbed' in str(x.message)]
    return e, rep, absorb_warns, w


# ----------------------------------------------------------------------
# 1. Full partition: intentional, silent, conserved
# ----------------------------------------------------------------------
def test_full_partition_silent():
    print(SEPARATOR)
    print("TEST 1: T1+T2 pinned to the exact envelope -> intentional, "
          "no warnings, conserved")
    print(SEPARATOR)
    e, rep, aw, _ = _run([Pin('T1', 400_000.0), Pin('T2', 600_000.0)])
    r = rep.set_index('pin_node')
    print(rep[['pin_node', 'unabsorbed', 'unabsorbed_reason',
               'intentional', 'feasible']].to_string(index=False))
    assert not aw, "absorption warnings leaked for an intentional case"
    assert (r['unabsorbed_reason'] == 'all_blocked').all()
    assert r['intentional'].all()
    assert not r['feasible'].any()          # semantics unchanged
    ix = e.set_index('node_id')
    assert abs(ix.loc['EMEA', 'base_quota'] - 1_000_000.0) < 0.05
    assert abs(ix.loc[['T1', 'T2'], 'base_quota'].sum()
               - 1_000_000.0) < 0.05        # deltas cancelled exactly


# ----------------------------------------------------------------------
# 2. Root pin: no siblings exist -> intentional, silent
# ----------------------------------------------------------------------
def test_root_pin_silent():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: root pin -> 'no_siblings', intentional, silent")
    print(SEPARATOR)
    _, rep, aw, _ = _run([Pin('EMEA', 1_200_000.0)])
    r = rep.iloc[0]
    assert not aw
    assert r['unabsorbed_reason'] == 'no_siblings'
    assert bool(r['intentional']) and not bool(r['feasible'])
    print(f"  reason={r['unabsorbed_reason']} intentional=True, "
          f"0 warnings")


# ----------------------------------------------------------------------
# 3. exclude-all: removing money on purpose -> intentional, silent
# ----------------------------------------------------------------------
def test_exclude_all_silent():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: exclude=all siblings -> 'all_blocked', intentional")
    print(SEPARATOR)
    _, rep, aw, _ = _run([Pin('T1', 100_000.0, exclude=['T2'])])
    r = rep.iloc[0]
    assert not aw
    assert r['unabsorbed_reason'] == 'all_blocked'
    assert bool(r['intentional'])
    print("  silent, recorded as intentional")


# ----------------------------------------------------------------------
# 4. Genuine floors: still warns, intentional=False
# ----------------------------------------------------------------------
def test_genuine_floors_warn():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: pin exceeds free capacity -> 'floors_at_zero', "
          "warning fires")
    print(SEPARATOR)
    _, rep, aw, _ = _run([Pin('T1', 2_000_000.0)])
    r = rep.iloc[0]
    assert len(aw) == 1
    assert 'floored at $0' in str(aw[0].message)
    assert r['unabsorbed_reason'] == 'floors_at_zero'
    assert not bool(r['intentional']) and not bool(r['feasible'])
    print("  1 warning; intentional=False")


# ----------------------------------------------------------------------
# 5. Frozen-mass shortfall stays genuine
# ----------------------------------------------------------------------
def test_shortfall_stays_genuine():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: protected mass > pin -> subtree_shortfall warns, "
          "not intentional")
    print(SEPARATOR)
    # frozen r1 carries 100K baseline; pinning T1 BELOW that forces a
    # subtree shortfall (protected mass exceeds the pin)
    _, rep, _, w = _run([Pin('T1', 50_000.0)], freeze_nodes=['r1'])
    r = rep.iloc[0]
    assert any('subtree_shortfall' in str(x.message) for x in w)
    assert r['subtree_shortfall'] > 0
    assert not bool(r['intentional'])       # shortfall disqualifies
    print(f"  shortfall={r['subtree_shortfall']:,.2f}, "
          f"intentional=False, warned")


# ----------------------------------------------------------------------
# 6. Columns on clean and skipped rows
# ----------------------------------------------------------------------
def test_columns_everywhere():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: fully-absorbed and skipped rows carry the columns")
    print(SEPARATOR)
    _, rep, aw, _ = _run([Pin('T1', 500_000.0), Pin('GHOST', 1.0)],
                         on_missing='skip')
    r = rep.set_index('pin_node')
    assert not aw
    assert pd.isna(r.loc['T1', 'unabsorbed_reason'])
    assert not bool(r.loc['T1', 'intentional'])
    assert bool(r.loc['T1', 'feasible'])
    assert pd.isna(r.loc['GHOST', 'unabsorbed_reason'])
    assert not bool(r.loc['GHOST', 'intentional'])
    print("  clean pin: reason NaN/None, feasible; skipped pin: "
          "columns present")


if __name__ == '__main__':
    test_full_partition_silent()
    test_root_pin_silent()
    test_exclude_all_silent()
    test_genuine_floors_warn()
    test_shortfall_stays_genuine()
    test_columns_everywhere()

    print(f"\n\n{SEPARATOR}")
    print("ALL FEASIBILITY-INTENT TESTS PASSED")
    print(SEPARATOR)
