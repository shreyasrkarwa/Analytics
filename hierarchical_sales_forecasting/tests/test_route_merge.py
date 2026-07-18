"""
Tests for issue #61 — route_targets(merge=True).

Covers:
  - the exact #61 footgun: rollup=True + concat leaves node-grain
    duplicates (two EMEA rows per quarter); merge=True folds them away
  - equivalence with the filer's hand-rolled groupby collapse
  - conservation + reconcile clean on the merged frame (by
    construction: rollup adds the same delta up the chain, cascaded
    derives from each row's own ratio)
  - provenance: routed flag OR'd, routed_base_quota exact,
    routed_from records the adopted-away orphan identity,
    attrs['route_report'] complete
  - derived columns recomputed (hedge_buffer / overassignment_pct /
    share_of_parent)
  - multi-target accumulation into the same rows
  - unmatched routed rows APPENDED with a warning
  - merge=True + rollup=False raises; merge=False bit-identical to
    the old additive-overlay behavior
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, route_targets, reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
RK = {'segment': 'Commercial', 'fiscal_quarter': 1}


def _quotas():
    """Commercial tree: EMEA -> NORTH -> r1/r2 (kw 1:3), 1M per
    quarter, hedge 1.1. Cascade keys: region/segment/fiscal_quarter."""
    hdf = pd.DataFrame([dict(region='EMEA', team='NORTH', rep=f'r{i+1}',
                             kw=[100, 300][i]) for i in range(2)])
    targets = pd.DataFrame([dict(region='EMEA', segment='Commercial',
                                 fiscal_quarter=fq, tgt=1_000_000.0)
                            for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region', 'segment'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW,
                        hedge_multiplier=1.1)
    return q


DROPPED = pd.DataFrame([dict(region='EMEA', segment='Government',
                             fiscal_quarter=1, tgt=500_000.0)])


# ----------------------------------------------------------------------
# 1. The footgun, then the fold
# ----------------------------------------------------------------------
def test_footgun_and_fold():
    print(SEPARATOR)
    print("TEST 1: concat doubles every node at node grain; merge=True "
          "folds — zero duplicates, conserved, reconcile clean")
    print(SEPARATOR)
    q = _quotas()
    r = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal')
    combined = pd.concat([q, r], ignore_index=True)
    # the #61 complaint: node-grain consumers see EMEA twice in Q1
    dup = combined.groupby(['node_id', 'fiscal_quarter']).size()
    assert dup.max() == 2 and dup[('EMEA', 1)] == 2

    m = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal',
                      merge=True)
    assert m.groupby(['node_id', 'fiscal_quarter']).size().max() == 1
    # conservation: depth-0 total grew by exactly the routed 500K
    grew = (m[m.depth == 0]['base_quota'].sum()
            - q[q.depth == 0]['base_quota'].sum())
    assert abs(grew - 500_000.0) < 0.05
    assert reconcile(m, hedge=1.1)['ok'].all()
    # Q2 untouched, bit-identical
    ix, ix0 = (f.set_index(['node_id', 'fiscal_quarter']) for f in (m, q))
    for n in ('EMEA', 'NORTH', 'r1', 'r2'):
        assert ix.loc[(n, 2), 'base_quota'] == ix0.loc[(n, 2),
                                                       'base_quota']
    print("  concat: EMEA x2; merged: x1, +500K, reconcile ok")


# ----------------------------------------------------------------------
# 2. Equivalence with the filer's hand collapse
# ----------------------------------------------------------------------
def test_hand_collapse_equivalence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: merge=True == retag + concat + groupby-sum collapse")
    print(SEPARATOR)
    q = _quotas()
    r = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal')
    hand = r.copy()
    hand['segment'] = 'Commercial'          # the retag merge implies
    hand = pd.concat([q.assign(routed=False), hand], ignore_index=True)
    hand = (hand.groupby(['node_id', 'segment', 'fiscal_quarter'],
                         as_index=False)
            .agg(base_quota=('base_quota', 'sum'),
                 cascaded_quota=('cascaded_quota', 'sum'),
                 routed=('routed', 'max')))
    m = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal',
                      merge=True)
    a = m.set_index(['node_id', 'fiscal_quarter']).sort_index()
    h = hand.set_index(['node_id', 'fiscal_quarter']).sort_index()
    for col in ('base_quota', 'cascaded_quota'):
        assert ((a[col] - h[col]).abs() < 0.05).all()
    assert (a['routed'] == h['routed']).all()
    print("  identical dollars + routed flags, without the 20 lines")


# ----------------------------------------------------------------------
# 3. Provenance + derived columns
# ----------------------------------------------------------------------
def test_provenance_and_derived():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: routed_base_quota / routed_from / route_report; "
          "hedge_buffer + share_of_parent recomputed")
    print(SEPARATOR)
    q = _quotas()
    m = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal',
                      merge=True)
    ix = m.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r1', 1), 'routed_base_quota'] - 250_000.0) < 0.05
    assert abs(ix.loc[('EMEA', 1), 'routed_base_quota']
               - 500_000.0) < 0.05
    assert ix.loc[('r1', 1), 'routed_from'] == 'segment=Government'
    assert not ix.loc[('r1', 2), 'routed']
    assert pd.isna(ix.loc[('r1', 2), 'routed_from']) or \
        ix.loc[('r1', 2), 'routed_from'] is None
    rep = pd.DataFrame(m.attrs['route_report'])
    assert rep['matched'].all() and len(rep) == 4
    assert set(rep['node_id']) == {'EMEA', 'NORTH', 'r1', 'r2'}
    # derived: r1 Q1 was 250K base / 275K cascaded; +250K/+275K routed
    b, c = ix.loc[('r1', 1), 'base_quota'], ix.loc[('r1', 1),
                                                   'cascaded_quota']
    assert abs(ix.loc[('r1', 1), 'hedge_buffer'] - (c - b)) < 0.05
    assert abs(ix.loc[('r1', 1), 'overassignment_pct']
               - (c - b) / b) < 1e-4
    # share_of_parent recomputed: equal split moved r1 from 1/4 to 1/3
    assert abs(ix.loc[('r1', 1), 'share_of_parent'] - 1 / 3) < 1e-4
    assert abs(ix.loc[('r1', 2), 'share_of_parent'] - 0.25) < 1e-4
    print("  provenance exact; buffers + shares recomputed")


# ----------------------------------------------------------------------
# 4. Multi-target accumulation
# ----------------------------------------------------------------------
def test_multi_target_accumulation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: two orphan segments fold into the same rows; "
          "deltas accumulate, routed_from lists both")
    print(SEPARATOR)
    q = _quotas()
    two = pd.DataFrame([
        dict(region='EMEA', segment='Government', fiscal_quarter=1,
             tgt=500_000.0),
        dict(region='EMEA', segment='Education', fiscal_quarter=1,
             tgt=100_000.0)])
    m = route_targets(two, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal',
                      merge=True)
    ix = m.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r1', 1), 'routed_base_quota']
               - 300_000.0) < 0.05
    assert abs(ix.loc[('EMEA', 1), 'base_quota'] - 1_600_000.0) < 0.5
    ftags = str(ix.loc[('r1', 1), 'routed_from'])
    assert 'Government' in ftags and 'Education' in ftags
    assert reconcile(m, hedge=1.1)['ok'].all()
    print(f"  routed_from: '{ftags}'; totals + reconcile ok")


# ----------------------------------------------------------------------
# 5. Unmatched -> appended with warning; validation
# ----------------------------------------------------------------------
def test_appended_and_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: combo not in the tree -> rows appended + warned; "
          "merge without rollup raises; merge=False unchanged")
    print(SEPARATOR)
    q = _quotas()
    # region is a cascade key NOT covered by recipient_keys, so the
    # adopted identity (APAC, Commercial, 1) has no tree combo -> append
    q3 = pd.DataFrame([dict(region='APAC', segment='Government',
                            fiscal_quarter=3, tgt=200_000.0)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = route_targets(q3, q, recipients=['r1', 'r2'],
                          target_col='tgt', recipient_keys=RK,
                          split='equal', merge=True)
    assert any('APPENDED' in str(x.message) for x in w)
    rep = pd.DataFrame(m.attrs['route_report'])
    assert not rep['matched'].any()
    ap = m[m.region == 'APAC']
    assert len(ap) == 4 and ap['routed'].all()
    assert (ap['fiscal_quarter'] == 1).all()      # retagged by adoption
    assert abs(ap[ap.depth == 0]['base_quota'].sum()
               - 200_000.0) < 0.05
    assert reconcile(m, hedge=1.1)['ok'].all()   # appended chain complete
    try:
        route_targets(DROPPED, q, recipients=['r1'], target_col='tgt',
                      recipient_keys=RK, rollup=False, merge=True)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'rollup' in str(e)
    # merge=False: old behavior byte-for-byte (orphan tags kept)
    r = route_targets(DROPPED, q, recipients=['r1', 'r2'],
                      target_col='tgt', recipient_keys=RK, split='equal')
    assert (r['segment'] == 'Government').all()
    assert 'routed_from' not in r.columns
    assert len(r) == 4 and r['routed'].all()
    print("  appended combo reconciles standalone; guards + default "
          "behavior intact")


if __name__ == '__main__':
    test_footgun_and_fold()
    test_hand_collapse_equivalence()
    test_provenance_and_derived()
    test_multi_target_accumulation()
    test_appended_and_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL ROUTE-MERGE TESTS PASSED")
    print(SEPARATOR)
