"""
Tests for issues #51 / #52 / #53 — callables receive the FULL cascade
identity, verbatim.

Covers:
  - #51 cascade_many: metrics/gate callables see group keys + ALL
    sub-target columns; the Migration-splits-by-dc scenario that was
    silently money-wrong now routes correctly (verified numerically)
  - #53: the documented mapping pattern keyed on a sub-target column
    just works; equivalence anchor vs manual split-and-concat
  - #51 cascade_levels: transition callables see parent key + root
    taxonomy key + sub-target columns
  - #52: values arrive VERBATIM — mixed case preserved end to end
  - weights_long: per-row records tagged with sub-target columns when
    slates vary; combo_report weights_source='mixed'
  - atomic skip: a callable that errors on a later sub-target row
    still drops the whole combination (no partial frames)
  - backward compat: group-key-only callables behave exactly as before
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, cascade_levels, MetricSpec,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
DC = [MetricSpec('dc', direction='proportional', weight=1.0,
                 columns=['dc'])]


def _hdf():
    return pd.DataFrame([
        dict(region='EMEA', team=f'T{i//2+1}', rep=f'r{i+1}',
             kw=[100, 200, 300, 400][i], dc=[9, 7, 5, 3][i])
        for i in range(4)])


def _targets():
    return pd.DataFrame([
        dict(region='EMEA', st1_sales_type='Migration', tgt=1_000_000.0),
        dict(region='EMEA', st1_sales_type='Expansion', tgt=800_000.0),
    ])


# ----------------------------------------------------------------------
# 1. #51/#53 — the exact silent-wrong-money scenario, now correct
# ----------------------------------------------------------------------
def test_sub_target_routing():
    print(SEPARATOR)
    print("TEST 1: Migration -> dc split, Expansion -> kw split, ONE "
          "call (the #51 incident, fixed)")
    print(SEPARATOR)
    seen = []
    policy = lambda g: (seen.append(dict(g)) or
                        (DC if g.get('st1_sales_type') == 'Migration'
                         else KW))
    q, w = cascade_many(_hdf(), _targets(), group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=policy)
    assert all('st1_sales_type' in g for g in seen)      # full identity
    mig = q[q.st1_sales_type == 'Migration'].set_index('node_id')
    exp = q[q.st1_sales_type == 'Expansion'].set_index('node_id')
    assert abs(mig.loc['r1', 'share_of_parent'] - 9 / 16) < 1e-4  # dc!
    assert abs(exp.loc['r1', 'share_of_parent'] - 1 / 3) < 1e-4   # kw
    # #53 equivalence anchor: == manual split-and-concat
    t = _targets()
    qa, _ = cascade_many(_hdf(), t[t.st1_sales_type == 'Migration'],
                         group_keys=['region'], target_col='tgt',
                         taxonomy=['region', 'team', 'rep'], metrics=DC)
    qb, _ = cascade_many(_hdf(), t[t.st1_sales_type == 'Expansion'],
                         group_keys=['region'], target_col='tgt',
                         taxonomy=['region', 'team', 'rep'], metrics=KW)
    manual = pd.concat([qa, qb], ignore_index=True).set_index(
        ['node_id', 'st1_sales_type']).sort_index()
    auto = q.set_index(['node_id', 'st1_sales_type']).sort_index()
    assert ((auto['base_quota'] - manual['base_quota']).abs()
            < 0.05).all()
    print("  routed per sub-target; identical to split-and-concat")


# ----------------------------------------------------------------------
# 2. #51 cascade_levels: transitions see the full identity
# ----------------------------------------------------------------------
def test_cascade_levels_identity():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: transition callables see parent + root key + "
          "sub-targets")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='EMEA', rvp=f'V{i//2+1}',
                             director=f'D{i+1}', kw=100 * (i + 1),
                             dc=[9, 7, 5, 3][i]) for i in range(4)])
    rt = pd.DataFrame([dict(region='EMEA', st1_sales_type='Migration',
                            tgt=1_000_000.0)])
    seen = []
    pol = lambda g: (seen.append(dict(g)) or KW)
    cascade_levels(hdf, rt, taxonomy=['region', 'rvp', 'director'],
                   target_col='tgt',
                   level_kwargs=[dict(metrics=pol), dict(metrics=pol)])
    t2 = [g for g in seen if 'rvp' in g]                # transition 2
    print(f"  transition-2 identities: {t2}")
    assert t2 and all(g.get('st1_sales_type') == 'Migration'
                      for g in t2)
    assert all(g.get('region') == 'EMEA' for g in t2)   # root key too
    assert all('rvp' in g for g in t2)


# ----------------------------------------------------------------------
# 3. #52 — values verbatim, mixed case preserved
# ----------------------------------------------------------------------
def test_values_verbatim():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: #52 disproven and pinned — no case normalization, "
          "ever")
    print(SEPARATOR)
    hdf = _hdf().assign(st1_sales_type='MiGrAtIoN')
    t = pd.DataFrame([dict(st1_sales_type='MiGrAtIoN', region='EMEA',
                           fiscal_quarter='q1_FY26', tgt=500_000.0)])
    seen = []
    pol = lambda g: (seen.append(dict(g)) or KW)
    cascade_many(hdf, t, group_keys=['st1_sales_type', 'region'],
                 target_col='tgt', taxonomy=['region', 'team', 'rep'],
                 metrics=pol)
    assert seen[0]['st1_sales_type'] == 'MiGrAtIoN'     # group key
    assert seen[0]['fiscal_quarter'] == 'q1_FY26'       # sub-target
    print(f"  received: {seen[0]}")


# ----------------------------------------------------------------------
# 4. weights_long per-row tagging + weights_source='mixed'
# ----------------------------------------------------------------------
def test_records_and_mixed_source():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: weights_long tagged per sub-target; combo_report "
          "says 'mixed' when policy partially decides")
    print(SEPARATOR)
    policy = lambda g: (DC if g.get('st1_sales_type') == 'Migration'
                        else None)                       # None -> legacy
    hdf = _hdf().assign(Q1_Attainment=1.0)
    q, w = cascade_many(hdf, _targets(), group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=policy)
    assert 'st1_sales_type' in w.columns                 # per-row tags
    wm = w[w.st1_sales_type == 'Migration']
    we = w[w.st1_sales_type == 'Expansion']
    assert (wm['weights_source'] == 'policy').all()
    assert wm.iloc[0]['metric'] == 'dc'
    assert (we['weights_source'] == 'default_attainment').all()
    assert we.iloc[0]['metric'] == '_Attainment'
    rep = pd.DataFrame(q.attrs['combo_report'])
    assert rep.iloc[0]['weights_source'] == 'mixed'
    print("  per-row provenance recorded; combo weights_source='mixed'")


# ----------------------------------------------------------------------
# 5. Atomic skip on late-row callable errors
# ----------------------------------------------------------------------
def test_atomic_skip():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: callable erroring on the SECOND sub-target row "
          "still drops the whole combo")
    print(SEPARATOR)
    def bomb(g):
        if g.get('st1_sales_type') == 'Expansion':
            raise RuntimeError('boom')
        return DC
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(_hdf(), _targets(), group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=bomb)
    assert q.empty                                      # no partial rows
    rep = pd.DataFrame(q.attrs['combo_report'])
    assert bool(rep.iloc[0]['skipped']) and 'boom' in rep.iloc[0]['reason']
    print("  combo skipped atomically, reason recorded")


# ----------------------------------------------------------------------
# 6. Backward compat: group-key-only callables unchanged
# ----------------------------------------------------------------------
def test_backward_compat():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: a callable keyed on group keys only behaves exactly "
          "as before")
    print(SEPARATOR)
    pol = lambda g: KW if g['region'] == 'EMEA' else DC
    q1, w1 = cascade_many(_hdf(), _targets(), group_keys=['region'],
                          target_col='tgt',
                          taxonomy=['region', 'team', 'rep'],
                          metrics=pol)
    q2, _ = cascade_many(_hdf(), _targets(), group_keys=['region'],
                         target_col='tgt',
                         taxonomy=['region', 'team', 'rep'], metrics=KW)
    a = q1.set_index(['node_id', 'st1_sales_type']).sort_index()
    b = q2.set_index(['node_id', 'st1_sales_type']).sort_index()
    assert ((a['base_quota'] - b['base_quota']).abs() < 0.005).all()
    assert (w1['weights_source'] == 'policy').all()
    print("  identical cascades; extra keys in g are purely additive")


if __name__ == '__main__':
    test_sub_target_routing()
    test_cascade_levels_identity()
    test_values_verbatim()
    test_records_and_mixed_source()
    test_atomic_skip()
    test_backward_compat()

    print(f"\n\n{SEPARATOR}")
    print("ALL CALLABLE-IDENTITY TESTS PASSED")
    print(SEPARATOR)
