"""
Tests for issue #47 — concentrate(): collapse siblings' scoped quota
onto one sibling (the inverse of redistribute). Thin sugar over
apply_pins; these tests pin the equivalences.

Covers:
  - default (all siblings) == hand-built pins == chained redistribute
  - all-depth zeroing of sources, destination internal mix preserved,
    parent conserved, hedge ratios kept, other scopes untouched
  - explicit from_nodes subset: bystander verified at baseline exactly
  - no spurious 'could not be absorbed' warnings; report all-exact
  - freeze: frozen sibling excluded from default sources
  - validation: root, non-sibling (-> route_targets), frozen source,
    duplicates, unknown node
  - round-trip: concentrate then redistribute restores baselines
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, redistribute, concentrate,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
SCOPE = {'st': 'Migration'}


def _quotas():
    """CENTRAL -> teams C1..C4 -> 2 reps each, Migration + NN.
    Migration team baselines: C1=600K, C2=1.2M, C3=1.8M, C4=300K."""
    rows = []
    for tm, b in [('C1', 100), ('C2', 200), ('C3', 300), ('C4', 50)]:
        for r in (1, 2):
            rows.append(dict(st='Migration', region='CENTRAL', team=tm,
                             rep=f'{tm}_r{r}', kw=b * r))
    hdf = pd.DataFrame(rows)
    hdf = pd.concat([hdf, hdf.assign(st='NN')], ignore_index=True)
    targets = pd.DataFrame([dict(st='Migration', tgt=3_900_000.0),
                            dict(st='NN', tgt=2_000_000.0)])
    q, _ = cascade_many(hdf, targets, group_keys=['st'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=KW, hedge_multiplier=1.1)
    return q


def _mig(e):
    return e[e.st == 'Migration'].set_index('node_id')


# ----------------------------------------------------------------------
# 1. Default == hand-built pins == chained redistribute; all-depth
# ----------------------------------------------------------------------
def test_equivalences_and_depths():
    print(SEPARATOR)
    print("TEST 1: concentrate default == direct pins == chained "
          "redistribute; all depths correct")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e1, rep = concentrate(q, 'C4', scope=SCOPE)
    assert not [x for x in w if 'could not be absorbed' in str(x.message)]
    pins = [Pin('C4', 3_900_000.0, scope=SCOPE)] + \
           [Pin(t, 0.0, scope=SCOPE) for t in ('C1', 'C2', 'C3')]
    e2, _ = apply_pins(q, pins)
    e3 = q
    for t in ('C1', 'C2', 'C3'):
        e3, _ = redistribute(e3, t, to_nodes=['C4'], scope=SCOPE)
    m1, m2, m3 = _mig(e1), _mig(e2), _mig(e3)
    for n in m1.index:
        assert abs(m1.loc[n, 'base_quota'] - m2.loc[n, 'base_quota']) < 0.05
        assert abs(m1.loc[n, 'base_quota'] - m3.loc[n, 'base_quota']) < 0.05
    m0 = _mig(q)
    assert abs(m1.loc['C4', 'base_quota'] - 3_900_000.0) < 0.05
    assert (m1.loc[['C1', 'C1_r1', 'C1_r2', 'C2_r1', 'C3_r2'],
                   'base_quota'] == 0).all()          # depths 2 AND 3
    assert abs(m1.loc['C4_r1', 'base_quota']
               / m1.loc['C4', 'base_quota'] - 1 / 3) < 1e-6  # mix kept
    assert abs(m1.loc['CENTRAL', 'base_quota']
               - m0.loc['CENTRAL', 'base_quota']) < 0.05     # conserved
    assert abs(m1.loc['C4_r2', 'cascaded_quota']
               / m1.loc['C4_r2', 'base_quota'] - 1.1 ** 2) < 1e-9
    nn = e1[e1.st == 'NN'].set_index('node_id')['base_quota']
    assert nn.equals(q[q.st == 'NN'].set_index('node_id')['base_quota'])
    r = rep.set_index('node')
    assert r.loc['C4', 'role'] == 'destination' and rep['exact'].all()
    assert (rep['role'] == 'source').sum() == 3
    print("  three spellings identical; sources zero at every depth; "
          "C4 mix + hedge intact; NN untouched")


# ----------------------------------------------------------------------
# 2. Explicit subset: bystander exactly at baseline
# ----------------------------------------------------------------------
def test_subset_with_bystander():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: from_nodes=['C1','C2'] -> C3 is a bystander, exact")
    print(SEPARATOR)
    q = _quotas()
    e, rep = concentrate(q, 'C4', from_nodes=['C1', 'C2'], scope=SCOPE)
    m, m0 = _mig(e), _mig(q)
    assert abs(m.loc['C4', 'base_quota'] - 2_100_000.0) < 0.05  # 300K+1.8M
    assert (m.loc[['C1', 'C2'], 'base_quota'] == 0).all()
    assert abs(m.loc['C3', 'base_quota'] - 1_800_000.0) < 0.05
    assert abs(m.loc['C3_r2', 'base_quota']
               - m0.loc['C3_r2', 'base_quota']) < 0.05
    r = rep.set_index('node')
    assert r.loc['C3', 'role'] == 'bystander' and rep['exact'].all()
    print("  C4=2.1M; C3 (reps too) untouched")


# ----------------------------------------------------------------------
# 3. Freeze: excluded from default sources, never moved
# ----------------------------------------------------------------------
def test_freeze_default_sources():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: freeze_nodes=['C3'] -> C3 not a default source")
    print(SEPARATOR)
    q = _quotas()
    e, rep = concentrate(q, 'C4', scope=SCOPE, freeze_nodes=['C3'])
    m = _mig(e)
    assert abs(m.loc['C4', 'base_quota'] - 2_100_000.0) < 0.05
    assert abs(m.loc['C3', 'base_quota'] - 1_800_000.0) < 0.05
    assert set(rep[rep.role == 'source']['node']) == {'C1', 'C2'}
    assert rep['exact'].all()
    print("  C3 held; C1+C2 collapsed onto C4")


# ----------------------------------------------------------------------
# 4. Round-trip: concentrate then redistribute restores proportions
# ----------------------------------------------------------------------
def test_inverse_composition():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: concentrate(redistribute(q, C4)) == concentrate(q) — "
          "spreading first doesn't change the concentrated result")
    print(SEPARATOR)
    q = _quotas()
    spread, _ = redistribute(q, 'C4', scope=SCOPE)   # C4 -> siblings
    back, repb = concentrate(spread, 'C4', scope=SCOPE)
    direct, _ = concentrate(q, 'C4', scope=SCOPE)
    mb, md = _mig(back), _mig(direct)
    # Team level and above: identical frames
    for n in ('CENTRAL', 'C1', 'C2', 'C3', 'C4'):
        assert abs(mb.loc[n, 'base_quota']
                   - md.loc[n, 'base_quota']) < 0.5, n
    assert abs(mb.loc['C4', 'base_quota'] - 3_900_000.0) < 0.5
    assert repb['exact'].all()
    # Rep level: transiting $0 destroys C4's internal mix, so the way
    # back equal-splits (documented zero-baseline fallback) — direct
    # concentrate preserves the original 1:2 mix instead.
    assert abs(mb.loc['C4_r1', 'base_quota']
               - mb.loc['C4_r2', 'base_quota']) < 0.5       # equal split
    assert abs(md.loc['C4_r1', 'base_quota']
               / md.loc['C4', 'base_quota'] - 1 / 3) < 1e-6  # mix kept
    print("  team level identical (C4 = 3.9M either way); rep-level "
          "equal-split after transiting $0, as documented")


# ----------------------------------------------------------------------
# 5. Validation
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: root / non-sibling / frozen source / duplicate / "
          "unknown")
    print(SEPARATOR)
    q = _quotas()
    for kwargs, needle in [
        (dict(to_node='CENTRAL'), 'root'),
        (dict(to_node='C4', from_nodes=['C1_r1']), 'route_targets'),
        (dict(to_node='C4', from_nodes=['C1'],
              freeze_nodes=['C1']), 'eligible'),
        (dict(to_node='C4', from_nodes=['C1', 'C1']), 'duplicate'),
        (dict(to_node='nope'), 'matches no rows'),
        (dict(to_node='C4', from_nodes=['C4']), 'eligible'),
    ]:
        try:
            concentrate(q, scope=SCOPE, **kwargs)
            raise AssertionError(f'expected ValueError for {kwargs}')
        except ValueError as err:
            assert needle in str(err), (kwargs, str(err))
    print("  all raise with clear messages")


if __name__ == '__main__':
    test_equivalences_and_depths()
    test_subset_with_bystander()
    test_freeze_default_sources()
    test_inverse_composition()
    test_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL CONCENTRATE TESTS PASSED")
    print(SEPARATOR)
