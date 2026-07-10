"""
Tests for issue #43 — redistribute(): move a node's scoped quota to
its siblings, reshaping source and destination subtrees at every
depth. Thin sugar over apply_pins; these tests pin the equivalences.

Covers:
  - default (proportional, all siblings) == the single scoped Pin
  - all-depth reshaping: source subtree zeroed, destination teams/reps
    scaled consistently, parent conserved, hedge ratio preserved,
    other scopes untouched
  - dict weights (70/30) with a bystander sibling back at baseline
    EXACTLY; no-buffer case (all siblings are recipients) is silent
  - equal split via to_nodes
  - validation: root source, non-sibling recipient (-> route_targets),
    bad weights, frozen recipients
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, redistribute,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
SCOPE = {'st': 'Migration'}


def _quotas():
    """AMER -> EAST/CENTRAL/WEST/NORTH -> 2 teams -> 2 reps, for two
    sales types; compounding hedge. Migration region baselines:
    EAST=1M, CENTRAL=2M, WEST=3M, NORTH=4M."""
    rows = []
    for rg, b in [('EAST', 100), ('CENTRAL', 200), ('WEST', 300),
                  ('NORTH', 400)]:
        for t in (1, 2):
            for r in (1, 2):
                rows.append(dict(st='Migration', amer='AMER', region=rg,
                                 team=f'{rg}_T{t}',
                                 rep=f'{rg}_T{t}_r{r}', kw=b * t * r))
    hdf = pd.DataFrame(rows)
    hdf = pd.concat([hdf, hdf.assign(st='NN')], ignore_index=True)
    targets = pd.DataFrame([dict(st='Migration', tgt=10_000_000.0),
                            dict(st='NN', tgt=8_000_000.0)])
    q, _ = cascade_many(hdf, targets, group_keys=['st'],
                        target_col='tgt',
                        taxonomy=['amer', 'region', 'team', 'rep'],
                        metrics=KW, hedge_multiplier=1.1)
    return q


def _mig(e):
    return e[e.st == 'Migration'].set_index('node_id')


# ----------------------------------------------------------------------
# 1. Default == the single scoped Pin; every depth reshaped
# ----------------------------------------------------------------------
def test_default_equals_single_pin():
    print(SEPARATOR)
    print("TEST 1: redistribute default == Pin(EAST, 0, scope) — all "
          "depths, hedge, scopes")
    print(SEPARATOR)
    q = _quotas()
    e1, rep = redistribute(q, 'EAST', scope=SCOPE)
    e2, _ = apply_pins(q, [Pin('EAST', 0.0, scope=SCOPE)])
    a = e1.set_index(['node_id', 'st'])['base_quota'].sort_index()
    b = e2.set_index(['node_id', 'st'])['base_quota'].sort_index()
    assert ((a - b).abs() < 0.05).all()          # the identity
    m, m0 = _mig(e1), _mig(q)
    east = [n for n in m.index if n.startswith('EAST')]
    assert (m.loc[east, 'base_quota'] == 0).all()          # depths 1-3
    for d in ('CENTRAL', 'WEST', 'NORTH'):                 # proportional
        assert abs(m.loc[d, 'base_quota'] / m0.loc[d, 'base_quota']
                   - 10 / 9) < 1e-6
    assert abs(m.loc['CENTRAL_T2', 'base_quota']
               - m.loc[['CENTRAL_T2_r1', 'CENTRAL_T2_r2'],
                       'base_quota'].sum()) < 0.05         # depth consistency
    assert abs(m.loc['AMER', 'base_quota']
               - m0.loc['AMER', 'base_quota']) < 0.05      # conserved
    r = m.loc['CENTRAL_T2_r1']
    assert abs(r['cascaded_quota'] / r['base_quota'] - 1.1 ** 3) < 1e-6
    nn = e1[e1.st == 'NN'].set_index('node_id')['base_quota']
    assert nn.equals(q[q.st == 'NN'].set_index('node_id')['base_quota'])
    assert rep['exact'].all() and (rep['role'] == 'destination').sum() == 3
    print("  identity holds; subtree zeroed; 10/9 growth; hedge 1.331 "
          "preserved; NN untouched")


# ----------------------------------------------------------------------
# 2. Dict weights + bystander back at baseline exactly
# ----------------------------------------------------------------------
def test_dict_weights_with_bystander():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: weights={CENTRAL:.7, WEST:.3}; NORTH is a bystander")
    print(SEPARATOR)
    q = _quotas()
    e, rep = redistribute(q, 'EAST', weights={'CENTRAL': .7, 'WEST': .3},
                          scope=SCOPE)
    m, m0 = _mig(e), _mig(q)
    assert abs(m.loc['CENTRAL', 'base_quota'] - 2_700_000.0) < 0.05
    assert abs(m.loc['WEST', 'base_quota'] - 3_300_000.0) < 0.05
    # bystander exact, reps included
    assert abs(m.loc['NORTH', 'base_quota'] - 4_000_000.0) < 0.05
    assert abs(m.loc['NORTH_T1_r1', 'base_quota']
               - m0.loc['NORTH_T1_r1', 'base_quota']) < 0.05
    r = rep.set_index('node')
    assert r.loc['NORTH', 'role'] == 'bystander' and r['exact'].all()
    print(f"  CENTRAL=2.7M WEST=3.3M NORTH untouched (reps too)")


# ----------------------------------------------------------------------
# 3. No-buffer case (every sibling a recipient) is exact AND silent
# ----------------------------------------------------------------------
def test_no_buffer_silent():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: all 3 siblings are recipients -> no spurious "
          "'could not be absorbed' warnings")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, rep = redistribute(q, 'EAST',
                              weights={'CENTRAL': .5, 'WEST': .3,
                                       'NORTH': .2}, scope=SCOPE)
    noise = [x for x in w if 'could not be absorbed' in str(x.message)]
    assert not noise, "absorption noise leaked"
    m = _mig(e)
    assert abs(m.loc['CENTRAL', 'base_quota'] - 2_500_000.0) < 0.05
    assert abs(m.loc['NORTH', 'base_quota'] - 4_200_000.0) < 0.05
    assert abs(m.loc['AMER', 'base_quota'] - 10_000_000.0) < 0.05
    assert rep['exact'].all()
    print("  exact + silent")


# ----------------------------------------------------------------------
# 4. equal split via to_nodes
# ----------------------------------------------------------------------
def test_equal_split():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: weights='equal', to_nodes=[CENTRAL, WEST]")
    print(SEPARATOR)
    q = _quotas()
    e, rep = redistribute(q, 'EAST', to_nodes=['CENTRAL', 'WEST'],
                          weights='equal', scope=SCOPE)
    m = _mig(e)
    assert abs(m.loc['CENTRAL', 'base_quota'] - 2_500_000.0) < 0.05
    assert abs(m.loc['WEST', 'base_quota'] - 3_500_000.0) < 0.05
    assert abs(m.loc['NORTH', 'base_quota'] - 4_000_000.0) < 0.05
    assert rep['exact'].all()
    print("  +500K each; NORTH untouched")


# ----------------------------------------------------------------------
# 5. Validation
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: root source / non-sibling / bad weights / frozen")
    print(SEPARATOR)
    q = _quotas()
    for kwargs, needle in [
        (dict(from_node='AMER'), 'root'),
        (dict(from_node='EAST', to_nodes=['CENTRAL_T1']),
         'route_targets'),                         # a niece, not a sibling
        (dict(from_node='EAST', weights={'CENTRAL': -1.0}),
         'non-negative'),
        (dict(from_node='EAST', to_nodes=['CENTRAL'],
              weights={'WEST': 1.0}), 'disagree'),
        (dict(from_node='EAST', to_nodes=['CENTRAL'],
              freeze_nodes=['CENTRAL']), 'eligible'),
        (dict(from_node='nope'), 'matches no rows'),
    ]:
        try:
            redistribute(q, scope=SCOPE, **kwargs)
            raise AssertionError(f'expected ValueError for {kwargs}')
        except ValueError as err:
            assert needle in str(err), (kwargs, str(err))
    # frozen siblings are excluded from DEFAULT recipients
    e, rep = redistribute(q, 'EAST', scope=SCOPE, freeze_nodes=['NORTH'])
    m = _mig(e)
    assert abs(m.loc['NORTH', 'base_quota'] - 4_000_000.0) < 0.05
    assert set(rep[rep.role == 'destination']['node']) == {'CENTRAL',
                                                           'WEST'}
    print("  all raise with clear messages; frozen sibling skipped")


if __name__ == '__main__':
    test_default_equals_single_pin()
    test_dict_weights_with_bystander()
    test_no_buffer_silent()
    test_equal_split()
    test_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL REDISTRIBUTE TESTS PASSED")
    print(SEPARATOR)
