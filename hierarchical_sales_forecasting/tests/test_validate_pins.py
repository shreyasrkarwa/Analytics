"""
Tests for issue #65 — validate_pins() pre-flight feasibility.

Covers the filer's four incident shapes, caught BEFORE anything runs:
  - STRATEGIC_APAC: base dollars pinned with basis='cascaded' ->
    infeasible + basis_hint naming the fix (both directions)
  - MM_AMER/MM_EMEA: children pinned above the parent with every
    sibling pinned (nothing can shed) -> slack < 0, free_children=0
  - SPEC_FINANCE: 0.1% arithmetic drift -> exact slack, no bogus hint
  - root pins vs targets= (capacity_source='target')
Plus:
  - the ground-truth anchor: validate_pins verdict == apply_pins +
    reconcile on both feasible and infeasible sets
  - free-sibling absorption marked feasible; frozen siblings excluded
    from free capacity
  - conflict column (read-only #63 detection); guards; pure-report
    (input frame untouched)
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, validate_pins,
    reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    hdf = pd.DataFrame([dict(region='A', team='T', rep=f'r{i+1}',
                             kw=100) for i in range(4)])
    t = pd.DataFrame([dict(region='A', tgt=1_100_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, t, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
    return q, t


# ----------------------------------------------------------------------
# 1. The basis mistake, both directions
# ----------------------------------------------------------------------
def test_basis_hint():
    print(SEPARATOR)
    print("TEST 1: base-dollars-as-cascaded and cascaded-dollars-as-"
          "base both named")
    print(SEPARATOR)
    q, t = _quotas()
    v = validate_pins(q, [Pin(f'r{i+1}', 1_100_000.0 / 4,
                              basis='cascaded') for i in range(4)])
    r = v.iloc[0]
    assert not r['feasible'] and r['free_children'] == 0
    assert 'BASE dollars pinned with basis=\'cascaded\'' in \
        r['basis_hint']
    assert '1.2100' in r['basis_hint']
    v2 = validate_pins(q, [Pin(f'r{i+1}', 1_100_000.0 * 1.21 / 4)
                           for i in range(4)])
    assert "CASCADED dollars pinned with basis='base'" in \
        v2.iloc[0]['basis_hint']
    print("  both hints fire with the 1.21 factor named")


# ----------------------------------------------------------------------
# 2. Over-pinned family + 0.1% drift + feasible absorption
# ----------------------------------------------------------------------
def test_slack_shapes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: all-siblings-over-pinned, tiny drift (no bogus "
          "hint), free-sibling feasibility, frozen exclusion")
    print(SEPARATOR)
    q, t = _quotas()
    # every sibling pinned, sum 1.4M vs 1.1M capacity
    v = validate_pins(q, [Pin('r1', 500_000.0), Pin('r2', 500_000.0),
                          Pin('r3', 300_000.0), Pin('r4', 100_000.0)])
    r = v.iloc[0]
    assert abs(r['slack'] + 300_000.0) < 0.5 and r['free_children'] == 0
    assert not r['feasible'] and r['basis_hint'] is None
    # SPEC_FINANCE: 0.1% over, all pinned -> exact slack, no hint
    v2 = validate_pins(q, [Pin(f'r{i+1}', 1_100_000.0 * 1.001 / 4)
                           for i in range(4)])
    r2 = v2.iloc[0]
    assert abs(r2['slack'] + 1_100.0) < 1.0
    assert not r2['feasible'] and r2['basis_hint'] is None
    # 2 pinned, 2 free -> feasible; frozen sibling drops out of free
    v3 = validate_pins(q, [Pin('r1', 500_000.0), Pin('r2', 300_000.0)])
    r3 = v3.iloc[0]
    assert r3['feasible'] and r3['free_children'] == 2
    assert abs(r3['free_capacity'] - 550_000.0) < 0.5
    v4 = validate_pins(q, [Pin('r1', 500_000.0), Pin('r2', 300_000.0)],
                       freeze_nodes=['r3'])
    assert v4.iloc[0]['free_children'] == 1
    print("  slack exact in every shape; hints only when earned")


# ----------------------------------------------------------------------
# 3. Ground-truth anchor vs apply_pins + reconcile
# ----------------------------------------------------------------------
def test_ground_truth_anchor():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: validate_pins verdict matches apply_pins + "
          "reconcile ground truth")
    print(SEPARATOR)
    q, t = _quotas()
    feasible_set = [Pin('r1', 400_000.0), Pin('r2', 300_000.0)]
    infeasible_set = [Pin('r1', 500_000.0), Pin('r2', 500_000.0),
                      Pin('r3', 300_000.0), Pin('r4', 100_000.0)]
    for pins, want in ((feasible_set, True), (infeasible_set, False)):
        v = validate_pins(q, pins)
        assert bool(v['feasible'].all()) == want
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            e, _ = apply_pins(q, pins, on_overshoot='allow')
        assert bool(reconcile(e, hedge=1.1)['ok'].all()) == want
    # pure report: input untouched
    q2, _ = _quotas()
    before = q2['base_quota'].copy()
    validate_pins(q2, infeasible_set)
    assert (q2['base_quota'] == before).all()
    print("  verdicts agree on both sets; frame untouched")


# ----------------------------------------------------------------------
# 4. Root pins vs targets, conflicts, guards
# ----------------------------------------------------------------------
def test_targets_conflicts_guards():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: root pin vs target; conflict column; guards")
    print(SEPARATOR)
    q, t = _quotas()
    # root pin off the plan total -> capacity_source='target'
    v = validate_pins(q, [Pin('A', 900_000.0)], targets=t,
                      target_col='tgt')
    root = v[v.capacity_source == 'target'].iloc[0]
    assert abs(root['slack'] - 200_000.0) < 0.5
    assert not root['feasible']            # root has nothing to absorb
    # matching root pin passes
    v2 = validate_pins(q, [Pin('A', 1_100_000.0)], targets=t,
                       target_col='tgt')
    assert v2[v2.capacity_source == 'target'].iloc[0]['feasible']
    # conflicts annotated read-only
    v3 = validate_pins(q, [Pin('r1', 500_000.0), Pin('r1', 100_000.0)])
    assert 'r1' in v3.iloc[0]['conflict']
    # guards
    try:
        validate_pins(q, [Pin('r1', 1.0)], targets=t)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'target_col' in str(e)
    try:
        validate_pins(q, [Pin('r1', 1.0, scope={'nope': 1})])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'nope' in str(e)
    print("  target capacity + conflicts + guards all work")


if __name__ == '__main__':
    test_basis_hint()
    test_slack_shapes()
    test_ground_truth_anchor()
    test_targets_conflicts_guards()

    print(f"\n\n{SEPARATOR}")
    print("ALL VALIDATE-PINS TESTS PASSED")
    print(SEPARATOR)
