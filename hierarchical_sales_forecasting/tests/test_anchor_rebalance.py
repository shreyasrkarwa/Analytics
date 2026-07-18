"""
Tests for issues #58 / #59 — enforce_identities anchoring + rebalance.

Covers:
  - #59's exact money-loss shape: aggregate-correct pins, per-combo
    over/undershoot. 'rebalance' conserves the node total AND passes
    reconcile — the combination neither old mode could deliver
  - aggregate pins stay EXACT under rebalance (a Pin is an aggregate)
  - anchor='leaves' (#58): leaf pins stand, parents derived as child
    sums, root floats up by the accepted delta; pins never scaled
  - rebalance fallback: genuinely-off aggregates still scale per
    combo, with the factor column + node@combo warning (the #58
    audit ask)
  - clean frames are no-ops under both anchors
  - apply_pins(on_overshoot='rebalance') passthrough
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


def _quotas():
    """T = 600K per quarter (1.2M agg); r1/r2 300K each per quarter."""
    hdf = pd.DataFrame([dict(region='EMEA', team='T', rep=f'r{i+1}',
                             kw=100) for i in range(2)])
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=600_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW,
                        hedge_multiplier=1.1)
    return q


def _concentrated():
    """#59's shape: Q1 pins 500+300=800K (over T's 600K by +200K), Q2
    pins 250+150=400K (under by -200K). NETTING: aggregate pins ==
    T aggregate == 1.2M — deliberate per-combo concentration."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(
            _quotas(),
            [Pin('r1', 500_000.0, scope={'fiscal_quarter': 1}),
             Pin('r2', 300_000.0, scope={'fiscal_quarter': 1}),
             Pin('r1', 250_000.0, scope={'fiscal_quarter': 2}),
             Pin('r2', 150_000.0, scope={'fiscal_quarter': 2})],
            on_overshoot='allow')
    return e


# ----------------------------------------------------------------------
# 1. #59: rebalance conserves AND reconciles
# ----------------------------------------------------------------------
def test_rebalance_conserves():
    print(SEPARATOR)
    print("TEST 1: #59's shape — rebalance floats T's combos to the "
          "child sums; nothing lost")
    print(SEPARATOR)
    e = _concentrated()
    reps_before = e[e.node_id.isin(['r1', 'r2'])]['base_quota'].sum()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f, rep = enforce_identities(e, on_overshoot='rebalance')
    ix = f.set_index(['node_id', 'fiscal_quarter'])
    # per-combo identities now hold: T floats to 800K / 200K
    assert abs(ix.loc[('T', 1), 'base_quota'] - 800_000.0) < 0.5
    assert abs(ix.loc[('T', 2), 'base_quota'] - 400_000.0) < 0.5
    # reps untouched — pins exact, aggregate conserved
    reps_after = f[f.node_id.isin(['r1', 'r2'])]['base_quota'].sum()
    assert abs(reps_after - reps_before) < 0.05
    assert abs(ix.loc[('r1', 1), 'base_quota'] - 500_000.0) < 0.05
    # root floats per combo too, aggregate conserved
    assert abs(ix.loc['EMEA', 'base_quota'].sum() - 1_200_000.0) < 0.5
    assert reconcile(f, hedge=1.1)['ok'].all()
    acts = set(rep['action'])
    assert acts == {'rebalanced'}
    print(f"  T -> 800K/200K, reps exact, EMEA agg conserved, "
          f"reconcile ok; actions={acts}")


# ----------------------------------------------------------------------
# 2. The old modes' failure shapes, pinned as documentation
# ----------------------------------------------------------------------
def test_old_modes_documented():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: scale_pins is one-directional (loses the "
          "overshoot); allow leaves identities broken")
    print(SEPARATOR)
    e = _concentrated()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f1, _ = enforce_identities(e, on_overshoot='scale_pins')
    lost = (e[e.node_id.isin(['r1', 'r2'])]['base_quota'].sum()
            - f1[f1.node_id.isin(['r1', 'r2'])]['base_quota'].sum())
    assert abs(lost - 200_000.0) < 0.5          # exactly the overshoot
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f2, _ = enforce_identities(e, on_overshoot='allow')
    assert not reconcile(f2, hedge=1.1)['ok'].all()
    print(f"  scale_pins lost {lost:,.0f}; allow fails reconcile — "
          f"rebalance is the only mode doing both")


# ----------------------------------------------------------------------
# 3. #58: anchor='leaves' — bottom-up, root floats
# ----------------------------------------------------------------------
def test_anchor_leaves():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: anchor='leaves' — parents derived, root floats, "
          "pins never touched")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(q, [Pin('r1', 800_000.0),
                              Pin('r2', 700_000.0)],
                          on_overshoot='allow')   # 1.5M vs T's 1.2M
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f, rep = enforce_identities(e, anchor='leaves')
    ix = f.set_index(['node_id', 'fiscal_quarter'])
    # leaf pins stand exactly
    assert abs(f[f.node_id == 'r1']['base_quota'].sum()
               - 800_000.0) < 0.05
    # parents derived as sums; root floated UP by the accepted +300K
    for fq in (1, 2):
        assert abs(ix.loc[('T', fq), 'base_quota']
                   - (ix.loc[('r1', fq), 'base_quota']
                      + ix.loc[('r2', fq), 'base_quota'])) < 0.05
    assert abs(f[f.node_id == 'EMEA']['base_quota'].sum()
               - 1_500_000.0) < 0.5
    assert reconcile(f, hedge=1.1)['ok'].all()
    assert set(rep['action']) == {'floated'}
    print("  pins exact; EMEA floated 1.2M -> 1.5M; reconcile ok")


# ----------------------------------------------------------------------
# 4. Rebalance fallback: off aggregates still scale, with factors
# ----------------------------------------------------------------------
def test_rebalance_fallback_factors():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: genuinely-off aggregate -> per-combo scale with "
          "factor column + node@combo warning")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(q, [Pin('r1', 900_000.0),
                              Pin('r2', 900_000.0)],
                          on_overshoot='allow')   # 1.8M vs 1.2M: off
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f, rep = enforce_identities(e, on_overshoot='rebalance')
    scaled = rep[rep.action == 'scaled_pins']
    assert len(scaled) == 2                       # both quarters
    assert (scaled['factor'] - 600_000.0 / 900_000.0).abs().max() < 1e-6
    msg = [str(x.message) for x in w
           if 'scaled pinned node' in str(x.message)]
    assert msg and 'x0.6667' in msg[0] and 'r1@' in msg[0]
    assert reconcile(f, hedge=1.1)['ok'].all()
    print(f"  factors 0.6667 recorded + named; reconcile ok")


# ----------------------------------------------------------------------
# 5. Clean no-ops + apply_pins passthrough
# ----------------------------------------------------------------------
def test_noops_and_passthrough():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: clean frames are no-ops for both anchors; "
          "apply_pins(on_overshoot='rebalance') works")
    print(SEPARATOR)
    q = _quotas()
    for kw in (dict(anchor='leaves'), dict(on_overshoot='rebalance'),
               dict()):
        same, rep = enforce_identities(q, **kw)
        assert same.set_index(['node_id', 'fiscal_quarter'])[
            'base_quota'].equals(
            q.set_index(['node_id', 'fiscal_quarter'])['base_quota'])
        assert rep.empty, kw
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, rep = apply_pins(
            q, [Pin('r1', 500_000.0, scope={'fiscal_quarter': 1}),
                Pin('r2', 300_000.0, scope={'fiscal_quarter': 1}),
                Pin('r1', 250_000.0, scope={'fiscal_quarter': 2}),
                Pin('r2', 150_000.0, scope={'fiscal_quarter': 2})],
            on_overshoot='rebalance')
    assert reconcile(e, hedge=1.1)['ok'].all()
    ix = e.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r1', 1), 'base_quota'] - 500_000.0) < 0.05
    print("  no-ops bit-identical; passthrough reconciles with pins "
          "exact")


if __name__ == '__main__':
    test_rebalance_conserves()
    test_old_modes_documented()
    test_anchor_leaves()
    test_rebalance_fallback_factors()
    test_noops_and_passthrough()

    print(f"\n\n{SEPARATOR}")
    print("ALL ANCHOR/REBALANCE TESTS PASSED")
    print(SEPARATOR)
