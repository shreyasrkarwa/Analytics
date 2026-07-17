"""
Tests for issues #54 / #55 — enforce_identities() + apply_pins'
on_overshoot policy.

Covers:
  - #55 repro: individually-fine pins collectively overshoot a team's
    envelope -> attrs['overshoot_report'] per (parent, combo) + ONE
    summary warning under 'allow' (the v0.29 blind spot, closed)
  - on_overshoot='scale_pins': pins scaled proportionally to fit,
    reconcile() clean afterwards, feasibility report shows
    overshoot_scaled + recomputed achieved_total, warning names pins
  - on_overshoot='error' raises naming the slice
  - standalone enforce_identities: hand-corrupted frames fixed
    (free children rescaled; pinned held when budget allows); clean
    frames returned bit-identical; undershoot with no free children
    left + reported
  - hedged identities restored without a hedge parameter (the #21
    ratio contract, proven via reconcile)
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
    hdf = pd.DataFrame([dict(region='EMEA', team=f'T{i//2+1}',
                             rep=f'r{i+1}', kw=[100, 200, 300, 400][i])
                        for i in range(4)])
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW,
                        hedge_multiplier=1.1)
    return q


OVERPINS = [Pin('r1', 600_000.0), Pin('r2', 600_000.0)]  # T1 holds 600K


# ----------------------------------------------------------------------
# 1. #55 repro: surfaced + warned under default 'allow'
# ----------------------------------------------------------------------
def test_overshoot_surfaced():
    print(SEPARATOR)
    print("TEST 1: collective overshoot -> per-(parent, combo) report "
          "+ one warning (no longer silent)")
    print(SEPARATOR)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, rep = apply_pins(_quotas(), OVERPINS)
    over = pd.DataFrame(e.attrs['overshoot_report'])
    print(over.to_string(index=False))
    assert len(over) == 2                          # T1 x both quarters
    assert set(over['node_id']) == {'T1'}
    assert (over['gap'] - 300_000.0).abs().max() < 0.5
    assert set(over['fiscal_quarter']) == {1, 2}
    hits = [x for x in w if 'collectively break' in str(x.message)]
    assert len(hits) == 1
    assert 'overshoot_scaled' not in rep.columns    # allow: no scaling


# ----------------------------------------------------------------------
# 2. scale_pins: fitted, reconciled, honestly reported
# ----------------------------------------------------------------------
def test_scale_pins_policy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: on_overshoot='scale_pins' -> pins halved to fit, "
          "reconcile clean, report updated")
    print(SEPARATOR)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, rep = apply_pins(_quotas(), OVERPINS,
                            on_overshoot='scale_pins')
    ix = e.set_index(['node_id', 'fiscal_quarter'])
    for fq in (1, 2):                     # 600K budget, 300K+300K pinned
        assert abs(ix.loc[('r1', fq), 'base_quota'] - 150_000.0) < 0.5
        assert abs(ix.loc[('r2', fq), 'base_quota'] - 150_000.0) < 0.5
        assert abs(ix.loc[('T1', fq), 'base_quota'] - 300_000.0) < 0.5
    f = reconcile(e, hedge=1.1)
    assert f['ok'].all()                  # hedged identities restored too
    r = rep.set_index('pin_node')
    assert bool(r.loc['r1', 'overshoot_scaled'])
    assert abs(r.loc['r1', 'achieved_total'] - 300_000.0) < 0.5
    assert not bool(r.loc['r1', 'feasible'])
    assert any('scaled' in str(x.message) for x in w)
    assert e.attrs['overshoot_report']    # detection still recorded
    print("  pins fit the envelope; reconcile ok; achieved recomputed")


# ----------------------------------------------------------------------
# 3. 'error' raises naming the slice
# ----------------------------------------------------------------------
def test_error_policy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: on_overshoot='error' raises with the offenders")
    print(SEPARATOR)
    try:
        apply_pins(_quotas(), OVERPINS, on_overshoot='error')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'T1' in str(e) and 'scale_pins' in str(e)
    print("  raised with pointer to the fix")


# ----------------------------------------------------------------------
# 4. Standalone enforce: manual corruption fixed; clean is a no-op
# ----------------------------------------------------------------------
def test_standalone_enforce():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: hand-corrupted frame fixed (pinned rep held); clean "
          "frame bit-identical")
    print(SEPARATOR)
    q = _quotas()
    # clean no-op
    same, rep0 = enforce_identities(q)
    assert same.set_index(['node_id', 'fiscal_quarter'])['base_quota'] \
        .equals(q.set_index(['node_id', 'fiscal_quarter'])['base_quota'])
    assert rep0.empty
    # corrupt: manually bump r3 on BOTH layers (a ratio-consistent
    # manual reallocation — enforce preserves per-row ratios, it can't
    # invent one; a base-only edit is rehedge() territory)
    q2 = q.copy()
    m = (q2.node_id == 'r3') & (q2.fiscal_quarter == 1)
    q2.loc[m, 'base_quota'] += 100_000.0
    q2.loc[m, 'cascaded_quota'] += 100_000.0 * 1.1 ** 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fixed, rep = enforce_identities(q2)
    ix = fixed.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r3', 1), 'base_quota']
               + ix.loc[('r4', 1), 'base_quota']
               - ix.loc[('T2', 1), 'base_quota']) < 0.05
    assert reconcile(fixed, hedge=1.1)['ok'].all()
    assert rep.iloc[0]['action'] == 'scaled_free'
    # pinned child held: freeze r3 at its corrupted value -> r4 absorbs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fixed2, _ = enforce_identities(q2, freeze_nodes=['r3'])
    ix2 = fixed2.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix2.loc[('r3', 1), 'base_quota']
               - (300_000.0 + 100_000.0)) < 0.5        # held
    assert abs(ix2.loc[('r4', 1), 'base_quota']
               - 300_000.0) < 0.5      # 700K budget - 400K frozen
    print("  free-rescale fix + frozen-hold fix both reconcile")


# ----------------------------------------------------------------------
# 5. Undershoot with no free children: left + reported
# ----------------------------------------------------------------------
def test_undershoot_left():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: pins undershoot with nothing free to grow -> gap "
          "left, warned (pins never scaled UP)")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(q, [Pin('r1', 100_000.0),
                              Pin('r2', 100_000.0)])   # 200K vs 600K T1
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        fixed, rep = enforce_identities(e)
    row = rep[rep.node_id == 'T1'].iloc[0]
    assert row['action'] == 'left' and not row['resolved']
    assert any('unresolved' in str(x.message) for x in w)
    ix = fixed.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc[('r1', 1), 'base_quota'] - 50_000.0) < 0.5  # held
    print("  pinned totals never inflated; gap honestly reported")


if __name__ == '__main__':
    test_overshoot_surfaced()
    test_scale_pins_policy()
    test_error_policy()
    test_standalone_enforce()
    test_undershoot_left()

    print(f"\n\n{SEPARATOR}")
    print("ALL ENFORCE-IDENTITIES TESTS PASSED")
    print(SEPARATOR)
