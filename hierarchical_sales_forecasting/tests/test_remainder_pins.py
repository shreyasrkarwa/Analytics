"""
Tests for issue #42 — "remainder to unpinned siblings" as plain pin
composition (no special mode needed since v0.20.0/v0.22.0).

Pins:
  A. sibling pins only -> unpinned siblings take the leftover at
     baseline proportions, parent conserved at its original total
     (deltas deliberately NON-canceling so the proportionality claim
     is actually exercised)
  B. parent pin + child pins == the requested
     Pin(parent, total, children={...}, remainder='auto')
  C. infeasible (children exceed the parent pin) -> floors + every
     dollar accounted for in the feasibility report
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec, Pin, apply_pins

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    """4 teams under EMEA, 2 quarters. Per-quarter baselines:
    T1=300K, T2=700K, T3=400K, T4=600K (totals 2x that)."""
    hdf = pd.DataFrame([
        dict(product='cloud', regional='EMEA', team=t, rep=f'{t}_r{j}',
             kw=k)
        for t, ks in [('T1', [100, 200]), ('T2', [300, 400]),
                      ('T3', [150, 250]), ('T4', [250, 350])]
        for j, k in enumerate(ks)])
    targets = pd.DataFrame([dict(product='cloud', fiscal_quarter=fq,
                                 tgt=2_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['product'],
                        target_col='tgt',
                        taxonomy=['regional', 'team', 'rep'], metrics=KW)
    return q


def _totals(e, nodes):
    e = e.set_index('node_id')
    return {n: float(e.loc[n, 'base_quota'].sum()) for n in nodes}


# ----------------------------------------------------------------------
# A. Sibling pins only: remainder at baseline proportions
# ----------------------------------------------------------------------
def test_remainder_to_unpinned_siblings():
    print(SEPARATOR)
    print("TEST A: Pin(T1)+Pin(T2), non-canceling deltas -> T3:T4 take "
          "the leftover at baseline 0.4:0.6")
    print(SEPARATOR)
    q = _quotas()
    # baselines 600K / 1.4M; net delta = -200K + +300K = +100K (!= 0)
    e, rep = apply_pins(q, [Pin('T1', 400_000.0), Pin('T2', 1_700_000.0)])
    t = _totals(e, ['EMEA', 'T1', 'T2', 'T3', 'T4'])
    remainder = 4_000_000.0 - 400_000.0 - 1_700_000.0
    print(f"  T3={t['T3']:,.2f} T4={t['T4']:,.2f} "
          f"(expected {remainder*0.4:,.0f} / {remainder*0.6:,.0f})")
    assert abs(t['T1'] - 400_000.0) < 0.05
    assert abs(t['T2'] - 1_700_000.0) < 0.05
    assert abs(t['T3'] - remainder * 0.4) < 0.5   # baseline 800K:1.2M
    assert abs(t['T4'] - remainder * 0.6) < 0.5
    assert abs(t['EMEA'] - 4_000_000.0) < 0.05    # parent conserved
    assert rep['feasible'].all()


# ----------------------------------------------------------------------
# B. The literal Ask: Pin(parent, total) + exact children
# ----------------------------------------------------------------------
def test_parent_total_with_fixed_children():
    print(f"\n\n{SEPARATOR}")
    print("TEST B: Pin(EMEA, 5M) + Pin(T1)+Pin(T2) == "
          "Pin(parent, total, children={...}, remainder='auto')")
    print(SEPARATOR)
    q = _quotas()
    e, rep = apply_pins(q, [Pin('EMEA', 5_000_000.0),
                            Pin('T1', 500_000.0), Pin('T2', 1_500_000.0)])
    t = _totals(e, ['EMEA', 'T1', 'T2', 'T3', 'T4'])
    remainder = 5_000_000.0 - 500_000.0 - 1_500_000.0
    print(f"  EMEA={t['EMEA']:,.2f}  T3={t['T3']:,.2f}  T4={t['T4']:,.2f}")
    assert abs(t['EMEA'] - 5_000_000.0) < 0.05
    assert abs(t['T1'] - 500_000.0) < 0.05
    assert abs(t['T2'] - 1_500_000.0) < 0.05
    assert abs(t['T3'] - remainder * 0.4) < 0.5
    assert abs(t['T4'] - remainder * 0.6) < 0.5
    # per-quarter structural conservation
    ix = e.set_index(['node_id', 'fiscal_quarter'])
    for fq in (1, 2):
        assert abs(ix.loc[('EMEA', fq), 'base_quota']
                   - sum(ix.loc[(x, fq), 'base_quota']
                         for x in ['T1', 'T2', 'T3', 'T4'])) < 0.05
    # reps inside the remainder teams keep their internal mix
    assert abs(ix.loc[('T3_r0', 1), 'base_quota']
               / ix.loc[('T3', 1), 'base_quota'] - 150 / 400) < 0.001


# ----------------------------------------------------------------------
# C. Infeasible: children exceed the parent pin -> honest accounting
# ----------------------------------------------------------------------
def test_overpinned_children_accounted():
    print(f"\n\n{SEPARATOR}")
    print("TEST C: children (1.7M) > parent pin (1.5M) -> floors + every "
          "dollar in the report")
    print(SEPARATOR)
    q = _quotas()
    import warnings as w
    with w.catch_warnings():
        w.simplefilter('ignore')
        e, rep = apply_pins(q, [Pin('EMEA', 1_500_000.0),
                                Pin('T1', 800_000.0),
                                Pin('T2', 900_000.0)])
    r = rep.set_index('pin_node')
    # the parent-child gap = parent's subtree_shortfall + the CHILD
    # pins' unabsorbed (the root pin's own 'unabsorbed' is a separate
    # bucket: its delta vs nonexistent root siblings)
    gap = (r.loc['EMEA', 'subtree_shortfall']
           + r.loc[['T1', 'T2'], 'unabsorbed'].sum())
    t = _totals(e, ['EMEA', 'T1', 'T2', 'T3', 'T4'])
    child_sum = t['T1'] + t['T2'] + t['T3'] + t['T4']
    print(f"  children sum {child_sum:,.2f} vs parent {t['EMEA']:,.2f}; "
          f"reported gap {gap:,.2f}")
    assert not bool(r.loc['EMEA', 'feasible'])
    # the parent-child inconsistency equals the reported gap exactly
    assert abs((child_sum - t['EMEA']) - gap) < 0.5
    assert (e.set_index('node_id')['base_quota'] >= 0).all()  # no negatives


if __name__ == '__main__':
    test_remainder_to_unpinned_siblings()
    test_parent_total_with_fixed_children()
    test_overpinned_children_accounted()

    print(f"\n\n{SEPARATOR}")
    print("ALL REMAINDER-PIN TESTS PASSED")
    print(SEPARATOR)
