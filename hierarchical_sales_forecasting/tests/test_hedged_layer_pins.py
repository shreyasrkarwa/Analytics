"""
Tests for issue #39 — pins on the hedged layer + protection-aware
subtree rescaling in apply_pins.

Covers:
  - the DISPROVEN headline claim: a cascaded-basis manager pin under
    HedgeByDepth DOES make descendants roll up to pinned x cross-level
    hedge (regression anchor so it stays true)
  - order-independence: descendant pin + manager pin give the same
    frame in either order, both pins held
  - freeze_nodes honored INSIDE a pinned subtree
  - Pin.exclude protects descendants of a manager pin
  - infeasible pin (protected mass > pin): $0 floors + subtree_shortfall
  - absorber subtrees respect protection (free-capacity absorption)
  - equivalence: with nothing protected, v0.20.0 == proportional rescale
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy, QuotaCascader, MetricSpec, HedgeByDepth,
    Pin, apply_pins,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
HEDGE = HedgeByDepth(from_leaves={1: 1.10, 2: 1.05})


def _long(df=None, target=10_000_000.0):
    """Cascade with the issue-#39 hedge and return the long frame with
    cascade_many-style column names."""
    src = df if df is not None else pd.DataFrame([
        dict(regional='EMEA', team='T1', rep='r1', kw=100),
        dict(regional='EMEA', team='T1', rep='r2', kw=300),
        dict(regional='EMEA', team='T2', rep='r3', kw=200),
        dict(regional='EMEA', team='T2', rep='r4', kw=400),
    ])
    h = SalesHierarchy()
    h.from_dataframe(src, path_cols=TAXONOMY, metrics_cols=['kw'])
    c = QuotaCascader(h)
    q = c.cascade_quota('EMEA', target, metrics=KW,
                        hedge_multiplier=HEDGE, verbose=False)
    out = c.quotas_to_dataframe(q, level_names=TAXONOMY,
                                unhedged_quotas='auto')
    return out.rename(columns={'unhedged_quota': 'base_quota'})


def _ix(df):
    return df.set_index('node_id')


# ----------------------------------------------------------------------
# 1. The headline claim of #39, disproven and pinned
# ----------------------------------------------------------------------
def test_issue39_hedged_pin_identity():
    print(SEPARATOR)
    print("TEST 1: cascaded-basis manager pin -> descendants roll to "
          "pinned x cross-level hedge")
    print(SEPARATOR)
    long = _long()
    pt = 5_750_000.0
    edited, rep = apply_pins(long, [Pin('T1', pt, basis='cascaded')],
                             row_keys=[])
    e = _ix(edited)
    reps_base = e.loc[['r1', 'r2'], 'base_quota'].sum()
    reps_casc = e.loc[['r1', 'r2'], 'cascaded_quota'].sum()
    print(f"  T1 cascaded={e.loc['T1', 'cascaded_quota']:,.2f}  "
          f"reps casc sum={reps_casc:,.2f}  pinned x 1.10="
          f"{pt * 1.10:,.2f}")
    assert abs(e.loc['T1', 'cascaded_quota'] - pt) < 0.5
    assert abs(reps_base - e.loc['T1', 'base_quota']) < 0.05
    assert abs(reps_casc - pt * 1.10) < 0.5          # the #39 identity
    # each rep keeps its own hedge ratio relative to the NEW base
    assert abs(e.loc['r1', 'cascaded_quota']
               - e.loc['r1', 'base_quota'] * 1.155) < 0.5
    assert bool(rep.iloc[0]['feasible'])


# ----------------------------------------------------------------------
# 2. Pin order no longer matters (the REAL bug behind #39)
# ----------------------------------------------------------------------
def test_pin_order_independence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: rep pin + team pin land identically in either order")
    print(SEPARATOR)
    long = _long()
    pt = 5_750_000.0
    p_rep = Pin('r1', 500_000.0)
    p_team = Pin('T1', pt, basis='cascaded')
    a = _ix(apply_pins(long, [p_rep, p_team], row_keys=[])[0])
    b = _ix(apply_pins(long, [p_team, p_rep], row_keys=[])[0])
    for order, e in (('rep-first', a), ('team-first', b)):
        print(f"  {order}: r1={e.loc['r1', 'base_quota']:,.2f}  "
              f"T1 casc={e.loc['T1', 'cascaded_quota']:,.2f}")
        assert abs(e.loc['r1', 'base_quota'] - 500_000.0) < 0.05, order
        assert abs(e.loc['T1', 'cascaded_quota'] - pt) < 0.5, order
        # conservation inside the pinned subtree
        assert abs(e.loc[['r1', 'r2'], 'base_quota'].sum()
                   - e.loc['T1', 'base_quota']) < 0.05, order
    for n in a.index:
        assert abs(a.loc[n, 'base_quota'] - b.loc[n, 'base_quota']) < 0.05, n


# ----------------------------------------------------------------------
# 3. freeze_nodes honored inside a pinned subtree
# ----------------------------------------------------------------------
def test_frozen_descendant_untouched():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: frozen rep inside a pinned team keeps its value; "
          "free sibling stretches")
    print(SEPARATOR)
    long = _long()
    nat = _ix(long)
    pt = 5_750_000.0
    e = _ix(apply_pins(long, [Pin('T1', pt, basis='cascaded')],
                       freeze_nodes=['r1'], row_keys=[])[0])
    print(f"  r1 {nat.loc['r1', 'base_quota']:,.2f} -> "
          f"{e.loc['r1', 'base_quota']:,.2f} (frozen)  r2 -> "
          f"{e.loc['r2', 'base_quota']:,.2f}")
    assert abs(e.loc['r1', 'base_quota']
               - nat.loc['r1', 'base_quota']) < 0.05
    assert abs(e.loc['r1', 'cascaded_quota']
               - nat.loc['r1', 'cascaded_quota']) < 0.05
    # r2 takes the whole remainder; subtree still sums to the pin
    assert abs(e.loc[['r1', 'r2'], 'base_quota'].sum()
               - e.loc['T1', 'base_quota']) < 0.05
    assert abs(e.loc['T1', 'cascaded_quota'] - pt) < 0.5


# ----------------------------------------------------------------------
# 4. Pin.exclude protects descendants of the pinned manager
# ----------------------------------------------------------------------
def test_exclude_protects_descendants():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: Pin(T1, exclude=['r1']) leaves r1 untouched (the "
          "#39 Ask)")
    print(SEPARATOR)
    long = _long()
    nat = _ix(long)
    pt = 5_750_000.0
    e = _ix(apply_pins(long,
                       [Pin('T1', pt, basis='cascaded', exclude=['r1'])],
                       row_keys=[])[0])
    print(f"  r1 {nat.loc['r1', 'base_quota']:,.2f} -> "
          f"{e.loc['r1', 'base_quota']:,.2f}")
    assert abs(e.loc['r1', 'base_quota']
               - nat.loc['r1', 'base_quota']) < 0.05
    assert abs(e.loc[['r1', 'r2'], 'base_quota'].sum()
               - e.loc['T1', 'base_quota']) < 0.05
    assert abs(e.loc['T1', 'cascaded_quota'] - pt) < 0.5


# ----------------------------------------------------------------------
# 5. Infeasible: protected mass exceeds the pin -> floors + shortfall
# ----------------------------------------------------------------------
def test_infeasible_protected_exceeds_pin():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: frozen mass > pinned total -> free rows floor at $0, "
          "subtree_shortfall reported")
    print(SEPARATOR)
    long = _long()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        edited, rep = apply_pins(long, [Pin('T1', 800_000.0)],
                                 freeze_nodes=['r1'], row_keys=[])
    e = _ix(edited)
    r = rep.iloc[0]
    print(f"  r1(frozen)={e.loc['r1', 'base_quota']:,.2f}  "
          f"r2={e.loc['r2', 'base_quota']:,.2f}  "
          f"shortfall={r['subtree_shortfall']:,.2f}")
    assert abs(e.loc['r1', 'base_quota'] - 1_000_000.0) < 0.05  # frozen
    assert e.loc['r2', 'base_quota'] == 0.0                     # floored
    assert abs(r['subtree_shortfall'] - 200_000.0) < 0.5
    assert not bool(r['feasible'])
    assert any('subtree_shortfall' in str(x.message) for x in w)


# ----------------------------------------------------------------------
# 6. Absorber subtrees respect protection (free-capacity absorption)
# ----------------------------------------------------------------------
def test_absorber_subtree_protection():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: T1 pinned up; frozen r3 under absorber T2 untouched; "
          "r4 sheds it all")
    print(SEPARATOR)
    long = _long()
    nat = _ix(long)
    e = _ix(apply_pins(long, [Pin('T1', 5_000_000.0)],
                       freeze_nodes=['r3'], row_keys=[])[0])
    shed = 1_000_000.0                        # T1: 4M -> 5M
    print(f"  r3 {nat.loc['r3', 'base_quota']:,.2f} -> "
          f"{e.loc['r3', 'base_quota']:,.2f}  r4 -> "
          f"{e.loc['r4', 'base_quota']:,.2f}")
    assert abs(e.loc['r3', 'base_quota']
               - nat.loc['r3', 'base_quota']) < 0.05
    assert abs(e.loc['r4', 'base_quota']
               - (nat.loc['r4', 'base_quota'] - shed)) < 0.05
    # T2 internally consistent and root conserved on the base layer
    assert abs(e.loc[['r3', 'r4'], 'base_quota'].sum()
               - e.loc['T2', 'base_quota']) < 0.05
    assert abs(e.loc[['T1', 'T2'], 'base_quota'].sum()
               - e.loc['EMEA', 'base_quota']) < 0.05


# ----------------------------------------------------------------------
# 7. Nothing protected -> exactly the old proportional rescale
# ----------------------------------------------------------------------
def test_unprotected_equivalence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 7: no protection -> mix preserved, proportional rescale")
    print(SEPARATOR)
    long = _long()
    e = _ix(apply_pins(long, [Pin('T1', 5_000_000.0)], row_keys=[])[0])
    # r1:r2 stays 1:3, subtree conserved, hedge ratios per row intact
    assert abs(e.loc['r1', 'base_quota'] - 1_250_000.0) < 0.05
    assert abs(e.loc['r2', 'base_quota'] - 3_750_000.0) < 0.05
    assert abs(e.loc['r1', 'cascaded_quota']
               - 1_250_000.0 * 1.155) < 0.5
    print(f"  r1={e.loc['r1', 'base_quota']:,.2f}  "
          f"r2={e.loc['r2', 'base_quota']:,.2f} (1:3 preserved)")


if __name__ == '__main__':
    test_issue39_hedged_pin_identity()
    test_pin_order_independence()
    test_frozen_descendant_untouched()
    test_exclude_protects_descendants()
    test_infeasible_protected_exceeds_pin()
    test_absorber_subtree_protection()
    test_unprotected_equivalence()

    print(f"\n\n{SEPARATOR}")
    print("ALL HEDGED-LAYER PIN TESTS PASSED")
    print(SEPARATOR)
