"""
Tests for the v0.13.0 override release — issues #28, #21, #23.

  #28: pins conserve the parent (renormalization was already correct —
       pinned regression proof) and the four REAL defects are fixed:
       all-unpinned-brand-new, pin>pool negatives, manager pins ignored,
       jagged-leaf pins ignored.
  #23: override_basis='base' (default) / 'cascaded' semantics with
       HedgeByDepth.
  #21: hedge_ratios() + rehedge() — edited base layers re-derive the
       hedged layer without polluting parents.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    HedgeByDepth,
)

SEPARATOR = "=" * 90
M = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]


def _build(rows, path=('pool', 'team', 'rep')):
    h = SalesHierarchy()
    h.from_dataframe(pd.DataFrame(rows), path_cols=list(path),
                     metrics_cols=['kw'])
    return h


# ----------------------------------------------------------------------
# 1. Issue #28 repro — parent conservation with a below-natural pin
#    (regression proof: renormalization was already correct)
# ----------------------------------------------------------------------
def test_issue28_repro_conserves():
    print(SEPARATOR)
    print("TEST 1: #28 repro — 5 reps, $5M pool, pin $2.6M -> total exactly "
          "$5M (the claimed shortfall does not exist)")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T', rep=f'r{i}', kw=k)
            for i, k in enumerate([3000, 500, 600, 400, 500])]
    c = QuotaCascader(_build(rows))
    q = c.cascade_quota('P', 5_000_000.0, metrics=M,
                        new_ic_overrides={'r0': 2_600_000.0}, verbose=False)
    total = sum(q[f'r{i}'] for i in range(5))
    print(f"  team total: ${total:,.2f} · r0 pinned: ${q['r0']:,.2f}")
    assert abs(total - 5_000_000.0) < 0.01
    assert q['r0'] == 2_600_000.0
    # unpinned shares renormalized: r2 (600) gets 600/2000 of $2.4M
    assert abs(q['r2'] - 2_400_000.0 * 600 / 2000) < 0.01


# ----------------------------------------------------------------------
# 2. Pin + all remaining reps brand-new — now conserves (was $6.6M/$5M)
# ----------------------------------------------------------------------
def test_pin_with_all_brand_new_conserves():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: pin + all-unpinned-brand-new — pool split equally, "
          "parent conserved (was over-distributing)")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T', rep='r0', kw=3000)] + \
           [dict(pool='P', team='T', rep=f'r{i}', kw=0) for i in range(1, 5)]
    c = QuotaCascader(_build(rows))
    q = c.cascade_quota('P', 5_000_000.0, metrics=M,
                        new_ic_overrides={'r0': 2_600_000.0}, verbose=False)
    total = sum(q[f'r{i}'] for i in range(5))
    print(f"  total: ${total:,.2f} · each new IC: ${q['r1']:,.2f} "
          f"(expected $600,000)")
    assert abs(total - 5_000_000.0) < 0.01
    assert abs(q['r1'] - 600_000.0) < 0.01          # (5M - 2.6M) / 4


# ----------------------------------------------------------------------
# 3. Pin exceeding the pool — $0 siblings, loud, reported; no negatives
# ----------------------------------------------------------------------
def test_overpin_never_negative():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: pin > pool — siblings $0 (never negative), warning + "
          "gating_report exposure")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T', rep=f'r{i}', kw=k)
            for i, k in enumerate([100, 200, 300])]
    c = QuotaCascader(_build(rows))
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        q = c.cascade_quota('P', 1_000_000.0, metrics=M,
                            new_ic_overrides={'r0': 1_500_000.0},
                            verbose=False)
    assert q['r1'] == 0.0 and q['r2'] == 0.0        # was -200k / -300k
    assert q['r0'] == 1_500_000.0
    assert any('exceeding its pool' in str(w.message) for w in wlog)
    rep = c.gating_report()
    print(f"  overpinned_amount: ${rep['overpinned_amount']:,.2f} at "
          f"{list(rep['overpinned_nodes'])}")
    assert abs(rep['overpinned_amount'] - 500_000.0) < 0.01
    assert not rep['reconciles']                     # honest about it


# ----------------------------------------------------------------------
# 4. Manager pin honored — subtree total fixed, cascades within
# ----------------------------------------------------------------------
def test_manager_pin_honored():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: pinning a MANAGER fixes the subtree total (was silently "
          "ignored)")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T1', rep='a', kw=100),
            dict(pool='P', team='T1', rep='b', kw=300),
            dict(pool='P', team='T2', rep='c', kw=200)]
    c = QuotaCascader(_build(rows))
    q = c.cascade_quota('P', 1_000_000.0, metrics=M,
                        new_ic_overrides={'T1': 700_000.0}, verbose=False)
    print(f"  T1: ${q['T1']:,.2f} (pinned) · T2: ${q['T2']:,.2f} · "
          f"a: ${q['a']:,.2f} b: ${q['b']:,.2f}")
    assert q['T1'] == 700_000.0                      # was 500,000 (ignored)
    assert abs(q['T2'] - 300_000.0) < 0.01           # remainder
    assert abs(q['a'] - 175_000.0) < 0.01            # 700k x 100/400
    assert abs(q['b'] - 525_000.0) < 0.01
    report = c.reconciliation_report(q, target=1_000_000.0, strict=True)
    assert report['reconciles'].all()


# ----------------------------------------------------------------------
# 5. Jagged-hierarchy leaf pin honored (leaf sibling of a manager)
# ----------------------------------------------------------------------
def test_jagged_leaf_pin_honored():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: pin on a leaf whose sibling is a manager (was silently "
          "ignored)")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T1', rep='a', kw=100),
            dict(pool='P', team='direct_ic', rep=None, kw=300)]
    c = QuotaCascader(_build(rows))
    q = c.cascade_quota('P', 1_000_000.0, metrics=M,
                        new_ic_overrides={'direct_ic': 100_000.0},
                        verbose=False)
    print(f"  direct_ic: ${q['direct_ic']:,.2f} (pinned) · "
          f"T1: ${q['T1']:,.2f} · a: ${q['a']:,.2f}")
    assert q['direct_ic'] == 100_000.0               # was 750,000 (ignored)
    assert abs(q['T1'] - 900_000.0) < 0.01
    assert abs(q['a'] - 900_000.0) < 0.01


# ----------------------------------------------------------------------
# 6. Issue #23 — override_basis semantics under HedgeByDepth
# ----------------------------------------------------------------------
def test_override_basis():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: override_basis 'base' (default) vs 'cascaded' with "
          "HedgeByDepth {1:1.10, 2:1.05}")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T', rep=f'r{i}', kw=100) for i in range(3)]
    hedge = HedgeByDepth(from_leaves={1: 1.10, 2: 1.05})
    factor = 1.05 * 1.10                             # rep-level compound

    # default 'base': pin is the plan number; hedged derived
    c1 = QuotaCascader(_build(rows))
    q1 = c1.cascade_quota('P', 900_000.0, hedge_multiplier=hedge, metrics=M,
                          new_ic_overrides={'r0': 300_000.0}, verbose=False)
    assert c1.base_quotas['r0'] == 300_000.0
    assert abs(q1['r0'] - 300_000.0 * factor) < 0.01
    # pinned rep now hedges like everyone else
    assert abs(q1['r0'] - q1['r1']) < 0.01           # same base -> same hedged
    rep1 = c1.reconciliation_report(c1.base_quotas, target=900_000.0,
                                    strict=True)
    assert rep1['reconciles'].all()
    print(f"  base:     r0 base ${c1.base_quotas['r0']:,.2f} -> hedged "
          f"${q1['r0']:,.2f} (= pin x {factor:.4f})")

    # 'cascaded': pin is the exact final number; base derived
    c2 = QuotaCascader(_build(rows))
    q2 = c2.cascade_quota('P', 900_000.0, hedge_multiplier=hedge, metrics=M,
                          new_ic_overrides={'r0': 300_000.0},
                          override_basis='cascaded', verbose=False)
    assert q2['r0'] == 300_000.0
    assert abs(c2.base_quotas['r0'] - 300_000.0 / factor) < 0.01
    rep2 = c2.reconciliation_report(c2.base_quotas, target=900_000.0,
                                    strict=True)
    assert rep2['reconciles'].all()
    print(f"  cascaded: r0 hedged ${q2['r0']:,.2f} -> base "
          f"${c2.base_quotas['r0']:,.2f} (= pin / {factor:.4f})")

    # no hedge -> both bases identical (backward compat)
    c3 = QuotaCascader(_build(rows))
    q3 = c3.cascade_quota('P', 900_000.0, metrics=M,
                          new_ic_overrides={'r0': 300_000.0}, verbose=False)
    assert q3['r0'] == 300_000.0 == c3.base_quotas['r0']

    # invalid basis rejected
    try:
        c3.cascade_quota('P', 900_000.0, metrics=M, override_basis='plan')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'override_basis' in str(e)


# ----------------------------------------------------------------------
# 7. Issue #21 — hedge_ratios() / rehedge() after editing the base
# ----------------------------------------------------------------------
def test_rehedge_after_base_edit():
    print(f"\n\n{SEPARATOR}")
    print("TEST 7: edit base, roll up base, rehedge() — depth 0/1 stay "
          "un-hedged (the #21 repro, fixed)")
    print(SEPARATOR)
    rows = [dict(pool='P', team='T1', rep='a', kw=100),
            dict(pool='P', team='T1', rep='b', kw=100),
            dict(pool='P', team='T2', rep='c', kw=200)]
    c = QuotaCascader(_build(rows))
    q = c.cascade_quota('P', 1_000_000.0,
                        hedge_multiplier=HedgeByDepth(from_leaves={1: 1.10,
                                                                   2: 1.05}),
                        metrics=M, verbose=False)
    base = dict(c.base_quotas)

    # Post-cascade edit ON THE BASE: move $50k from 'b' to 'a', then roll
    # parents up as sums of children BASE values.
    base['a'] += 50_000.0
    base['b'] -= 50_000.0
    base['T1'] = base['a'] + base['b']
    base['T2'] = base['c']
    base['P'] = base['T1'] + base['T2']

    hedged = c.rehedge(base)
    ratios = c.hedge_ratios()
    # Root/team ratios preserved: root un-hedged, teams x1.05
    assert abs(hedged['P'] - 1_000_000.0) < 0.01          # NOT x1.155
    assert abs(hedged['T1'] - base['T1'] * 1.05) < 0.01
    assert abs(hedged['a'] - base['a'] * 1.05 * 1.10) < 0.01
    # The #21 failure mode: summing HEDGED leaves into the root
    polluted = hedged['a'] + hedged['b'] + hedged['c']
    print(f"  rehedged root: ${hedged['P']:,.2f} (correct) vs hedged-leaf "
          f"rollup: ${polluted:,.2f} (the bug this prevents)")
    assert polluted > 1_100_000.0                        # visibly wrong
    # ratios dict sanity
    assert abs(ratios['a'] - 1.05 * 1.10) < 1e-9 and ratios['P'] == 1.0

    # requires a prior cascade
    try:
        QuotaCascader(_build(rows)).rehedge(base)
        raise AssertionError('expected RuntimeError')
    except RuntimeError:
        print("  rehedge() before any cascade raises RuntimeError")


if __name__ == '__main__':
    test_issue28_repro_conserves()
    test_pin_with_all_brand_new_conserves()
    test_overpin_never_negative()
    test_manager_pin_honored()
    test_jagged_leaf_pin_honored()
    test_override_basis()
    test_rehedge_after_base_edit()

    print(f"\n\n{SEPARATOR}")
    print("ALL OVERRIDE-HARDENING TESTS PASSED")
    print(SEPARATOR)
