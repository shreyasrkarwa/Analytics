"""
Tests for issue #46 — reconcile(): one-call validation of per-parent
conservation and per-depth hedge identities on cascade outputs.

Covers:
  - fresh HedgeByDepth cascade (the issue's exact 1.10/1.05 spec):
    every check ok — cross-validates the frame-side expectation
    against the engine's own resolution
  - flat float hedge (f**depth) and explicit {depth: cum} dict
  - corrupted base -> conservation row flagged + ONE summary warning
  - wrong hedge expectation -> hedge_ratio rows flagged
  - post-apply_pins frames still reconcile (the #21 contract)
  - multi-quarter isolation; gated (base=0) rows skipped for ratios
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, HedgeByDepth, Pin, apply_pins, reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
HEDGE = HedgeByDepth(from_leaves={1: 1.10, 2: 1.05})


def _quotas(hedge=HEDGE, **kw):
    hdf = pd.DataFrame([
        dict(region='EMEA', team=f'T{i//2+1}', rep=f'r{i+1}',
             kw=[100, 200, 300, 400][i], seats=[10, 20, 30, 40][i])
        for i in range(4)])
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=KW, hedge_multiplier=hedge, **kw)
    return q


# ----------------------------------------------------------------------
# 1. Fresh HedgeByDepth run: everything ok (the engine anchor)
# ----------------------------------------------------------------------
def test_hedge_by_depth_clean():
    print(SEPARATOR)
    print("TEST 1: issue's exact spec — d1=1.05 cum, d2=1.155 cum; all "
          "checks ok, silent")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = reconcile(q, hedge=HEDGE)
    assert not [x for x in w if 'reconcile' in str(x.message)]
    assert f['ok'].all()
    hx = f[f.check == 'hedge_ratio'].set_index(['node_id',
                                                'fiscal_quarter'])
    print(f[f.fiscal_quarter == 1][['node_id', 'check', 'expected',
                                    'actual', 'ok']].to_string(index=False))
    assert abs(hx.loc[('T1', 1), 'expected'] - 1.05) < 1e-9
    assert abs(hx.loc[('r1', 1), 'expected'] - 1.155) < 1e-9
    assert (f[f.check == 'conservation']['node_id']
            .isin(['EMEA', 'T1', 'T2']).all())
    assert len(f[f.check == 'conservation']) == 6      # 3 parents x 2 qtrs


# ----------------------------------------------------------------------
# 2. Float and explicit-dict hedge forms
# ----------------------------------------------------------------------
def test_float_and_dict_forms():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: flat float (f**depth) and {depth: cum} dict")
    print(SEPARATOR)
    q = _quotas(hedge=1.1)
    f1 = reconcile(q, hedge=1.1)
    assert f1['ok'].all()
    hx = f1[f1.check == 'hedge_ratio'].set_index(['node_id',
                                                  'fiscal_quarter'])
    assert abs(hx.loc[('r1', 1), 'expected'] - 1.1 ** 2) < 1e-9
    # the literal hand-written identity list
    f2 = reconcile(q, hedge={0: 1.0, 1: 1.1, 2: 1.21})
    assert f2['ok'].all()
    print("  f**depth and explicit cum dict both clean")


# ----------------------------------------------------------------------
# 3. Corruption is caught: conservation + wrong-hedge flags
# ----------------------------------------------------------------------
def test_violations_flagged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: corrupted base -> conservation flagged; wrong hedge "
          "dict -> ratio rows flagged")
    print(SEPARATOR)
    q = _quotas()
    q2 = q.copy()
    q2.loc[(q2.node_id == 'r1') & (q2.fiscal_quarter == 1),
           'base_quota'] += 50_000.0                 # break T1's sum
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = reconcile(q2, hedge=HEDGE)
    assert len([x for x in w if 'reconcile' in str(x.message)]) == 1
    bad = f[~f.ok]
    cons_bad = bad[bad.check == 'conservation']
    assert len(cons_bad) == 1
    assert cons_bad.iloc[0]['node_id'] == 'T1'
    assert cons_bad.iloc[0]['fiscal_quarter'] == 1     # Q2 untouched
    assert abs(cons_bad.iloc[0]['delta'] - 50_000.0) < 0.05
    # r1's ratio also broke (base moved, cascaded didn't)
    assert 'r1' in set(bad[bad.check == 'hedge_ratio']['node_id'])
    # wrong expectation flags every hedged row
    f3 = reconcile(q, hedge={1: 1.5, 2: 2.0})
    assert not f3[(f3.check == 'hedge_ratio')
                  & (f3.depth > 0)]['ok'].any()
    print(f"  T1/Q1 delta=+50,000 caught; wrong dict flags all ratios")


# ----------------------------------------------------------------------
# 4. Post-apply_pins frames reconcile (the #21 contract, verified)
# ----------------------------------------------------------------------
def test_post_pins_clean():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: after apply_pins, conservation AND hedge identities "
          "still hold")
    print(SEPARATOR)
    q = _quotas()
    e, rep = apply_pins(q, [Pin('T1', 900_000.0),
                            Pin('r3', 500_000.0)])
    assert rep['feasible'].all()
    f = reconcile(e, hedge=HEDGE)
    assert f['ok'].all()
    print(f"  {len(f)} checks, all ok — pins preserved ratios and "
          f"conserved parents")


# ----------------------------------------------------------------------
# 5. Gated rows skipped for ratios; conservation still checked
# ----------------------------------------------------------------------
def test_gated_zero_base_skipped():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: gated (base=0) nodes have no ratio row; "
          "conservation intact")
    print(SEPARATOR)
    q = _quotas(gate_metrics=[MetricSpec('seats', columns=['seats'],
                                         gate_threshold=15.0)])
    f = reconcile(q, hedge=HEDGE)
    assert f['ok'].all()
    hr = f[f.check == 'hedge_ratio']
    assert 'r1' not in set(hr['node_id'])              # gated, base 0
    assert 'r2' in set(hr['node_id'])
    print("  r1 (gated) skipped for ratios; everything else clean")


if __name__ == '__main__':
    test_hedge_by_depth_clean()
    test_float_and_dict_forms()
    test_violations_flagged()
    test_post_pins_clean()
    test_gated_zero_base_skipped()

    print(f"\n\n{SEPARATOR}")
    print("ALL RECONCILE TESTS PASSED")
    print(SEPARATOR)
