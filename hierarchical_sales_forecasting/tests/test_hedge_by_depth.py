"""
Tests for issue #13 — HedgeByDepth: per-depth hedge for cascade_quota
and (critically) cascade_many, where per-node dicts are impossible.

Covers:
  - the issue's exact policy {1: 1.10, 2: 1.05} from leaves (pinned)
  - resolve(): correct multipliers on a uniform 4-level tree
  - jagged hierarchy: from_leaves != from_root, both bases correct
  - both bases together compose multiplicatively
  - backward compat: float / per-node dict byte-for-byte unchanged
  - cascade_many end-to-end: resolved per combination, base reconciles
  - validation errors
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    HedgeByDepth,
    cascade_many,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'sub_region', 'team', 'territory']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]
POLICY = HedgeByDepth(from_leaves={1: 1.10, 2: 1.05})   # the issue's table


def _uniform_df():
    """regional -> 2 sub_regions -> 2 teams each -> 2 territories each."""
    rows = []
    for s in ['W', 'E']:
        for t in [1, 2]:
            for r in [1, 2]:
                rows.append(dict(regional='AMER', sub_region=f'S{s}',
                                 team=f'T{s}{t}', territory=f'r{s}{t}{r}',
                                 kw=100))
    return pd.DataFrame(rows)


def _build(df):
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY, metrics_cols=['kw'])
    return h


# ----------------------------------------------------------------------
# 1. The issue's exact policy on a uniform tree (pinned worked example)
# ----------------------------------------------------------------------
def test_issue13_policy_uniform_tree():
    print(SEPARATOR)
    print("TEST 1: {1: 1.10, 2: 1.05} from leaves — teams 1.10, sub-regions "
          "1.05, root 1.00")
    print(SEPARATOR)
    h = _build(_uniform_df())
    resolved = POLICY.resolve(h.graph)
    assert resolved['TW1'] == 1.10          # deepest managers (children=ICs)
    assert resolved['SW'] == 1.05           # second-deepest
    assert resolved['AMER'] == 1.0          # default
    assert resolved['rW11'] == 1.0          # leaves never hedged

    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 800_000.0, hedge_multiplier=POLICY,
                        metrics=KW, verbose=False)
    base = c.base_quotas
    # Uniform kw -> each territory's base is 100k; hedged = base x 1.05 x 1.10
    # (sub-region hedge applies distributing to teams, team hedge to ICs;
    #  root hedge = 1.0)
    assert abs(base['rW11'] - 100_000.0) < 0.01
    expected = 100_000.0 * 1.05 * 1.10
    print(f"  territory base ${base['rW11']:,.2f} -> hedged "
          f"${q['rW11']:,.2f} (expected ${expected:,.2f})")
    assert abs(q['rW11'] - expected) < 0.01
    # Base layer reconciles at every depth regardless of hedge shape
    assert c.reconciliation_report(base, target=800_000.0,
                                   strict=True)['reconciles'].all()


# ----------------------------------------------------------------------
# 2. Jagged hierarchy — from_leaves and from_root genuinely differ
# ----------------------------------------------------------------------
def test_jagged_bases_differ():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: jagged tree — a root-depth-1 node can be a DEEPEST "
          "manager; bases disagree and both resolve correctly")
    print(SEPARATOR)
    df = pd.DataFrame([
        # Full-depth branch: AMER -> SW -> TW1 -> rW11
        dict(regional='AMER', sub_region='SW', team='TW1',
             territory='rW11', kw=100),
        # Jagged branch: AMER -> SJ -> rJ1 (no team level)
        dict(regional='AMER', sub_region='SJ', team=None,
             territory='rJ1', kw=100),
    ])
    h = _build(df)
    res_leaves = HedgeByDepth(from_leaves={1: 1.10}).resolve(h.graph)
    res_root = HedgeByDepth(from_root={2: 1.10}).resolve(h.graph)
    # SJ is root-depth 1 but a DEEPEST manager (its child is an IC)
    assert res_leaves['SJ'] == 1.10 and res_leaves['SW'] == 1.0
    assert res_leaves['TW1'] == 1.10
    # From-root: depth 2 hits TW1 only; SJ (depth 1) is untouched
    assert res_root['TW1'] == 1.10 and res_root['SJ'] == 1.0
    print("  from_leaves: SJ=1.10, TW1=1.10, SW=1.00 · "
          "from_root(d=2): TW1=1.10, SJ=1.00")


# ----------------------------------------------------------------------
# 3. Both bases compose multiplicatively
# ----------------------------------------------------------------------
def test_bases_compose():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: node matched by BOTH bases gets the product")
    print(SEPARATOR)
    h = _build(_uniform_df())
    spec = HedgeByDepth(from_leaves={1: 1.10}, from_root={2: 1.02},
                        default=1.0)
    resolved = spec.resolve(h.graph)
    # Teams are root-depth 2 AND leaf-distance 1 -> 1.10 * 1.02
    assert abs(resolved['TW1'] - 1.10 * 1.02) < 1e-12
    assert resolved['SW'] == 1.0
    print(f"  TW1 = 1.10 x 1.02 = {resolved['TW1']:.4f}")


# ----------------------------------------------------------------------
# 4. Backward compat — float and per-node dict unchanged
# ----------------------------------------------------------------------
def test_backward_compat_exact():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: HedgeByDepth-resolved dict == hand-built per-node dict; "
          "float path untouched")
    print(SEPARATOR)
    df = _uniform_df()
    h1, h2 = _build(df), _build(df)
    c1, c2 = QuotaCascader(h1), QuotaCascader(h2)
    q_spec = c1.cascade_quota('AMER', 800_000.0, hedge_multiplier=POLICY,
                              metrics=KW, verbose=False)
    hand_dict = POLICY.resolve(h2.graph)          # what a consumer would build
    q_dict = c2.cascade_quota('AMER', 800_000.0, hedge_multiplier=hand_dict,
                              metrics=KW, verbose=False)
    assert q_spec == q_dict
    # Float path still exact
    h3 = _build(df)
    q_float = QuotaCascader(h3).cascade_quota('AMER', 800_000.0,
                                              hedge_multiplier=1.05,
                                              metrics=KW, verbose=False)
    assert abs(q_float['rW11'] - 100_000.0 * 1.05 ** 3) < 0.01
    print("  spec-resolved == hand-built dict · float compounds 1.05^3")


# ----------------------------------------------------------------------
# 5. cascade_many — resolved per combination, base reconciles
# ----------------------------------------------------------------------
def test_cascade_many_hedge_by_depth():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: cascade_many(hedge_multiplier=HedgeByDepth(...)) — the "
          "issue's blocked use case")
    print(SEPARATOR)
    hdf = _uniform_df()
    hdf2 = hdf.copy(); hdf2['regional'] = 'EMEA'
    for c_ in ['sub_region', 'team', 'territory']:
        hdf2[c_] = hdf2[c_] + '_e'
    hdf_all = pd.concat([hdf, hdf2], ignore_index=True)
    targets = pd.DataFrame([
        dict(regional='AMER', fiscal_quarter=1, q_target=800_000.0),
        dict(regional='EMEA', fiscal_quarter=1, q_target=400_000.0),
    ])
    quotas, _ = cascade_many(
        hdf_all, targets, group_keys=['regional'], target_col='q_target',
        taxonomy=TAXONOMY, metrics=KW, hedge_multiplier=POLICY,
    )
    for regional, target, leaf in [('AMER', 800_000.0, 'rW11'),
                                   ('EMEA', 400_000.0, 'rW11_e')]:
        combo = quotas[quotas.regional == regional]
        row = combo[combo.node_id == leaf].iloc[0]
        expected = (target / 8) * 1.05 * 1.10
        assert abs(row['cascaded_quota'] - expected) < 0.01, regional
        per_depth = combo.groupby('depth')['base_quota'].sum()
        assert (abs(per_depth - target) < 0.05).all(), regional
        print(f"  {regional}: leaf hedged ${row['cascaded_quota']:,.2f} "
              f"(= base x 1.05 x 1.10); base reconciles at all depths")


# ----------------------------------------------------------------------
# 6. Validation
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: constructor validation")
    print(SEPARATOR)
    for bad in [dict(),                                   # no mapping at all
                dict(from_leaves={'1': 1.1}),             # str key
                dict(from_leaves={1: 0.0}),               # non-positive mult
                dict(from_leaves={1: 1.1}, default=-1)]:  # bad default
        try:
            HedgeByDepth(**bad)
            raise AssertionError(f'expected ValueError for {bad}')
        except ValueError as e:
            print(f"  {bad} -> {str(e)[:60]}...")


if __name__ == '__main__':
    test_issue13_policy_uniform_tree()
    test_jagged_bases_differ()
    test_bases_compose()
    test_backward_compat_exact()
    test_cascade_many_hedge_by_depth()
    test_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL HEDGE-BY-DEPTH TESTS PASSED")
    print(SEPARATOR)
