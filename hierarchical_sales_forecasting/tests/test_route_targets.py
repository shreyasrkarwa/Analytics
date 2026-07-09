"""
Tests for the v0.14.0 routing release — issues #26, #25 (and #32 via
composition).

  #26: cascade_many exposes dropped target rows as DATA (attrs + opt-in
       third return value), with reasons and full original columns.
  #25: route_targets() carries dropped/matched targets on named
       recipients in another tree — base-layer split, ratio-derived
       hedged values, ancestor rollups, original tags preserved.
  #32: the Government+EMEA scenario end-to-end via #26 -> #25.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    MetricSpec,
    HedgeByDepth,
    cascade_many,
    route_targets,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]


def _hierarchy_df():
    """Enterprise_EMEA + Enterprise_AMER; NO Government branch at all."""
    rows = []
    for reg in ['Enterprise_EMEA', 'Enterprise_AMER']:
        for t, reps in [('T1', [100, 300]), ('T2', [200, 400])]:
            for i, kw in enumerate(reps):
                rows.append(dict(regional=reg, team=f'{reg}_{t}',
                                 rep=f'{reg}_{t}_r{i+1}', kw=kw))
    return pd.DataFrame(rows)


def _target_df():
    return pd.DataFrame([
        dict(regional='Enterprise_EMEA', segment='Enterprise',
             fiscal_quarter=1, q_target=1_000_000.0),
        dict(regional='Enterprise_AMER', segment='Enterprise',
             fiscal_quarter=1, q_target=2_000_000.0),
        # No Government branch exists -> this row gets dropped
        dict(regional='Government_EMEA', segment='Government',
             fiscal_quarter=1, q_target=300_000.0),
    ])


def _run(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cascade_many(
            _hierarchy_df(), _target_df(), group_keys=['regional'],
            target_col='q_target', taxonomy=TAXONOMY, metrics=KW,
            hedge_multiplier=HedgeByDepth(from_leaves={1: 1.10, 2: 1.05}),
            **kw)


# ----------------------------------------------------------------------
# 1. Issue #26 — dropped targets are data (attrs + third return value)
# ----------------------------------------------------------------------
def test_dropped_targets_frame():
    print(SEPARATOR)
    print("TEST 1: #26 — dropped rows returned with reason + full columns")
    print(SEPARATOR)
    quotas, weights, dropped = _run(return_dropped=True)
    print(f"  dropped rows: {len(dropped)} · reason: "
          f"{dropped['reason'].iloc[0][:60]}...")
    assert len(dropped) == 1
    assert dropped['regional'].iloc[0] == 'Government_EMEA'
    assert dropped['segment'].iloc[0] == 'Government'      # tags intact
    assert dropped['q_target'].iloc[0] == 300_000.0        # money visible
    assert 'no rows' in dropped['reason'].iloc[0]
    # attrs channel works without the flag too
    q2, _ = _run()
    assert len(q2.attrs['dropped_targets']) == 1
    # nothing dropped -> EMPTY frame with the right schema, not missing
    ok_targets = _target_df().iloc[:2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q3, _, d3 = cascade_many(
            _hierarchy_df(), ok_targets, group_keys=['regional'],
            target_col='q_target', taxonomy=TAXONOMY, metrics=KW,
            return_dropped=True)
    assert len(d3) == 0 and 'reason' in d3.columns


# ----------------------------------------------------------------------
# 2. Issues #25/#32 — the Government/EMEA scenario end-to-end
# ----------------------------------------------------------------------
def test_government_routing_end_to_end():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: #25/#32 — route dropped Government money onto named "
          "Enterprise_EMEA reps, split by base_quota")
    print(SEPARATOR)
    quotas, _, dropped = _run(return_dropped=True)
    recipients = ['Enterprise_EMEA_T1_r2', 'Enterprise_EMEA_T2_r1',
                  'Enterprise_EMEA_T2_r2']
    routed = route_targets(
        dropped, quotas, recipients=recipients, target_col='q_target',
        recipient_keys={'regional': 'Enterprise_EMEA'},
    )
    leaf = routed[routed.node_id.isin(recipients)]
    # Base split proportional to existing base_quota (kw 300/200/400)
    assert abs(leaf['base_quota'].sum() - 300_000.0) < 0.05
    r2 = leaf[leaf.node_id == 'Enterprise_EMEA_T1_r2'].iloc[0]
    assert abs(r2['base_quota'] - 300_000.0 * 300 / 900) < 1.0
    # Hedged derived from each recipient's OWN ratio (1.05 x 1.10)
    assert abs(r2['cascaded_quota'] - r2['base_quota'] * 1.155) < 0.5
    # Original tags survive on every routed row
    assert (routed['segment'] == 'Government').all()
    assert (routed['regional'] == 'Government_EMEA').all()
    assert routed['routed'].all()
    # Rollup: per-depth base sums equal the routed amount
    per_depth = routed.groupby('depth')['base_quota'].sum()
    print(f"  routed per-depth base sums:\n{per_depth.to_string()}")
    assert (abs(per_depth - 300_000.0) < 0.05).all()
    # Ancestor hedged values use the ANCESTOR's ratio (teams x1.05, root x1)
    root = routed[routed.node_id == 'Enterprise_EMEA'].iloc[0]
    assert abs(root['cascaded_quota'] - root['base_quota']) < 0.5
    t2 = routed[routed.node_id == 'Enterprise_EMEA_T2'].iloc[0]
    assert abs(t2['cascaded_quota'] - t2['base_quota'] * 1.05) < 0.5
    # Additive by construction
    combined = pd.concat([quotas, routed], ignore_index=True)
    carried = combined[combined.node_id == 'Enterprise_EMEA_T1_r2']
    print(f"  T1_r2 now carries {len(carried)} rows "
          f"(normal + routed Government)")
    assert len(carried) == 2


# ----------------------------------------------------------------------
# 3. split='equal' and zero-baseline fallback
# ----------------------------------------------------------------------
def test_split_modes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: split='equal' + zero-sum split column falls back with "
          "a warning")
    print(SEPARATOR)
    quotas, _, dropped = _run(return_dropped=True)
    recipients = ['Enterprise_EMEA_T1_r1', 'Enterprise_EMEA_T1_r2']
    eq = route_targets(dropped, quotas, recipients=recipients,
                       target_col='q_target',
                       recipient_keys={'regional': 'Enterprise_EMEA'},
                       split='equal', rollup=False)
    assert (abs(eq['base_quota'] - 150_000.0) < 0.05).all()
    # zero-sum split column -> warn + equal
    q0 = quotas.copy()
    q0['zero_col'] = 0.0
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        z = route_targets(dropped, q0, recipients=recipients,
                          target_col='q_target',
                          recipient_keys={'regional': 'Enterprise_EMEA'},
                          split='zero_col', rollup=False)
    assert any('equal split' in str(w.message) for w in wlog)
    assert (abs(z['base_quota'] - 150_000.0) < 0.05).all()
    print("  equal split OK · zero-baseline warned and fell back")


# ----------------------------------------------------------------------
# 4. Exclusion recipe — two calls, different targets/recipients
# ----------------------------------------------------------------------
def test_exclusion_recipe():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: per-product exclusion via two route_targets calls")
    print(SEPARATOR)
    quotas, _, _ = _run(return_dropped=True)
    gov = pd.DataFrame([
        dict(product='Cloud', q_target=100_000.0),
        dict(product='DC',    q_target=200_000.0),
    ])
    all_reps = ['Enterprise_EMEA_T1_r1', 'Enterprise_EMEA_T1_r2',
                'Enterprise_EMEA_T2_r1']
    no_cloud_rep = [r for r in all_reps if r != 'Enterprise_EMEA_T1_r1']
    routed = pd.concat([
        route_targets(gov[gov['product'] == 'Cloud'], quotas,
                      recipients=no_cloud_rep, target_col='q_target',
                      recipient_keys={'regional': 'Enterprise_EMEA'},
                      rollup=False),
        route_targets(gov[gov['product'] == 'DC'], quotas,
                      recipients=all_reps, target_col='q_target',
                      recipient_keys={'regional': 'Enterprise_EMEA'},
                      rollup=False),
    ], ignore_index=True)
    cloud = routed[routed['product'] == 'Cloud']
    assert 'Enterprise_EMEA_T1_r1' not in set(cloud['node_id'])
    assert abs(routed['base_quota'].sum() - 300_000.0) < 0.1
    print(f"  Cloud excluded r1; total routed "
          f"${routed['base_quota'].sum():,.2f}")


# ----------------------------------------------------------------------
# 5. Validation: unknown recipients, ambiguity, bad columns
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: validation errors are clear")
    print(SEPARATOR)
    quotas, _, dropped = _run(return_dropped=True)
    def expect(err_frag, **kw):
        try:
            route_targets(dropped, quotas, target_col='q_target', **kw)
            raise AssertionError(f'expected ValueError ({err_frag})')
        except ValueError as e:
            assert err_frag in str(e), str(e)
            print(f"  {err_frag}: OK")
    expect('non-empty', recipients=[])
    expect('not found in quotas_long', recipients=['ghost_rep'],
           recipient_keys={'regional': 'Enterprise_EMEA'})
    expect('split column', recipients=['Enterprise_EMEA_T1_r1'],
           recipient_keys={'regional': 'Enterprise_EMEA'}, split='nope')
    # Two quarters in the frame -> ambiguous without recipient_keys
    targets2 = pd.concat([_target_df().iloc[:1]] * 1, ignore_index=True)
    two_q = pd.concat([_target_df().assign(fiscal_quarter=1),
                       _target_df().assign(fiscal_quarter=2)],
                      ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q2, _ = cascade_many(_hierarchy_df(), two_q, group_keys=['regional'],
                             target_col='q_target', taxonomy=TAXONOMY,
                             metrics=KW)
    try:
        route_targets(targets2, q2,
                      recipients=['Enterprise_EMEA_T1_r1'],
                      target_col='q_target',
                      recipient_keys={'regional': 'Enterprise_EMEA'})
        raise AssertionError('expected ambiguity ValueError')
    except ValueError as e:
        assert 'ambiguous' in str(e)
        print("  ambiguity without full recipient_keys: OK")
    # ...and resolvable by narrowing
    ok = route_targets(targets2, q2, recipients=['Enterprise_EMEA_T1_r1'],
                       target_col='q_target',
                       recipient_keys={'regional': 'Enterprise_EMEA',
                                       'fiscal_quarter': 1}, rollup=False)
    assert len(ok) == 1


if __name__ == '__main__':
    test_dropped_targets_frame()
    test_government_routing_end_to_end()
    test_split_modes()
    test_exclusion_recipe()
    test_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL ROUTE-TARGETS TESTS PASSED")
    print(SEPARATOR)
