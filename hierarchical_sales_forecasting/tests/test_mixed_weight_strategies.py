"""
Tests for issue #35 — mixed weight strategies in cascade_many: a
callable metrics= policy honored verbatim per combination, with None
falling through to suggest_config or the legacy path.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import MetricSpec, cascade_many

SEPARATOR = "=" * 90
TAXONOMY = ['st_regional', 'team', 'rep']
GROUP_KEYS = ['st1_sales_type', 'st_regional']
DC_ONLY = [MetricSpec('dc_seats', direction='proportional', weight=1.0,
                      columns=['dc_seats'])]
POLICY = lambda g: (DC_ONLY if g['st1_sales_type'] == 'Migration' else None)
SUGGEST = dict(
    target_column='kw',
    candidate_metrics=[{'name': 'cloud', 'column': 'cloud',
                        'direction': 'proportional', 'columns': ['cloud']}],
    warn_on_direction_mismatch=False,
)


def _hdf():
    """Two sales types; kw/cloud correlated so suggestion yields weight>0."""
    rows = []
    for st in ['Migration', 'Expansion']:
        reg = f'{st}_R'
        for i, (kw, cloud, dc) in enumerate(
                [(100, 110, 900), (200, 190, 50), (300, 310, 40),
                 (400, 390, 10)]):
            rows.append(dict(st1_sales_type=st, st_regional=reg,
                             team=f'{reg}_T{i//2+1}', rep=f'{reg}_r{i+1}',
                             kw=kw, cloud=cloud, dc_seats=dc))
    return pd.DataFrame(rows)


def _targets():
    return pd.DataFrame([
        dict(st1_sales_type='Migration', st_regional='Migration_R',
             q=1_000_000.0),
        dict(st1_sales_type='Expansion', st_regional='Expansion_R',
             q=1_000_000.0),
    ])


def _run(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cascade_many(_hdf(), _targets(), group_keys=GROUP_KEYS,
                            target_col='q', taxonomy=TAXONOMY, **kw)


# ----------------------------------------------------------------------
# 1. Mixed: Migration fixed DC-only, Expansion suggested
# ----------------------------------------------------------------------
def test_mixed_fixed_and_suggested():
    print(SEPARATOR)
    print("TEST 1: Migration -> fixed pure-DC slate (verbatim); "
          "Expansion -> suggested weights")
    print(SEPARATOR)
    quotas, weights = _run(metrics=POLICY, suggest_config=SUGGEST,
                           weights_mode='per_group')
    mig = quotas[quotas.st1_sales_type == 'Migration'].set_index('node_id')
    # Pure DC-seat share via team rollups:
    # r1 = 1M x (950/1000) x (900/950) = 900k ; r4 = 1M x (50/1000) x (10/50)
    assert abs(mig.loc['Migration_R_r1', 'base_quota'] - 900_000.0) < 0.05
    assert abs(mig.loc['Migration_R_r4', 'base_quota'] - 10_000.0) < 0.05
    # Fixed slate recorded verbatim in weights_long for Migration
    w_mig = weights[(weights.st1_sales_type == 'Migration')]
    assert list(w_mig['metric']) == ['dc_seats']
    assert (w_mig['input_weight'] == 1.0).all()
    # Expansion used the SUGGESTED metric (cloud), not dc
    w_exp = weights[(weights.st1_sales_type == 'Expansion')]
    assert list(w_exp['metric']) == ['cloud']
    assert (w_exp['input_weight'] > 0.9).all()   # near-perfect correlation
    exp = quotas[quotas.st1_sales_type == 'Expansion'].set_index('node_id')
    assert abs(exp.loc['Expansion_R_r1', 'base_quota']
               - 1_000_000.0 * 110 / 1000) < 1.0
    print(f"  Migration r1: ${mig.loc['Migration_R_r1', 'base_quota']:,.2f} "
          f"(pure DC) · Expansion by suggested cloud "
          f"(w={w_exp['input_weight'].iloc[0]:.3f})")


# ----------------------------------------------------------------------
# 2. Exact equivalence with the two-call workaround
# ----------------------------------------------------------------------
def test_equivalence_with_two_calls():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: mixed policy == split-and-concat two-call workaround")
    print(SEPARATOR)
    unified, _ = _run(metrics=POLICY, suggest_config=SUGGEST,
                      weights_mode='per_group')
    hdf, tdf = _hdf(), _targets()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mig, _ = cascade_many(hdf, tdf[tdf.st1_sales_type == 'Migration'],
                              group_keys=GROUP_KEYS, target_col='q',
                              taxonomy=TAXONOMY, metrics=DC_ONLY)
        exp, _ = cascade_many(hdf, tdf[tdf.st1_sales_type == 'Expansion'],
                              group_keys=GROUP_KEYS, target_col='q',
                              taxonomy=TAXONOMY, suggest_config=SUGGEST,
                              weights_mode='per_group')
    def norm(df):
        return (df[['st1_sales_type', 'node_id', 'base_quota',
                    'cascaded_quota']]
                .sort_values(['st1_sales_type', 'node_id'])
                .reset_index(drop=True))
    pd.testing.assert_frame_equal(
        norm(unified), norm(pd.concat([mig, exp], ignore_index=True)))
    print("  identical output frames")


# ----------------------------------------------------------------------
# 3. Callable + no suggest_config: None -> legacy path
# ----------------------------------------------------------------------
def test_none_falls_to_legacy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: policy None without suggest_config -> legacy "
          "'_Attainment' path for that combination")
    print(SEPARATOR)
    hdf = _hdf()
    hdf['Q1_Attainment'] = [10, 20, 30, 40] * 2      # legacy capacity
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        quotas, weights = cascade_many(
            hdf, _targets(), group_keys=GROUP_KEYS, target_col='q',
            taxonomy=TAXONOMY, metrics=POLICY)
    exp = quotas[quotas.st1_sales_type == 'Expansion'].set_index('node_id')
    # Legacy capacity split: r1 = 10/100
    assert abs(exp.loc['Expansion_R_r1', 'base_quota'] - 100_000.0) < 0.05
    # v0.28.0 (#50): legacy combos are no longer silently absent from
    # weights_long — they get an explicit '_Attainment' provenance row.
    leg = weights[weights['st1_sales_type'] == 'Expansion']
    assert len(leg[leg.role == 'blend']) == 1
    assert leg.iloc[0]['metric'] == '_Attainment'
    assert leg.iloc[0]['weights_source'] == 'default_attainment'
    print(f"  Expansion r1: ${exp.loc['Expansion_R_r1', 'base_quota']:,.2f} "
          f"(legacy attainment split)")


# ----------------------------------------------------------------------
# 4. Validation: static+suggest exclusive; bad returns via on_error
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: static list + suggest_config still rejected; bad "
          "callable returns follow on_error")
    print(SEPARATOR)
    try:
        _run(metrics=DC_ONLY, suggest_config=SUGGEST)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'not both' in str(e)
        print(f"  static+suggest rejected: {str(e)[:60]}...")
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        quotas, _, dropped = cascade_many(
            _hdf(), _targets(), group_keys=GROUP_KEYS, target_col='q',
            taxonomy=TAXONOMY,
            metrics=lambda g: ('oops'
                               if g['st1_sales_type'] == 'Migration'
                               else DC_ONLY),
            return_dropped=True)
    assert len(dropped) == 1
    assert 'must return a list of MetricSpec' in dropped['reason'].iloc[0]
    assert set(quotas['st1_sales_type']) == {'Expansion'}
    print("  bad return dropped with reason; other combos proceed")


# ----------------------------------------------------------------------
# 5. Backward compat: static list identical to pre-0.19.0
# ----------------------------------------------------------------------
def test_static_list_unchanged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: static metrics list — behavior unchanged")
    print(SEPARATOR)
    q1, _ = _run(metrics=DC_ONLY)
    q2, _ = _run(metrics=lambda g: DC_ONLY)   # always-slate callable
    def norm(df):
        return (df[['st1_sales_type', 'node_id', 'base_quota']]
                .sort_values(['st1_sales_type', 'node_id'])
                .reset_index(drop=True))
    pd.testing.assert_frame_equal(norm(q1), norm(q2))
    print("  static list == always-slate callable")


if __name__ == '__main__':
    test_mixed_fixed_and_suggested()
    test_equivalence_with_two_calls()
    test_none_falls_to_legacy()
    test_validation()
    test_static_list_unchanged()

    print(f"\n\n{SEPARATOR}")
    print("ALL MIXED-WEIGHT-STRATEGY TESTS PASSED")
    print(SEPARATOR)
