"""
Tests for issue #14 — per-group / conditional gating in cascade_many.

Covers:
  - callable gate policy: Migration combos gated, others untouched
  - exact equivalence with the two-call split workaround
  - mapping-style policy via dict.get lambda
  - callable returning None everywhere == no gates
  - bad return type / raising callable -> on_error machinery
  - static list unchanged (backward compat)
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import MetricSpec, cascade_many

SEPARATOR = "=" * 90
TAXONOMY = ['sales_type_regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]
DC_GATE = [MetricSpec('dc_seats', columns=['dc_seats'])]
GATE_POLICY = lambda g: (DC_GATE if g['st1_sales_type'] == 'Migration'
                         else None)


def _hierarchy_df():
    """Identical structure per sales type; r2 has 0 dc_seats everywhere."""
    rows = []
    for st in ['Migration', 'Expansion']:
        reg = f'{st}_EMEA'
        for i, (kw, dc) in enumerate([(100, 10), (300, 0)]):
            rows.append(dict(st1_sales_type=st, sales_type_regional=reg,
                             team=f'{reg}_T1', rep=f'{reg}_r{i+1}',
                             kw=kw, dc_seats=dc))
    return pd.DataFrame(rows)


def _targets():
    return pd.DataFrame([
        dict(st1_sales_type='Migration', sales_type_regional='Migration_EMEA',
             q=1_000_000.0),
        dict(st1_sales_type='Expansion', sales_type_regional='Expansion_EMEA',
             q=1_000_000.0),
    ])


GROUP_KEYS = ['st1_sales_type', 'sales_type_regional']


def _run(gates, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cascade_many(
            _hierarchy_df(), _targets(), group_keys=GROUP_KEYS,
            target_col='q', taxonomy=TAXONOMY, metrics=KW,
            gate_metrics=gates, **kw)


# ----------------------------------------------------------------------
# 1. Callable policy: gate ONLY Migration
# ----------------------------------------------------------------------
def test_callable_gates_migration_only():
    print(SEPARATOR)
    print("TEST 1: callable policy — DC gate on Migration only")
    print(SEPARATOR)
    quotas, _ = _run(GATE_POLICY)
    mig = quotas[quotas.st1_sales_type == 'Migration'].set_index('node_id')
    exp = quotas[quotas.st1_sales_type == 'Expansion'].set_index('node_id')
    # Migration: r2 (0 dc) gated -> $0; r1 carries everything
    assert mig.loc['Migration_EMEA_r2', 'cascaded_quota'] == 0.0
    assert abs(mig.loc['Migration_EMEA_r1', 'cascaded_quota']
               - 1_000_000.0) < 0.01
    assert bool(mig.loc['Migration_EMEA_r2', 'is_gated'])
    # Expansion: SAME data, no gate -> kw split 1:3
    assert abs(exp.loc['Expansion_EMEA_r2', 'cascaded_quota']
               - 750_000.0) < 0.01
    assert 'is_gated' not in exp.columns or not exp['is_gated'].any()
    # Both combos conserve
    for st, grp in quotas.groupby('st1_sales_type'):
        per_depth = grp.groupby('depth')['base_quota'].sum()
        assert (abs(per_depth - 1_000_000.0) < 0.05).all(), st
    print(f"  Migration r2 gated $0, r1 $1M · Expansion split 250k/750k · "
          f"both conserve")


# ----------------------------------------------------------------------
# 2. Exact equivalence with the two-call workaround
# ----------------------------------------------------------------------
def test_equivalence_with_two_call_workaround():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: callable == the old split-and-concat workaround, "
          "number for number")
    print(SEPARATOR)
    unified, _ = _run(GATE_POLICY)
    hdf, tdf = _hierarchy_df(), _targets()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mig, _ = cascade_many(hdf, tdf[tdf.st1_sales_type == 'Migration'],
                              group_keys=GROUP_KEYS, target_col='q',
                              taxonomy=TAXONOMY, metrics=KW,
                              gate_metrics=DC_GATE)
        rest, _ = cascade_many(hdf, tdf[tdf.st1_sales_type == 'Expansion'],
                               group_keys=GROUP_KEYS, target_col='q',
                               taxonomy=TAXONOMY, metrics=KW,
                               gate_metrics=None)
    def normal(df):
        return (df[['st1_sales_type', 'node_id', 'cascaded_quota',
                    'base_quota']]
                .sort_values(['st1_sales_type', 'node_id'])
                .reset_index(drop=True))
    merged = normal(pd.concat([mig, rest], ignore_index=True))
    uni = normal(unified)
    pd.testing.assert_frame_equal(uni, merged)
    print("  identical output frames")


# ----------------------------------------------------------------------
# 3. Mapping-style policy + all-None policy
# ----------------------------------------------------------------------
def test_mapping_and_none_policies():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: dict.get mapping recipe · always-None == no gates")
    print(SEPARATOR)
    by_type = {'Migration': DC_GATE}
    q_map, _ = _run(lambda g: by_type.get(g['st1_sales_type']))
    q_lam, _ = _run(GATE_POLICY)
    pd.testing.assert_frame_equal(
        q_map.sort_values(['st1_sales_type', 'node_id']).reset_index(drop=True),
        q_lam.sort_values(['st1_sales_type', 'node_id']).reset_index(drop=True))
    q_none, _ = _run(lambda g: None)
    q_static_none, _ = _run(None)
    pd.testing.assert_frame_equal(
        q_none.sort_values(['st1_sales_type', 'node_id']).reset_index(drop=True),
        q_static_none.sort_values(['st1_sales_type', 'node_id']).reset_index(drop=True))
    print("  mapping recipe == lambda · always-None == gate_metrics=None")


# ----------------------------------------------------------------------
# 4. Bad returns and raising callables flow through on_error
# ----------------------------------------------------------------------
def test_callable_errors():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: bad return type / raising policy handled per on_error")
    print(SEPARATOR)
    # Bad return type -> skip mode drops the combo into the dropped frame
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        quotas, _, dropped = cascade_many(
            _hierarchy_df(), _targets(), group_keys=GROUP_KEYS,
            target_col='q', taxonomy=TAXONOMY, metrics=KW,
            gate_metrics=lambda g: ('not-a-list'
                                    if g['st1_sales_type'] == 'Migration'
                                    else None),
            return_dropped=True)
    assert len(dropped) == 1
    assert 'must return a list of MetricSpec' in dropped['reason'].iloc[0]
    assert set(quotas['st1_sales_type']) == {'Expansion'}
    assert any('skipped' in str(w.message) for w in wlog)
    print(f"  bad return: combo dropped with reason "
          f"({dropped['reason'].iloc[0][:55]}...)")
    # Raising policy + on_error='raise' -> propagates
    def boom(g):
        raise RuntimeError('policy exploded')
    try:
        _run(boom, on_error='raise')
        raise AssertionError('expected RuntimeError')
    except RuntimeError as e:
        assert 'policy exploded' in str(e)
        print("  raising policy propagates under on_error='raise'")


# ----------------------------------------------------------------------
# 5. Static list unchanged (backward compat)
# ----------------------------------------------------------------------
def test_static_list_unchanged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: plain list still gates every combination")
    print(SEPARATOR)
    quotas, _ = _run(DC_GATE)
    for st in ['Migration', 'Expansion']:
        grp = quotas[quotas.st1_sales_type == st].set_index('node_id')
        assert grp.loc[f'{st}_EMEA_r2', 'cascaded_quota'] == 0.0, st
    print("  r2 gated in BOTH combos, as before")


if __name__ == '__main__':
    test_callable_gates_migration_only()
    test_equivalence_with_two_call_workaround()
    test_mapping_and_none_policies()
    test_callable_errors()
    test_static_list_unchanged()

    print(f"\n\n{SEPARATOR}")
    print("ALL CONDITIONAL-GATES TESTS PASSED")
    print(SEPARATOR)
