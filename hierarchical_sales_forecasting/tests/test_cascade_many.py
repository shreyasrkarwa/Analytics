"""
Tests for issue #4 — cascade_many, the native batch / multi-combination
cascade API.

Covers:
  - multi-combination cascade with tidy long output + group-key tagging
  - per-depth reconciliation of the base layer for every combination
  - prepare-once/cascade-many: quarters reuse the prepared combination
  - weights_mode='global' vs 'per_group' (suggest_config path)
  - gates + fully-gated combination redistribute (v0.5.0 semantics inside)
  - on_error='skip' warns and continues; 'raise' fails fast
  - dirty data (duplicate levels) auto-healed per v0.6.0 inside the batch
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import MetricSpec, cascade_many

SEPARATOR = "=" * 90

TAXONOMY = ['regional', 'sub_region', 'team', 'territory']
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['knowledge_workers'])]


def _hierarchy_df():
    """Two products x two regionals, 4 territories each."""
    rows = []
    for product in ['Jira', 'Confluence']:
        for regional in ['Ent_AMER', 'Ent_EMEA']:
            for i, (sub, team) in enumerate(
                    [('W', 'W1'), ('W', 'W2'), ('E', 'E1'), ('E', 'E2')]):
                rows.append(dict(
                    base_product=product,
                    regional=regional,
                    sub_region=f"{regional}_{sub}",
                    team=f"{regional}_{team}",
                    territory=f"{regional}_{team}_{product[:1]}{i}",
                    knowledge_workers=100 + 50 * i,
                    dc_seats=(0 if (product == 'Jira' and regional == 'Ent_EMEA')
                              else 10 + i),
                ))
    return pd.DataFrame(rows)


def _target_df():
    rows = []
    for product in ['Jira', 'Confluence']:
        for regional in ['Ent_AMER', 'Ent_EMEA']:
            for quarter in [1, 2]:
                rows.append(dict(base_product=product, regional=regional,
                                 fiscal_quarter=quarter,
                                 nn_acv_target=1_000_000.0 * quarter))
    return pd.DataFrame(rows)


GROUP_KEYS = ['base_product', 'regional']


# ----------------------------------------------------------------------
# 1. End-to-end batch: every combination x quarter, tidy + reconciled
# ----------------------------------------------------------------------
def test_batch_end_to_end_reconciles():
    print(SEPARATOR)
    print("TEST 1: 4 combinations x 2 quarters -> tidy long df; base layer "
          "reconciles at every depth for every cascade")
    print(SEPARATOR)
    quotas, weights = cascade_many(
        _hierarchy_df(), _target_df(),
        group_keys=GROUP_KEYS, target_col='nn_acv_target',
        taxonomy=TAXONOMY, metrics=KW,
        hedge_multiplier=1.05,
    )
    # 8 cascades (4 combos x 2 quarters), 9 nodes each (1+2+4... wait: root,
    # 2 sub-regions, 4 teams? -> 1 root + 2 subs + 4 teams + 4 territories = 11)
    n_casc = quotas.groupby(GROUP_KEYS + ['fiscal_quarter']).ngroups
    print(f"  cascades: {n_casc} (expected 8) · rows: {len(quotas)}")
    assert n_casc == 8
    # Group keys, passthrough, and target present on every row
    for col in GROUP_KEYS + ['fiscal_quarter', 'nn_acv_target',
                             'base_quota', 'cascaded_quota', 'depth']:
        assert col in quotas.columns, f"missing column {col}"
    # Base layer reconciles per cascade at EVERY depth
    # Tolerance: the long df is rounded to cents per node, so allow a
    # cent of drift per node at a depth (in-package reconciliation_report
    # runs on unrounded values and is exact).
    for keys, grp in quotas.groupby(GROUP_KEYS + ['fiscal_quarter']):
        target = grp['nn_acv_target'].iloc[0]
        per_depth = grp.groupby('depth')['base_quota'].sum()
        assert (abs(per_depth - target) < 0.05).all(), \
            f"{keys}: {per_depth.to_dict()} vs {target}"
    print("  every cascade reconciles: sum(base_quota @ depth d) == target")
    # Weights table tagged per combination
    assert set(weights['base_product']) == {'Jira', 'Confluence'}
    print(f"  weights_long rows: {len(weights)} (1 metric x 4 combos)")
    assert len(weights) == 4


# ----------------------------------------------------------------------
# 2. Prepare-once semantics: same combo/quarters share identical shares
# ----------------------------------------------------------------------
def test_quarters_share_prepared_group():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: quarters reuse the prepared combination — Q2 quotas are "
          "exactly 2x Q1 (same shares, double target)")
    print(SEPARATOR)
    quotas, _ = cascade_many(
        _hierarchy_df(), _target_df(),
        group_keys=GROUP_KEYS, target_col='nn_acv_target',
        taxonomy=TAXONOMY, metrics=KW,
    )
    combo = quotas[(quotas.base_product == 'Confluence')
                   & (quotas.regional == 'Ent_AMER')
                   & quotas.is_leaf]
    q1 = combo[combo.fiscal_quarter == 1].set_index('node_id')['cascaded_quota']
    q2 = combo[combo.fiscal_quarter == 2].set_index('node_id')['cascaded_quota']
    ratio = (q2 / q1).round(6).unique()
    print(f"  Q2/Q1 ratio across territories: {ratio} (expected [2.0])")
    assert list(ratio) == [2.0]


# ----------------------------------------------------------------------
# 3. Gates inside the batch — fully-gated combo redistributes (issue #12)
# ----------------------------------------------------------------------
def test_gates_and_fully_gated_combo():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: gate metrics flow through; the fully-gated Jira/EMEA "
          "combo still lands its target (gate_fallback default)")
    print(SEPARATOR)
    quotas, _ = cascade_many(
        _hierarchy_df(), _target_df(),
        group_keys=GROUP_KEYS, target_col='nn_acv_target',
        taxonomy=TAXONOMY, metrics=KW,
        gate_metrics=[MetricSpec('dc', columns=['dc_seats'])],
    )
    gated_combo = quotas[(quotas.base_product == 'Jira')
                         & (quotas.regional == 'Ent_EMEA')
                         & (quotas.fiscal_quarter == 1)]
    per_depth = gated_combo.groupby('depth')['base_quota'].sum()
    print(f"  Jira/EMEA (all dc_seats=0) per-depth sums:\n{per_depth.to_string()}")
    assert (abs(per_depth - 1_000_000.0) < 0.05).all()
    assert bool(gated_combo['gate_relaxed'].any())
    # A normally-gated combo is untouched: AMER territories with dc pass
    amer = quotas[(quotas.base_product == 'Jira')
                  & (quotas.regional == 'Ent_AMER')
                  & (quotas.fiscal_quarter == 1)]
    assert not amer.get('gate_relaxed', pd.Series(dtype=bool)).any()


# ----------------------------------------------------------------------
# 4. suggest_config: global vs per_group weight suggestion
# ----------------------------------------------------------------------
def test_weights_modes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: suggest_config with weights_mode='global' vs 'per_group'")
    print(SEPARATOR)
    hdf = _hierarchy_df()
    cfg = dict(
        target_column='knowledge_workers',
        candidate_metrics=[{'name': 'dc_seats', 'column': 'dc_seats',
                            'direction': 'proportional', 'lookback': 1,
                            'columns': ['dc_seats']}],
        warn_on_direction_mismatch=False,
        # This test demonstrates per-group WEIGHT VARIATION, which needs
        # the zero-on-degenerate path; since v0.12.0 (issue #33) the
        # default keeps declared weights instead.
        on_degenerate='equal',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, w_global = cascade_many(
            hdf, _target_df(), group_keys=GROUP_KEYS,
            target_col='nn_acv_target', taxonomy=TAXONOMY,
            suggest_config=cfg, weights_mode='global',
        )
        _, w_per = cascade_many(
            hdf, _target_df(), group_keys=GROUP_KEYS,
            target_col='nn_acv_target', taxonomy=TAXONOMY,
            suggest_config=cfg, weights_mode='per_group',
        )
    gw = w_global['input_weight'].round(6).nunique()
    pw = w_per.groupby(GROUP_KEYS)['input_weight'].first().round(6)
    print(f"  global: {gw} distinct weight value(s) across combos (expected 1)")
    print(f"  per_group weights by combo:\n{pw.to_string()}")
    assert gw == 1
    # Per-group must differ somewhere (Jira/EMEA slice has all-zero dc_seats)
    assert pw.nunique() > 1


# ----------------------------------------------------------------------
# 5. on_error='skip' warns + continues; 'raise' fails fast
# ----------------------------------------------------------------------
def test_on_error_modes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: a combination with no hierarchy rows is skipped with a "
          "warning (default) or raises (on_error='raise')")
    print(SEPARATOR)
    bad_targets = pd.concat([_target_df(), pd.DataFrame([
        dict(base_product='Bitbucket', regional='Ent_APAC',
             fiscal_quarter=1, nn_acv_target=500_000.0),
    ])], ignore_index=True)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        quotas, _ = cascade_many(
            _hierarchy_df(), bad_targets,
            group_keys=GROUP_KEYS, target_col='nn_acv_target',
            taxonomy=TAXONOMY, metrics=KW,
        )
    msgs = [str(w.message) for w in wlog]
    assert any('Bitbucket' in m and 'skipped' in m for m in msgs)
    assert quotas.groupby(GROUP_KEYS + ['fiscal_quarter']).ngroups == 8
    print("  skip mode: 8 good cascades completed, 1 warned + skipped")
    try:
        cascade_many(_hierarchy_df(), bad_targets,
                     group_keys=GROUP_KEYS, target_col='nn_acv_target',
                     taxonomy=TAXONOMY, metrics=KW, on_error='raise')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        print(f"  raise mode: {e}")


# ----------------------------------------------------------------------
# 6. Dirty data inside the batch — v0.6.x fixes apply per slice
# ----------------------------------------------------------------------
def test_dirty_data_healed_inside_batch():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: duplicate-level rows + string metrics inside one combo "
          "are healed by the v0.6.x layers, not crashes")
    print(SEPARATOR)
    hdf = _hierarchy_df()
    dirty = pd.DataFrame([dict(
        base_product='Jira', regional='Ent_AMER',
        sub_region='Ent_AMER_W', team='Ent_AMER_W1',
        territory='Ent_AMER_W1',            # == team -> self-loop pre-0.6.0
        knowledge_workers='1,000',           # string -> silent 0 pre-0.6.1
        dc_seats=5,
    )])
    hdf = pd.concat([hdf, dirty], ignore_index=True)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        quotas, _ = cascade_many(
            hdf, _target_df(),
            group_keys=GROUP_KEYS, target_col='nn_acv_target',
            taxonomy=TAXONOMY, metrics=KW,
        )
    combo = quotas[(quotas.base_product == 'Jira')
                   & (quotas.regional == 'Ent_AMER')
                   & (quotas.fiscal_quarter == 1)]
    renamed = combo[combo.node_id == 'Ent_AMER_W1__territory']
    assert len(renamed) == 1 and renamed['cascaded_quota'].iloc[0] > 0
    per_depth = combo.groupby('depth')['base_quota'].sum()
    assert (abs(per_depth - 1_000_000.0) < 0.05).all()
    print(f"  renamed territory funded: "
          f"${renamed['cascaded_quota'].iloc[0]:,.2f}; all depths reconcile")


if __name__ == '__main__':
    test_batch_end_to_end_reconciles()
    test_quarters_share_prepared_group()
    test_gates_and_fully_gated_combo()
    test_weights_modes()
    test_on_error_modes()
    test_dirty_data_healed_inside_batch()

    print(f"\n\n{SEPARATOR}")
    print("ALL CASCADE_MANY TESTS PASSED")
    print(SEPARATOR)
