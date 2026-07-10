"""
Tests for issues #20 / #19 — attrs['combo_report'] per-combination
diagnostics + batch-level direction-mismatch summarization.

Covers:
  - report shape: one record per combination, skipped combos included
    with reason; targets_matched / rows_produced correct
  - gate bookkeeping: n_gated_nodes, gate_relaxed, unallocated_total
    (strand_at_root) surface per combo
  - weights_source across fixed / policy / suggested_per_group /
    default_attainment
  - #19: per_group flood -> exactly ONE summary warning with N/M and
    per-combo detail in the report; explicit True keeps per-group
    warnings; explicit False fully silent (report still populated)
  - attrs survive pd.concat of two outputs
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _hdf():
    rows = []
    for region, kws in [('EMEA', [100, 200, 300, 400]),
                        ('APAC', [50, 150, 250, 350])]:
        for i, k in enumerate(kws):
            rows.append(dict(region=region, team=f'{region}_T{i//2+1}',
                             rep=f'{region}_r{i+1}', kw=k,
                             dc=[0, 0, 5, 9][i]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 1. Shape + skipped combos + counts
# ----------------------------------------------------------------------
def test_report_shape_and_skips():
    print(SEPARATOR)
    print("TEST 1: one record per combo; skipped combo carries reason; "
          "counts correct")
    print(SEPARATOR)
    targets = pd.DataFrame([
        dict(region='EMEA', fiscal_quarter=1, tgt=1_000_000.0),
        dict(region='EMEA', fiscal_quarter=2, tgt=1_100_000.0),
        dict(region='APAC', fiscal_quarter=1, tgt=900_000.0),
        dict(region='GHOST', fiscal_quarter=1, tgt=500_000.0),  # no rows
    ])
    q, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW)
    rep = pd.DataFrame(q.attrs['combo_report']).set_index('region')
    print(rep[['skipped', 'targets_matched', 'rows_produced',
               'weights_source']].to_string())
    assert len(rep) == 3
    assert not rep.loc['EMEA', 'skipped']
    assert rep.loc['EMEA', 'targets_matched'] == 2
    assert rep.loc['EMEA', 'rows_produced'] == 14      # 7 nodes x 2 qtrs
    assert bool(rep.loc['GHOST', 'skipped'])
    assert 'no rows' in rep.loc['GHOST', 'reason']
    assert rep.loc['EMEA', 'weights_source'] == 'fixed'


# ----------------------------------------------------------------------
# 2. Gate bookkeeping per combo (gated / relaxed / unallocated)
# ----------------------------------------------------------------------
def test_gate_bookkeeping():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: n_gated_nodes / unallocated_total (strand_at_root) "
          "surface per combo")
    print(SEPARATOR)
    # EMEA entirely below the gate -> the fallback path fires;
    # APAC has no gates at all.
    hdf = _hdf()
    hdf.loc[hdf.region == 'EMEA', 'dc'] = 0
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                            dict(region='APAC', tgt=900_000.0)])
    gates = lambda g: ([MetricSpec('dc', columns=['dc'],
                                   gate_threshold=1.0)]
                       if g['region'] == 'EMEA' else None)
    common = dict(group_keys=['region'], target_col='tgt',
                  taxonomy=['region', 'team', 'rep'], metrics=KW,
                  gate_metrics=gates)
    # strand_at_root: the whole EMEA target is stranded, not relaxed
    q, _ = cascade_many(hdf, targets, gate_fallback='strand_at_root',
                        **common)
    rep = pd.DataFrame(q.attrs['combo_report']).set_index('region')
    print(rep[['n_gated_nodes', 'gate_relaxed',
               'unallocated_total']].to_string())
    assert rep.loc['EMEA', 'n_gated_nodes'] > 0
    assert rep.loc['EMEA', 'unallocated_total'] > 0    # stranded money
    assert not bool(rep.loc['EMEA', 'gate_relaxed'])
    assert rep.loc['APAC', 'n_gated_nodes'] == 0       # no gates there
    assert rep.loc['APAC', 'unallocated_total'] == 0.0
    # redistribute (default): gate relaxed instead, nothing stranded
    q2, _ = cascade_many(hdf, targets, **common)
    rep2 = pd.DataFrame(q2.attrs['combo_report']).set_index('region')
    assert bool(rep2.loc['EMEA', 'gate_relaxed'])
    assert rep2.loc['EMEA', 'unallocated_total'] == 0.0


# ----------------------------------------------------------------------
# 3. weights_source variants
# ----------------------------------------------------------------------
def test_weights_sources():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: policy vs suggested_per_group vs default_attainment")
    print(SEPARATOR)
    hdf = _hdf()
    hdf['Q1_Attainment'] = 1.0                # for the legacy path
    hdf['perf'] = hdf['kw'] * 2 + 1           # correlated target
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                            dict(region='APAC', tgt=900_000.0)])
    # policy fixes EMEA; APAC falls through to per_group suggestion
    q, _ = cascade_many(
        hdf, targets, group_keys=['region'], target_col='tgt',
        taxonomy=['region', 'team', 'rep'],
        metrics=lambda g: KW if g['region'] == 'EMEA' else None,
        suggest_config=dict(target_column='perf',
                            candidate_metrics=[
                                MetricSpec('kw', columns=['kw'])]),
        weights_mode='per_group')
    rep = pd.DataFrame(q.attrs['combo_report']).set_index('region')
    assert rep.loc['EMEA', 'weights_source'] == 'policy'
    assert rep.loc['APAC', 'weights_source'] == 'suggested_per_group'
    # legacy path (no metrics, no suggest_config)
    q2, _ = cascade_many(hdf, targets, group_keys=['region'],
                         target_col='tgt',
                         taxonomy=['region', 'team', 'rep'])
    rep2 = pd.DataFrame(q2.attrs['combo_report'])
    assert (rep2['weights_source'] == 'default_attainment').all()
    print("  policy / suggested_per_group / default_attainment all "
          "recorded")


# ----------------------------------------------------------------------
# 4. #19 — flood becomes one summary; explicit settings honored
# ----------------------------------------------------------------------
def test_direction_warning_summary():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: per_group mismatches -> ONE summary warning; "
          "explicit True/False honored; report always populated")
    print(SEPARATOR)
    hdf = _hdf()
    hdf['perf'] = hdf['kw'] * 3 + 7           # POSITIVE correlation
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                            dict(region='APAC', tgt=900_000.0)])
    # declared 'inverse' vs positive data -> mismatch in every combo
    def cfg(**extra):
        return dict(target_column='perf',
                    candidate_metrics=[MetricSpec(
                        'kw', direction='inverse', columns=['kw'])],
                    **extra)
    base = dict(group_keys=['region'], target_col='tgt',
                taxonomy=['region', 'team', 'rep'],
                weights_mode='per_group')
    # default: summarized
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        q, _ = cascade_many(hdf, targets, suggest_config=cfg(), **base)
    per_group = [x for x in w
                 if 'warn_on_direction_mismatch=False' in str(x.message)]
    summary = [x for x in w
               if 'across the batch' in str(x.message)]
    assert not per_group, "per-group warnings leaked"
    assert len(summary) == 1
    assert "'kw' in 2/2 combinations" in str(summary[0].message)
    rep = pd.DataFrame(q.attrs['combo_report'])
    assert all(m == ['kw'] for m in rep['direction_mismatches'])
    # explicit True: per-group warnings kept, no summary
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter('always')
        cascade_many(hdf, targets,
                     suggest_config=cfg(warn_on_direction_mismatch=True),
                     **base)
    assert [x for x in w2
            if 'warn_on_direction_mismatch=False' in str(x.message)]
    assert not [x for x in w2 if 'across the batch' in str(x.message)]
    # explicit False: fully silent, report still populated
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter('always')
        q3, _ = cascade_many(
            hdf, targets,
            suggest_config=cfg(warn_on_direction_mismatch=False), **base)
    assert not [x for x in w3 if 'direction' in str(x.message).lower()]
    rep3 = pd.DataFrame(q3.attrs['combo_report'])
    assert all(m == ['kw'] for m in rep3['direction_mismatches'])
    print("  1 summary (2/2); True -> per-group; False -> silent + "
          "report intact")


# ----------------------------------------------------------------------
# 5. attrs survive concat
# ----------------------------------------------------------------------
def test_attrs_concat_safe():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: pd.concat of two outputs doesn't blow up on attrs")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q1, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                         target_col='tgt',
                         taxonomy=['region', 'team', 'rep'], metrics=KW)
    q2 = q1.copy()
    both = pd.concat([q1, q2], ignore_index=True)
    assert len(both) == 2 * len(q1)
    print("  concat ok")


if __name__ == '__main__':
    test_report_shape_and_skips()
    test_gate_bookkeeping()
    test_weights_sources()
    test_direction_warning_summary()
    test_attrs_concat_safe()

    print(f"\n\n{SEPARATOR}")
    print("ALL COMBO-REPORT TESTS PASSED")
    print(SEPARATOR)
