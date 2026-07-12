"""
Tests for issue #15 — adjust_many(): PipelineAdjuster on batch outputs.

Covers:
  - EQUIVALENCE ANCHOR: adjust_many == hand-built SalesHierarchy +
    PipelineAdjuster (diagnose AND redistribute), so batch numbers
    can't drift from the single-cascade class
  - flag_only: risk columns added, quotas untouched, quarters isolated
  - redistribute: zero-sum per team on the cascaded layer, base
    re-derived per row, managers unchanged, max_adjustment_pct rail,
    locked_nodes respected, quota_delta audit column
  - reconcile() stays clean after adjustment (conservation + hedge)
  - threshold inheritance via region key; missing pipeline column
    error points at metadata_cols
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy, QuotaCascader, PipelineAdjuster,
    cascade_many, MetricSpec, adjust_many, reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _hdf():
    # T1: r1 receiver (huge pipeline), r2 donor (thin pipeline).
    # T2: both moderate -> untouched by redistribution.
    return pd.DataFrame([
        dict(region='EMEA', team='T1', rep='r1', kw=100, pipe=800_000.0),
        dict(region='EMEA', team='T1', rep='r2', kw=200, pipe=100_000.0),
        dict(region='EMEA', team='T2', rep='r3', kw=300, pipe=700_000.0),
        dict(region='EMEA', team='T2', rep='r4', kw=400, pipe=900_000.0),
    ])


def _quotas(**kw):
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=KW, hedge_multiplier=1.1,
                        metadata_cols=['pipe'], **kw)
    return q


# ----------------------------------------------------------------------
# 1. Equivalence anchor vs the real single-cascade class
# ----------------------------------------------------------------------
def test_equivalence_with_pipeline_adjuster():
    print(SEPARATOR)
    print("TEST 1: adjust_many == SalesHierarchy + PipelineAdjuster, "
          "diagnose AND redistribute")
    print(SEPARATOR)
    q = _quotas()
    out = adjust_many(q, 'pipe', mode='redistribute')

    # hand-drive the single-cascade path on the same data
    h = SalesHierarchy()
    h.from_dataframe(_hdf(), path_cols=['region', 'team', 'rep'],
                     metrics_cols=['kw', 'pipe'])
    c = QuotaCascader(h)
    quotas = c.cascade_quota('EMEA', 1_000_000.0, metrics=KW,
                             hedge_multiplier=1.1, verbose=False)
    pa = PipelineAdjuster(h, quotas, pipeline_attr='pipe')
    diag = pa.diagnose().set_index('Node')
    adjusted = pa.adjust('redistribute')

    ix = out[out.fiscal_quarter == 1].set_index('node_id')
    for node in ('EMEA', 'T1', 'r1', 'r2', 'r3'):
        assert abs(ix.loc[node, 'coverage_ratio']
                   - diag.at[node, 'Coverage_Ratio']) < 1e-6, node
        assert ix.loc[node, 'risk_status'] == diag.at[node,
                                                      'Risk_Status'], node
        assert abs(ix.loc[node, 'cascaded_quota']
                   - adjusted[node]) < 0.05, node
    print("  coverage, risk bands, and adjusted quotas identical")


# ----------------------------------------------------------------------
# 2. flag_only: columns on, money untouched
# ----------------------------------------------------------------------
def test_flag_only():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: flag_only adds diagnosis, changes nothing")
    print(SEPARATOR)
    q = _quotas()
    out = adjust_many(q, 'pipe')
    a = out.set_index(['node_id', 'fiscal_quarter']).sort_index()
    b = q.set_index(['node_id', 'fiscal_quarter']).sort_index()
    assert a['cascaded_quota'].equals(b['cascaded_quota'])
    assert a['base_quota'].equals(b['base_quota'])
    assert out['risk_status'].notna().all()
    assert 'quota_delta' not in out.columns
    # quarters carry identical diagnosis (same data both quarters)
    assert (a.loc[('r2', 1), 'risk_status']
            == a.loc[('r2', 2), 'risk_status'])
    print(f"  r1={a.loc[('r1', 1), 'risk_status']} "
          f"r2={a.loc[('r2', 1), 'risk_status']}; quotas untouched")


# ----------------------------------------------------------------------
# 3. redistribute: zero-sum, rails, managers, reconcile-clean
# ----------------------------------------------------------------------
def test_redistribute_invariants():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: zero-sum per team, 20% rail, managers unchanged, "
          "reconcile clean")
    print(SEPARATOR)
    q = _quotas()
    out = adjust_many(q, 'pipe', mode='redistribute')
    o1 = out[out.fiscal_quarter == 1].set_index('node_id')
    q1 = q[q.fiscal_quarter == 1].set_index('node_id')
    # something moved: r1 (receiver) up, r2 (donor) down
    assert o1.loc['r1', 'quota_delta'] > 0
    assert o1.loc['r2', 'quota_delta'] < 0
    # zero-sum within T1 on the cascaded layer
    assert abs(o1.loc[['r1', 'r2'], 'cascaded_quota'].sum()
               - q1.loc[['r1', 'r2'], 'cascaded_quota'].sum()) < 0.05
    # rail: nobody moved more than 20%
    for r in ('r1', 'r2', 'r3', 'r4'):
        assert (abs(o1.loc[r, 'quota_delta'])
                <= 0.20 * q1.loc[r, 'cascaded_quota'] + 0.05), r
    # managers never change
    for m in ('EMEA', 'T1', 'T2'):
        assert abs(o1.loc[m, 'cascaded_quota']
                   - q1.loc[m, 'cascaded_quota']) < 0.005, m
    # base re-derived via per-row ratios -> reconcile fully clean
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = reconcile(out, hedge=1.1)
    assert f['ok'].all() and not [x for x in w
                                  if 'reconcile' in str(x.message)]
    # share_of_parent recomputed and consistent
    assert abs(out[(out.parent == 'T1') & (out.fiscal_quarter == 1)]
               ['share_of_parent'].sum() - 1.0) < 1e-4
    print(f"  r1 +{o1.loc['r1', 'quota_delta']:,.2f} / "
          f"r2 {o1.loc['r2', 'quota_delta']:,.2f}; reconcile ok")


# ----------------------------------------------------------------------
# 4. locked_nodes + threshold inheritance
# ----------------------------------------------------------------------
def test_locked_and_thresholds():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: locked IC blocks its team's redistribution; region "
          "threshold inherited")
    print(SEPARATOR)
    q = _quotas()
    out = adjust_many(q, 'pipe', mode='redistribute',
                      locked_nodes={'r2'})
    o1 = out[out.fiscal_quarter == 1].set_index('node_id')
    # r2 locked -> <2 adjustable ICs in T1 -> nothing moves there
    assert (o1.loc[['r1', 'r2'], 'quota_delta'] == 0).all()
    # inherited region threshold flips the bands
    strict = adjust_many(q, 'pipe', coverage_thresholds={
        'EMEA': {'healthy': 50.0, 'at_risk': 25.0}})
    s1 = strict[strict.fiscal_quarter == 1].set_index('node_id')
    assert s1.loc['r1', 'healthy_threshold'] == 50.0   # inherited
    assert s1.loc['r1', 'risk_status'] in ('At Risk', 'Critical')
    print("  locked team frozen; EMEA thresholds inherited by reps")


# ----------------------------------------------------------------------
# 5. Missing pipeline column -> metadata_cols hint
# ----------------------------------------------------------------------
def test_missing_column_hint():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: missing pipeline column errors with the "
          "metadata_cols pointer")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW)
    try:
        adjust_many(q, 'pipe')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'metadata_cols' in str(e)
    print("  clear error with the fix in it")


if __name__ == '__main__':
    test_equivalence_with_pipeline_adjuster()
    test_flag_only()
    test_redistribute_invariants()
    test_locked_and_thresholds()
    test_missing_column_hint()

    print(f"\n\n{SEPARATOR}")
    print("ALL ADJUST-MANY TESTS PASSED")
    print(SEPARATOR)
