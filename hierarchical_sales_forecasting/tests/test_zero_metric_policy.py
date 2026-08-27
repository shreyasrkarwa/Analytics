"""
Tests for issue #66 — zero-metric sibling sets: policy, recording, and
the carve-out fix.

Covers:
  - the RECEIPTS on the disproven claim: an all-zero slice conserves
    (equal fallback), d0 == Σ targets, reconcile clean — money never
    silently disappears
  - the REAL finding: the brand-new-IC carve-out misfires on zero-
    metric slices (a zero-signal rep out-earns a rep WITH signal);
    new_ic_rule='none' turns it off
  - cascade_levels: non-final transitions default the rule OFF — a
    team with zero metric no longer gets a full equal share (the
    500/500-instead-of-1M/0 distortion, fixed)
  - on_zero_metric='fallback' + metric_fallback chain (first column
    with signal wins; 'equal' terminates); 'error' raises naming the
    parent; guards on bad values
  - combo_report accounting: zero_metric_parents / _fallbacks /
    carveout_nodes — the allocation basis is never invisible
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, cascade_levels, MetricSpec, reconcile,
)

SEPARATOR = "=" * 90
DC = [MetricSpec('dc', direction='proportional', weight=1.0,
                 columns=['dc'])]
CL = [MetricSpec('cloud', direction='proportional', weight=1.0,
                 columns=['cloud'])]


# ----------------------------------------------------------------------
# 1. Receipts: all-zero slice conserves (no silent loss)
# ----------------------------------------------------------------------
def test_conservation_receipts():
    print(SEPARATOR)
    print("TEST 1: all-zero sibling slice -> equal split, d0 == "
          "targets, reconcile clean, recorded")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='NA', team='T', rep=f'r{i+1}',
                             cloud_P1=[100, 300, 50, 50][i],
                             cloud_P2=0.0) for i in range(4)])
    targets = pd.DataFrame([dict(region='NA', product=p,
                                 tgt=1_000_000.0) for p in ('P1', 'P2')])
    M = lambda g: [MetricSpec('m', direction='proportional', weight=1.0,
                              columns=[f"cloud_{g['product']}"])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=M, hedge_multiplier=1.1)
    ix = q.set_index(['node_id', 'product'])
    assert all(abs(ix.loc[(f'r{i+1}', 'P2'), 'base_quota'] - 250_000.0)
               < 0.5 for i in range(4))
    assert abs(q[q.depth == 0]['base_quota'].sum() - 2_000_000.0) < 0.5
    assert reconcile(q, hedge=1.1)['ok'].all()
    audit = reconcile(q, targets=targets, target_col='tgt')
    assert audit[audit.check == 'target_total']['ok'].all()
    cr = pd.DataFrame(q.attrs['combo_report'])
    rec = cr.iloc[0]
    # NA->T recorded as a zero-metric split; T->reps went through the
    # carve-out (all reps zero for P2) — BOTH now visible in the report
    assert rec['zero_metric_parents'] == ['NA']
    assert rec['zero_metric_fallbacks'] == ['equal']
    assert rec['carveout_nodes'] == ['r1', 'r2', 'r3', 'r4']
    print("  equal, conserved, target_total clean, recorded")


# ----------------------------------------------------------------------
# 2. Carve-out misfire + new_ic_rule='none'
# ----------------------------------------------------------------------
def test_carveout_off_switch():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: zero-signal rep out-earns a signal rep via the "
          "carve-out; 'none' fixes it")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='NA', team='T', rep=f'r{i+1}',
                             dc=[100, 300, 0, 50][i])
                        for i in range(4)])
    t = pd.DataFrame([dict(region='NA', tgt=1_000_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q0, _ = cascade_many(hdf, t, group_keys=['region'],
                             target_col='tgt',
                             taxonomy=['region', 'team', 'rep'],
                             metrics=DC)
        q1, _ = cascade_many(hdf, t, group_keys=['region'],
                             target_col='tgt',
                             taxonomy=['region', 'team', 'rep'],
                             metrics=DC, new_ic_rule='none')
    i0 = q0.set_index('node_id')
    i1 = q1.set_index('node_id')
    # default: r3 (dc=0) got a FULL equal share, above r4 (dc=50)
    assert i0.loc['r3', 'base_quota'] > i0.loc['r4', 'base_quota']
    assert pd.DataFrame(q0.attrs['combo_report']).iloc[0][
        'carveout_nodes'] == ['r3']
    # 'none': r3 gets 0, others split proportionally
    assert abs(i1.loc['r3', 'base_quota']) < 0.5
    assert abs(i1.loc['r2', 'base_quota']
               - 1_000_000.0 * 300 / 450) < 0.5
    assert pd.DataFrame(q1.attrs['combo_report']).iloc[0][
        'carveout_nodes'] == []
    print(f"  default r3={i0.loc['r3','base_quota']:,.0f} > "
          f"r4={i0.loc['r4','base_quota']:,.0f}; 'none' -> r3=0")


# ----------------------------------------------------------------------
# 3. cascade_levels: non-final transitions default OFF
# ----------------------------------------------------------------------
def test_levels_transition_default():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: a zero-metric TEAM no longer gets a carve-out share "
          "(500/500 -> 1M/0)")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='NA', team=f'T{i//2+1}',
                             rep=f'r{i+1}', dc=[10, 30, 0, 0][i])
                        for i in range(4)])
    t = pd.DataFrame([dict(region='NA', tgt=1_000_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lvl = cascade_levels(hdf, t, taxonomy=['region', 'team', 'rep'],
                             target_col='tgt',
                             level_kwargs=[dict(metrics=DC),
                                           dict(metrics=DC)])
    ix = lvl.set_index('node_id')
    assert abs(ix.loc['T1', 'base_quota'] - 1_000_000.0) < 0.5
    assert abs(ix.loc['T2', 'base_quota']) < 0.5
    assert abs(lvl[lvl.is_leaf]['base_quota'].sum() - 1_000_000.0) < 0.5
    # opting back in per transition still works
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lvl2 = cascade_levels(hdf, t,
                              taxonomy=['region', 'team', 'rep'],
                              target_col='tgt',
                              level_kwargs=[
                                  dict(metrics=DC,
                                       new_ic_rule='all_metrics_zero'),
                                  dict(metrics=DC)])
    ix2 = lvl2.set_index('node_id')
    assert abs(ix2.loc['T2', 'base_quota'] - 500_000.0) < 0.5
    print("  fixed by default; per-transition opt-in preserved")


# ----------------------------------------------------------------------
# 4. Fallback chain + error + guards
# ----------------------------------------------------------------------
def test_fallback_and_error():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: metric_fallback chain, error mode, validation")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='NA', team='T', rep=f'r{i+1}',
                             cloud=0.0, kworkers=[5, 15, 0, 0][i])
                        for i in range(4)])
    t = pd.DataFrame([dict(region='NA', tgt=1_000_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, t, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=CL, on_zero_metric='fallback',
                            metric_fallback=['kworkers', 'equal'],
                            new_ic_rule='none')
    ix = q.set_index('node_id')
    assert abs(ix.loc['r1', 'base_quota'] - 250_000.0) < 0.5
    assert abs(ix.loc['r2', 'base_quota'] - 750_000.0) < 0.5
    cr = pd.DataFrame(q.attrs['combo_report']).iloc[0]
    assert cr['zero_metric_fallbacks'] == ['kworkers']
    assert reconcile(q)['ok'].all()
    # dead fallback column -> lands on 'equal'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q2, _ = cascade_many(hdf, t, group_keys=['region'],
                             target_col='tgt',
                             taxonomy=['region', 'team', 'rep'],
                             metrics=CL, on_zero_metric='fallback',
                             metric_fallback=['nope', 'equal'],
                             new_ic_rule='none')
    assert abs(q2.set_index('node_id').loc['r1', 'base_quota']
               - 250_000.0) < 0.5
    # error mode raises through on_error='raise'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cascade_many(hdf, t, group_keys=['region'],
                         target_col='tgt',
                         taxonomy=['region', 'team', 'rep'],
                         metrics=CL, on_zero_metric='error',
                         new_ic_rule='none', on_error='raise')
        raise AssertionError('expected raise')
    except Exception as e:
        assert 'No metric signal' in str(e)
    # guards
    for kw in (dict(on_zero_metric='nope'),
               dict(on_zero_metric='fallback')):     # missing chain
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cascade_many(hdf, t, group_keys=['region'],
                             target_col='tgt',
                             taxonomy=['region', 'team', 'rep'],
                             metrics=CL, on_error='raise', **kw)
            raise AssertionError(f'expected ValueError for {kw}')
        except Exception as e:
            assert ('on_zero_metric' in str(e)
                    or 'metric_fallback' in str(e)), str(e)
    print("  chain works, dead column skipped, error + guards raise")


if __name__ == '__main__':
    test_conservation_receipts()
    test_carveout_off_switch()
    test_levels_transition_default()
    test_fallback_and_error()

    print(f"\n\n{SEPARATOR}")
    print("ALL ZERO-METRIC-POLICY TESTS PASSED")
    print(SEPARATOR)
