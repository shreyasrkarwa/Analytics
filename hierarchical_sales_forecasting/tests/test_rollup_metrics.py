"""
Tests for issues #17 / #49 — rollup_metrics() + cascade_many's
attach_metrics=: subtree metric aggregates on cascade outputs.

Covers:
  - subtree sums correct at every depth, per cascade (quarters
    isolated); leaf rows carry their own value; source columns stay
    leaf-grain (NaN on managers — the v0.19.2 contract)
  - root consistency anchor: root _subtree == source-frame sum
  - attach_metrics=True == manual rollup_metrics; list form
  - agg='mean'/'max' aggregate over descendant LEAVES
  - cascade_levels outputs work via their new cascade_row_keys stamp
  - validation: missing column (message points at metadata_cols),
    non-numeric, poisoned keys raise (the #40 guard)
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, cascade_levels, MetricSpec, rollup_metrics,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas(**kw):
    hdf = pd.DataFrame([
        dict(product='cloud', regional='EMEA', team=f'T{i//2+1}',
             rep=f'r{i+1}', kw=[100, 200, 300, 400][i],
             seats=[10, 20, 30, 40][i]) for i in range(4)])
    targets = pd.DataFrame([dict(product='cloud', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['product'],
                        target_col='tgt',
                        taxonomy=['regional', 'team', 'rep'],
                        metrics=KW, **kw)
    return q


# ----------------------------------------------------------------------
# 1. Correct sums at every depth; quarters isolated; sources untouched
# ----------------------------------------------------------------------
def test_subtree_sums():
    print(SEPARATOR)
    print("TEST 1: kw_subtree/seats_subtree at rep/team/region, per "
          "quarter; leaf-grain columns untouched")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['kw', 'seats'])
    out = rollup_metrics(q, ['kw', 'seats'])
    ix = out.set_index(['node_id', 'fiscal_quarter'])
    for fq in (1, 2):                       # per-cascade isolation
        assert ix.loc[('r1', fq), 'kw_subtree'] == 100      # own value
        assert ix.loc[('T1', fq), 'kw_subtree'] == 300
        assert ix.loc[('T2', fq), 'seats_subtree'] == 70
        assert ix.loc[('EMEA', fq), 'kw_subtree'] == 1000   # root anchor
    # source columns stay leaf-grain: NaN on managers (v0.19.2 contract)
    assert pd.isna(ix.loc[('T1', 1), 'kw'])
    assert pd.isna(ix.loc[('EMEA', 2), 'seats'])
    print("  T1=300 T2 seats=70 EMEA=1000 per quarter; kw still NaN on "
          "managers")


# ----------------------------------------------------------------------
# 2. attach_metrics equivalence
# ----------------------------------------------------------------------
def test_attach_metrics():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: attach_metrics=True == manual rollup; list form "
          "selects columns")
    print(SEPARATOR)
    q_manual = rollup_metrics(_quotas(metadata_cols=['kw', 'seats']),
                              ['kw', 'seats'])
    q_auto = _quotas(metadata_cols=['kw', 'seats'], attach_metrics=True)
    a = q_auto.set_index(['node_id', 'fiscal_quarter']).sort_index()
    b = q_manual.set_index(['node_id', 'fiscal_quarter']).sort_index()
    for col in ('kw_subtree', 'seats_subtree'):
        assert a[col].equals(b[col]), col
    assert q_auto.attrs['cascade_row_keys'] == ['product',
                                                'fiscal_quarter']
    q_list = _quotas(metadata_cols=['kw', 'seats'],
                     attach_metrics=['kw'])
    assert 'kw_subtree' in q_list.columns
    assert 'seats_subtree' not in q_list.columns
    # attach_metrics=True without carried metrics -> clear error
    try:
        _quotas(attach_metrics=True)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'metadata_cols' in str(e)
    print("  identical columns; attrs preserved; list form works")


# ----------------------------------------------------------------------
# 3. mean/max aggregate over descendant leaves
# ----------------------------------------------------------------------
def test_leaf_aggs():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: agg='mean'/'max' over LEAVES (no mean-of-means)")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['kw'])
    mean = rollup_metrics(q, 'kw', agg='mean').set_index(
        ['node_id', 'fiscal_quarter'])
    mx = rollup_metrics(q, 'kw', agg='max').set_index(
        ['node_id', 'fiscal_quarter'])
    assert mean.loc[('T1', 1), 'kw_subtree'] == 150     # (100+200)/2
    assert mean.loc[('EMEA', 1), 'kw_subtree'] == 250   # leaves, not teams
    assert mx.loc[('EMEA', 2), 'kw_subtree'] == 400
    print("  EMEA mean=250 (leaf mean, not mean of team means)")


# ----------------------------------------------------------------------
# 4. cascade_levels output (via its new cascade_row_keys stamp)
# ----------------------------------------------------------------------
def test_cascade_levels_output():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: rollup on cascade_levels output — per-node parent "
          "columns don't poison the key")
    print(SEPARATOR)
    hdf = pd.DataFrame([
        dict(region='EMEA', rvp=f'V{i//2+1}', director=f'D{i+1}',
             nn=100 * (i + 1)) for i in range(4)])
    rt = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    NN = [MetricSpec('nn', direction='proportional', weight=1.0,
                     columns=['nn'])]
    res = cascade_levels(hdf, rt, taxonomy=['region', 'rvp', 'director'],
                         target_col='tgt',
                         level_kwargs=[dict(metrics=NN),
                                       dict(metrics=NN,
                                            metadata_cols=['nn'])])
    assert res.attrs['cascade_row_keys'] == ['region']
    out = rollup_metrics(res, 'nn').set_index('node_id')
    assert out.loc['V1', 'nn_subtree'] == 300           # D1+D2
    assert out.loc['V2', 'nn_subtree'] == 700
    assert out.loc['EMEA', 'nn_subtree'] == 1000
    print("  V1=300 V2=700 EMEA=1000; stamp = ['region']")


# ----------------------------------------------------------------------
# 5. Validation
# ----------------------------------------------------------------------
def test_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: missing column / non-numeric / poisoned keys")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['kw'])
    try:
        rollup_metrics(q, 'nope')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'metadata_cols' in str(e)        # points at the fix
    q2 = q.copy()
    q2['label'] = 'x'
    try:
        rollup_metrics(q2, 'label')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'not numeric' in str(e)
    # poisoned keys: strip the stamp, add a per-node column
    q3 = _quotas(metadata_cols=['kw', 'seats'])
    q3.attrs = {}
    try:
        rollup_metrics(q3, 'kw')                # 'seats' poisons inference
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'seats' in str(e) and 'row_keys' in str(e)
    # explicit correct keys fix it
    out = rollup_metrics(q3, 'kw', row_keys=['product', 'fiscal_quarter'])
    assert out.set_index(['node_id', 'fiscal_quarter']) \
              .loc[('EMEA', 1), 'kw_subtree'] == 1000
    print("  all raise with clear messages; explicit row_keys works")


if __name__ == '__main__':
    test_subtree_sums()
    test_attach_metrics()
    test_leaf_aggs()
    test_cascade_levels_output()
    test_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL ROLLUP-METRICS TESTS PASSED")
    print(SEPARATOR)
