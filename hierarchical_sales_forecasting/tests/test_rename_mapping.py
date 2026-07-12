"""
Tests for issue #18 — sanitized ids always map back to originals.

v0.8.0 (#7) added id_map + original_id; #18 exposed the batch holes.
Covers:
  - mixed batch (one combo renamed, one clean): original_id is
    self-mapping EVERYWHERE — no concat NaN hole
  - original_parent emitted; children of a renamed MANAGER map back
  - attrs['id_map'] records per combination (sanitized -> original,
    tagged with group keys)
  - single-cascade quotas_to_dataframe emits both columns
  - clean batches: no columns, empty attrs mapping (no clutter)
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy, QuotaCascader, cascade_many, MetricSpec,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _batch():
    # EMEA: TEAM 'EMEA' collides with its region -> renamed manager,
    # so its reps' parent is sanitized. APAC is clean.
    hdf = pd.DataFrame([
        dict(region='EMEA', team='EMEA', rep='r1', kw=100),  # collision
        dict(region='EMEA', team='EMEA', rep='r2', kw=200),
        dict(region='APAC', team='A1',   rep='a1', kw=300),
        dict(region='APAC', team='A1',   rep='a2', kw=400),
    ])
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                            dict(region='APAC', tgt=900_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW)
    return q


# ----------------------------------------------------------------------
# 1. No concat NaN hole; renamed manager's children map back
# ----------------------------------------------------------------------
def test_batch_self_mapping_everywhere():
    print(SEPARATOR)
    print("TEST 1: original_id/original_parent self-map batch-wide; "
          "renamed manager's reps map back")
    print(SEPARATOR)
    q = _batch()
    print(q[['region', 'node_id', 'parent', 'original_id',
             'original_parent']].to_string(index=False))
    assert q['original_id'].notna().all()              # the NaN hole
    ix = q.set_index('node_id')
    assert ix.loc['EMEA__team', 'original_id'] == 'EMEA'
    # children of the renamed manager: parent sanitized, original mapped
    assert ix.loc['r1', 'parent'] == 'EMEA__team'
    assert ix.loc['r1', 'original_parent'] == 'EMEA'
    # clean combo: pure self-mapping (was NaN before v0.31.0)
    assert ix.loc['a1', 'original_id'] == 'a1'
    assert ix.loc['a1', 'original_parent'] == 'A1'
    # roots keep None/NaN parent on both columns
    assert pd.isna(ix.loc['APAC', 'original_parent'])


# ----------------------------------------------------------------------
# 2. attrs['id_map'] records per combination
# ----------------------------------------------------------------------
def test_attrs_id_map():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: attrs['id_map'] = per-combination sanitized->original "
          "records")
    print(SEPARATOR)
    q = _batch()
    recs = q.attrs['id_map']
    print(f"  {recs}")
    assert recs == [{'region': 'EMEA', 'sanitized': 'EMEA__team',
                     'original': 'EMEA'}]
    # attrs stay concat-safe (records, not frames)
    both = pd.concat([q, q.copy()], ignore_index=True)
    assert len(both) == 2 * len(q)


# ----------------------------------------------------------------------
# 3. Single-cascade path emits both columns
# ----------------------------------------------------------------------
def test_single_cascade_columns():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: quotas_to_dataframe emits original_parent too")
    print(SEPARATOR)
    df = pd.DataFrame([dict(region='R', team='R', rep='x1', kw=100),
                       dict(region='R', team='R', rep='x2', kw=300)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = SalesHierarchy()
        h.from_dataframe(df, path_cols=['region', 'team', 'rep'],
                         metrics_cols=['kw'])
    c = QuotaCascader(h)
    quotas = c.cascade_quota('R', 500_000.0, metrics=KW, verbose=False)
    out = c.quotas_to_dataframe(quotas).set_index('node_id')
    assert out.loc['R__team', 'original_id'] == 'R'
    assert out.loc['x1', 'original_parent'] == 'R'
    assert pd.isna(out.loc['R', 'original_parent'])
    print("  original_id + original_parent present and correct")


# ----------------------------------------------------------------------
# 4. Clean runs stay clean: no columns, empty mapping
# ----------------------------------------------------------------------
def test_clean_runs_unchanged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: no renames -> no original_* columns, empty "
          "attrs['id_map']")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='EMEA', team='T1', rep=f'r{i}',
                             kw=100 * (i + 1)) for i in range(2)])
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q, _ = cascade_many(hdf, targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW)
    assert 'original_id' not in q.columns
    assert 'original_parent' not in q.columns
    assert q.attrs['id_map'] == []
    print("  clean output uncluttered")


if __name__ == '__main__':
    test_batch_self_mapping_everywhere()
    test_attrs_id_map()
    test_single_cascade_columns()
    test_clean_runs_unchanged()

    print(f"\n\n{SEPARATOR}")
    print("ALL RENAME-MAPPING TESTS PASSED")
    print(SEPARATOR)
