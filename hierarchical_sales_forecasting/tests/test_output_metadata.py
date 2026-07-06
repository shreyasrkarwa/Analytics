"""
Tests for issue #7 — quotas_to_dataframe carries source hierarchy
attributes and sanitized->original id mapping.

Covers:
  - metadata_cols: from_dataframe stores raw; quotas_to_dataframe emits
  - metadata is NOT coerced and NOT read as a metric signal
  - original_id column appears when the collision policy renamed nodes
  - source_df left-join onto leaf rows, keyed on ORIGINAL ids
  - column collision handling ('_source' suffix)
  - cascade_many(metadata_cols=...) passthrough
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    cascade_many,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _df():
    return pd.DataFrame([
        dict(regional='AMER', team='T1', rep='r1', kw=100,
             rep_name='Ada Lovelace', segment='Enterprise', geo='US-West'),
        dict(regional='AMER', team='T1', rep='r2', kw=300,
             rep_name='Grace Hopper', segment='Enterprise', geo='US-East'),
        dict(regional='AMER', team='T2', rep='r3', kw=600,
             rep_name='Katherine Johnson', segment='Strategic', geo='EMEA-UK'),
    ])


def _build(df=None, **kw):
    h = SalesHierarchy()
    h.from_dataframe(df if df is not None else _df(), path_cols=TAXONOMY,
                     metrics_cols=['kw'],
                     metadata_cols=['rep_name', 'segment', 'geo'], **kw)
    return h


# ----------------------------------------------------------------------
# 1. metadata_cols round-trip: stored raw, emitted on leaf rows
# ----------------------------------------------------------------------
def test_metadata_roundtrip():
    print(SEPARATOR)
    print("TEST 1: metadata_cols stored raw and emitted by "
          "quotas_to_dataframe")
    print(SEPARATOR)
    h = _build()
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW, verbose=False)
    df = c.quotas_to_dataframe(q, level_names=TAXONOMY,
                               metadata_cols=['rep_name', 'segment', 'geo'])
    leaf = df[df.node_id == 'r1'].iloc[0]
    print(f"  r1 -> {leaf['rep_name']} / {leaf['segment']} / {leaf['geo']}")
    assert leaf['rep_name'] == 'Ada Lovelace'
    assert leaf['segment'] == 'Enterprise' and leaf['geo'] == 'US-West'
    # Non-leaf rows carry NaN, not garbage
    assert pd.isna(df[df.node_id == 'T1']['rep_name'].iloc[0])
    # Metadata was NOT coerced ('US-West' would have failed coercion)
    assert h.graph.nodes['r1']['geo'] == 'US-West'


# ----------------------------------------------------------------------
# 2. Metadata never influences the cascade (not a metric)
# ----------------------------------------------------------------------
def test_metadata_not_a_signal():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: metadata does not affect quotas (kw-only split intact)")
    print(SEPARATOR)
    h = _build()
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW, verbose=False)
    print(f"  r1: ${q['r1']:,.2f} (expected $100,000)")
    assert abs(q['r1'] - 100_000.0) < 0.01
    assert abs(q['r2'] - 300_000.0) < 0.01
    assert abs(q['r3'] - 600_000.0) < 0.01


# ----------------------------------------------------------------------
# 3. original_id column after a collision rename
# ----------------------------------------------------------------------
def test_original_id_after_rename():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: original_id maps sanitized ids back (id_map)")
    print(SEPARATOR)
    df = _df()
    # rep 'T1' collides with its team -> renamed to 'T1__rep'
    df.loc[0, 'rep'] = 'T1'
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = _build(df)
    assert h.id_map == {'T1__rep': 'T1'}
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW, verbose=False)
    out = c.quotas_to_dataframe(q, metadata_cols=['rep_name'])
    renamed = out[out.node_id == 'T1__rep'].iloc[0]
    print(f"  node_id={renamed['node_id']} -> original_id="
          f"{renamed['original_id']} ({renamed['rep_name']})")
    assert renamed['original_id'] == 'T1'
    assert renamed['rep_name'] == 'Ada Lovelace'
    # Untouched nodes map to themselves
    assert out[out.node_id == 'r2']['original_id'].iloc[0] == 'r2'


# ----------------------------------------------------------------------
# 4. source_df left-join on leaf rows, keyed on ORIGINAL ids
# ----------------------------------------------------------------------
def test_source_df_join():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: source_df joins onto leaf rows via original ids "
          "(survives renames)")
    print(SEPARATOR)
    df = _df()
    df.loc[0, 'rep'] = 'T1'   # force a rename of row 0's leaf
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = _build(df)
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW, verbose=False)
    out = c.quotas_to_dataframe(
        q,
        source_df=df[['rep', 'rep_name', 'kw']],
        source_join_col='rep',
    )
    renamed = out[out.node_id == 'T1__rep'].iloc[0]
    print(f"  T1__rep joined -> rep_name={renamed['rep_name']}, "
          f"kw={renamed['kw']}")
    assert renamed['rep_name'] == 'Ada Lovelace'   # joined via original 'T1'
    assert renamed['kw'] == 100
    # Non-leaf rows are not joined
    assert pd.isna(out[out.node_id == 'T2']['rep_name'].iloc[0])
    # source_df without join col raises clearly
    try:
        c.quotas_to_dataframe(q, source_df=df)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'source_join_col' in str(e)


# ----------------------------------------------------------------------
# 5. Column collisions get the '_source' suffix
# ----------------------------------------------------------------------
def test_join_column_collision_suffix():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: overlapping source_df columns arrive as '<col>_source'")
    print(SEPARATOR)
    h = _build()
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=KW, verbose=False)
    src = _df()[['rep', 'rep_name']].copy()
    src['depth'] = 'from-source'          # collides with output's depth
    out = c.quotas_to_dataframe(q, source_df=src, source_join_col='rep')
    assert 'depth_source' in out.columns  # suffixed, original intact
    assert out['depth'].dtype != object or 'from-source' not in set(out['depth'])
    print(f"  columns: ...{[c for c in out.columns if 'depth' in c]}")


# ----------------------------------------------------------------------
# 6. cascade_many passthrough
# ----------------------------------------------------------------------
def test_cascade_many_metadata():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: cascade_many(metadata_cols=...) tags leaf rows")
    print(SEPARATOR)
    hdf = _df()
    targets = pd.DataFrame([dict(regional='AMER', q_target=1_000_000.0)])
    quotas, _ = cascade_many(
        hdf, targets, group_keys=['regional'], target_col='q_target',
        taxonomy=TAXONOMY, metrics=KW,
        metadata_cols=['rep_name', 'segment', 'geo'],
    )
    r3 = quotas[quotas.node_id == 'r3'].iloc[0]
    print(f"  r3 -> {r3['rep_name']} / {r3['segment']} / {r3['geo']}")
    assert r3['rep_name'] == 'Katherine Johnson'
    assert r3['segment'] == 'Strategic'
    # Metadata excluded from metrics: kw-only split preserved
    assert abs(r3['cascaded_quota'] - 600_000.0) < 0.01


if __name__ == '__main__':
    test_metadata_roundtrip()
    test_metadata_not_a_signal()
    test_original_id_after_rename()
    test_source_df_join()
    test_join_column_collision_suffix()
    test_cascade_many_metadata()

    print(f"\n\n{SEPARATOR}")
    print("ALL OUTPUT-METADATA TESTS PASSED")
    print(SEPARATOR)
