"""
Tests for issue #30 — cascade_levels: the multi-level / recursive
cascade driver with per-transition kwargs.

Covers:
  - equivalence anchor: uniform kwargs -> chained base layer == a
    single full-tree cascade_many
  - different metric blends per level (the genuinely-new capability)
  - per-transition hedge: only its own step hedged, base conserved
  - per-transition pins + sub-target key threading (quarters)
  - dropped aggregation across levels; validation errors
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    MetricSpec,
    cascade_many,
    cascade_levels,
)

SEPARATOR = "=" * 90
TAX = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]
CLOUD = [MetricSpec('cloud', direction='proportional', weight=1.0,
                    columns=['cloud'])]


def _hdf():
    rows = []
    for t, reps in [('AMER_T1', [(100, 10), (300, 30)]),
                    ('AMER_T2', [(200, 100), (400, 100)])]:
        for i, (kw, cloud) in enumerate(reps):
            rows.append(dict(regional='AMER', team=t, rep=f'{t}_r{i+1}',
                             kw=kw, cloud=cloud))
    return pd.DataFrame(rows)


def _targets(**extra):
    return pd.DataFrame([dict(regional='AMER', q=1_000_000.0, **extra)])


# ----------------------------------------------------------------------
# 1. Equivalence anchor — uniform kwargs == full-tree cascade
# ----------------------------------------------------------------------
def test_uniform_equivalence():
    print(SEPARATOR)
    print("TEST 1: uniform kwargs — chained base layer equals a single "
          "full-tree cascade_many")
    print(SEPARATOR)
    chained = cascade_levels(_hdf(), _targets(), taxonomy=TAX,
                             target_col='q',
                             level_kwargs=[dict(metrics=KW),
                                           dict(metrics=KW)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full, _ = cascade_many(_hdf(), _targets(), group_keys=['regional'],
                               target_col='q', taxonomy=TAX, metrics=KW)
    for node in full['node_id']:
        a = chained[chained.node_id == node]['base_quota'].iloc[0]
        b = full[full.node_id == node]['base_quota'].iloc[0]
        assert abs(a - b) < 0.05, node
    print(f"  all {len(full)} nodes match the full-tree base layer")
    # level / depth / is_leaf bookkeeping
    assert list(chained[chained.depth == 0]['level'].unique()) == ['regional']
    assert chained[chained.level == 'rep']['is_leaf'].all()
    assert not chained[chained.level == 'team']['is_leaf'].any()


# ----------------------------------------------------------------------
# 2. Different metric blends per level — the new capability
# ----------------------------------------------------------------------
def test_per_level_metric_blends():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: regional->team by kw, team->rep by cloud")
    print(SEPARATOR)
    r = cascade_levels(_hdf(), _targets(), taxonomy=TAX, target_col='q',
                       level_kwargs=[dict(metrics=KW),
                                     dict(metrics=CLOUD)])
    ix = r.set_index('node_id')
    # Level 1 by kw: T1 has 400/1000 kw -> 400k
    assert abs(ix.loc['AMER_T1', 'base_quota'] - 400_000.0) < 0.05
    # Level 2 by CLOUD within T1: r1 10/40 -> 100k, r2 30/40 -> 300k
    assert abs(ix.loc['AMER_T1_r1', 'base_quota'] - 100_000.0) < 0.05
    assert abs(ix.loc['AMER_T1_r2', 'base_quota'] - 300_000.0) < 0.05
    # And within T2 cloud is equal (100/100) -> 300k each of 600k
    assert abs(ix.loc['AMER_T2_r1', 'base_quota'] - 300_000.0) < 0.05
    print("  teams split by kw · reps split by cloud — hand-verified")


# ----------------------------------------------------------------------
# 3. Per-transition hedge — hedges only its own step
# ----------------------------------------------------------------------
def test_per_level_hedge():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: hedge only the team->rep transition (1.10)")
    print(SEPARATOR)
    r = cascade_levels(_hdf(), _targets(), taxonomy=TAX, target_col='q',
                       level_kwargs=[dict(metrics=KW),
                                     dict(metrics=KW,
                                          hedge_multiplier=1.10)])
    teams = r[r.level == 'team']
    reps = r[r.level == 'rep']
    # Teams un-hedged; reps hedged exactly 10% over their base
    assert (teams['cascaded_quota'] == teams['base_quota']).all()
    assert (abs(reps['cascaded_quota'] - reps['base_quota'] * 1.10)
            < 0.05).all()
    # Base conserves per parent at every level
    for parent, grp in reps.groupby('parent'):
        parent_base = teams[teams.node_id == parent]['base_quota'].iloc[0]
        assert abs(grp['base_quota'].sum() - parent_base) < 0.05
    assert abs(teams['base_quota'].sum() - 1_000_000.0) < 0.05
    print("  teams flat · reps +10% · base conserved at every level")


# ----------------------------------------------------------------------
# 4. Per-transition pins + quarter threading
# ----------------------------------------------------------------------
def test_pins_and_keys():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: pin one rep at the last transition; quarters thread "
          "through as separate cascades")
    print(SEPARATOR)
    targets = pd.DataFrame([
        dict(regional='AMER', fiscal_quarter=1, q=1_000_000.0),
        dict(regional='AMER', fiscal_quarter=2, q=2_000_000.0),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = cascade_levels(
            _hdf(), targets, taxonomy=TAX, target_col='q',
            level_kwargs=[dict(metrics=KW),
                          dict(metrics=KW,
                               new_ic_overrides={'AMER_T1_r1': 50_000.0})])
    reps = r[r.level == 'rep']
    pinned = reps[reps.node_id == 'AMER_T1_r1'].set_index('fiscal_quarter')
    # Per-cascade pin applies in EACH quarter's cascade
    assert abs(pinned.loc[1, 'base_quota'] - 50_000.0) < 0.05
    assert abs(pinned.loc[2, 'base_quota'] - 50_000.0) < 0.05
    # Sibling absorbed; conservation per (quarter, parent)
    for (q, parent), grp in reps.groupby(['fiscal_quarter', 'parent']):
        parent_base = r[(r.level == 'team') & (r.node_id == parent)
                        & (r.fiscal_quarter == q)]['base_quota'].iloc[0]
        assert abs(grp['base_quota'].sum() - parent_base) < 0.05
    # Quarters stayed separate end to end
    assert set(reps['fiscal_quarter']) == {1, 2}
    print("  pin honored per quarter · conservation holds per "
          "(quarter, parent)")


# ----------------------------------------------------------------------
# 5. Dropped aggregation + validation
# ----------------------------------------------------------------------
def test_dropped_and_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: unmatched root target dropped with level tag; "
          "validation errors are clear")
    print(SEPARATOR)
    bad_targets = pd.concat([_targets(), pd.DataFrame(
        [dict(regional='GHOST', q=500_000.0)])], ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, dropped = cascade_levels(_hdf(), bad_targets, taxonomy=TAX,
                                    target_col='q',
                                    level_kwargs=[dict(metrics=KW)] * 2,
                                    return_dropped=True)
    assert len(dropped) == 1 and dropped['regional'].iloc[0] == 'GHOST'
    assert dropped['level'].iloc[0] == 'regional'
    assert len(r.attrs['dropped_targets']) == 1
    print(f"  GHOST dropped at level 'regional' "
          f"(${dropped['q'].iloc[0]:,.2f} visible)")

    def expect(frag, **kw):
        try:
            cascade_levels(**kw)
            raise AssertionError(f'expected ValueError ({frag})')
        except ValueError as e:
            assert frag in str(e), str(e)
            print(f"  {frag}: OK")
    expect('at least 2', hierarchy_df=_hdf(), root_targets=_targets(),
           taxonomy=['regional'], target_col='q')
    expect('one dict per transition', hierarchy_df=_hdf(),
           root_targets=_targets(), taxonomy=TAX, target_col='q',
           level_kwargs=[{}])
    # Ambiguous child ids (same team name under two regionals)
    amb = _hdf()
    amb2 = amb.copy(); amb2['regional'] = 'EMEA'   # same team ids!
    expect('multiple', hierarchy_df=pd.concat([amb, amb2]),
           root_targets=_targets(), taxonomy=TAX, target_col='q')


if __name__ == '__main__':
    test_uniform_equivalence()
    test_per_level_metric_blends()
    test_per_level_hedge()
    test_pins_and_keys()
    test_dropped_and_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL CASCADE-LEVELS TESTS PASSED")
    print(SEPARATOR)
