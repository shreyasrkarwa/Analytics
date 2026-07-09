"""
Tests for the v0.16.0 aggregate-pinning release — issues #22, #31, #24.

Covers:
  - leaf pin to an exact TOTAL across multiple cascades: baseline mix
    preserved, per-cascade sibling absorption, parents conserved,
    hedged derived per-row (never re-hedged)
  - manager/subtree pin: descendants rescaled, sibling team absorbs
  - freeze_nodes / per-pin exclude (#24): protected nodes untouched
  - infeasible pin: siblings floor at $0, unabsorbed reported loudly
  - basis='cascaded' math
  - scope filters (pin Q1 only)
  - validation and provenance columns
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    MetricSpec,
    HedgeByDepth,
    cascade_many,
    Pin,
    apply_pins,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'team', 'rep']
KW = [MetricSpec('kw', direction='proportional', weight=1.0, columns=['kw'])]
RATIO = 1.05 * 1.10          # rep-level compound hedge in these fixtures


def _quotas():
    """2 quarters x 1 regional; T1=(r1:100, r2:200, r3:300), T2=(r4:400)."""
    rows = [dict(regional='AMER', team='T1', rep=f'r{i+1}', kw=kw)
            for i, kw in enumerate([100, 200, 300])]
    rows.append(dict(regional='AMER', team='T2', rep='r4', kw=400))
    hdf = pd.DataFrame(rows)
    targets = pd.DataFrame([
        dict(regional='AMER', fiscal_quarter=1, q=1_000_000.0),
        dict(regional='AMER', fiscal_quarter=2, q=2_000_000.0),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        quotas, _ = cascade_many(
            hdf, targets, group_keys=['regional'], target_col='q',
            taxonomy=TAXONOMY, metrics=KW,
            hedge_multiplier=HedgeByDepth(from_leaves={1: 1.10, 2: 1.05}))
    return quotas


def _depth_ok(df, keycols=('regional', 'fiscal_quarter')):
    """Per-cascade, per-depth base sums must match the root's base."""
    for _, grp in df.groupby(list(keycols)):
        root = grp[grp.depth == 0]['base_quota'].iloc[0]
        per_depth = grp.groupby('depth')['base_quota'].sum()
        if not (abs(per_depth - root) < 0.05).all():
            return False
    return True


# ----------------------------------------------------------------------
# 1. Leaf pin to an exact total across 2 quarters (the #31 scenario)
# ----------------------------------------------------------------------
def test_leaf_pin_total_across_cascades():
    print(SEPARATOR)
    print("TEST 1: pin r1 to $150,000 TOTAL across Q1+Q2 — mix preserved, "
          "siblings absorb per quarter, parents conserved")
    print(SEPARATOR)
    quotas = _quotas()
    # r1 baseline: Q1 100k/600k -> wait, kw split: r1 base Q1 = 1M x 100/1000
    base_before = quotas[quotas.node_id == 'r1']['base_quota'].sum()
    edited, report = apply_pins(quotas, [Pin('r1', 150_000.0)])
    r1 = edited[edited.node_id == 'r1'].set_index('fiscal_quarter')
    # Total hit exactly; mix preserved (Q2 baseline was 2x Q1 -> stays 2x)
    assert abs(r1['base_quota'].sum() - 150_000.0) < 0.05
    assert abs(r1.loc[2, 'base_quota'] / r1.loc[1, 'base_quota'] - 2.0) < 0.01
    # Hedged derived from the row's own ratio
    assert abs(r1.loc[1, 'cascaded_quota']
               - r1.loc[1, 'base_quota'] * RATIO) < 0.5
    # Siblings in T1 absorbed within each quarter; T2 untouched
    r4 = edited[edited.node_id == 'r4']
    r4_orig = quotas[quotas.node_id == 'r4']
    pd.testing.assert_series_equal(
        r4['base_quota'].reset_index(drop=True),
        r4_orig['base_quota'].reset_index(drop=True))
    # Every depth still reconciles per cascade
    assert _depth_ok(edited)
    # Provenance + report
    assert edited[edited.node_id == 'r1']['is_pinned'].all()
    assert (edited[edited.node_id == 'r1']['pin_type'] == 'leaf').all()
    rep = report.iloc[0]
    print(f"  baseline ${rep['baseline_total']:,.2f} -> achieved "
          f"${rep['achieved_total']:,.2f} · absorbed ${rep['absorbed']:,.2f} "
          f"· feasible {rep['feasible']}")
    assert rep['feasible'] and abs(rep['achieved_total'] - 150_000.0) < 0.05
    assert abs(base_before - rep['baseline_total']) < 0.05


# ----------------------------------------------------------------------
# 2. Manager/subtree pin
# ----------------------------------------------------------------------
def test_subtree_pin():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: pin team T1 to $1.2M total — subtree rescales, T2 absorbs")
    print(SEPARATOR)
    quotas = _quotas()
    t1_before = quotas[quotas.node_id == 'T1']['base_quota'].sum()  # 1.8M
    edited, report = apply_pins(quotas, [Pin('T1', 1_200_000.0)])
    assert abs(edited[edited.node_id == 'T1']['base_quota'].sum()
               - 1_200_000.0) < 0.05
    # Subtree scaled proportionally: r1:r2:r3 stays 1:2:3 within each Q
    q1 = edited[edited.fiscal_quarter == 1].set_index('node_id')
    assert abs(q1.loc['r2', 'base_quota'] / q1.loc['r1', 'base_quota']
               - 2.0) < 0.01
    # T2 (and r4) absorbed the shed within each quarter
    t2_gain = (edited[edited.node_id == 'T2']['base_quota'].sum()
               - quotas[quotas.node_id == 'T2']['base_quota'].sum())
    assert abs(t2_gain - (t1_before - 1_200_000.0)) < 0.1
    assert _depth_ok(edited)
    assert report.iloc[0]['pin_type'] == 'subtree'
    print(f"  T1 {t1_before:,.0f} -> 1,200,000 · T2 gained ${t2_gain:,.2f} "
          f"· depths reconcile")


# ----------------------------------------------------------------------
# 3. Freeze / exclude (#24)
# ----------------------------------------------------------------------
def test_freeze_and_exclude():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: #24 — frozen sibling never absorbs, never changes")
    print(SEPARATOR)
    quotas = _quotas()
    edited, report = apply_pins(quotas, [Pin('r1', 150_000.0,
                                             exclude=['r2'])])
    r2 = edited[edited.node_id == 'r2']['base_quota']
    r2_orig = quotas[quotas.node_id == 'r2']['base_quota']
    pd.testing.assert_series_equal(r2.reset_index(drop=True),
                                   r2_orig.reset_index(drop=True))
    # r3 absorbed everything instead; still conserves
    assert _depth_ok(edited)
    # Same via global freeze_nodes
    edited2, _ = apply_pins(quotas, [Pin('r1', 150_000.0)],
                            freeze_nodes=['r2'])
    pd.testing.assert_frame_equal(
        edited.sort_values(['fiscal_quarter', 'node_id']).reset_index(drop=True),
        edited2.sort_values(['fiscal_quarter', 'node_id']).reset_index(drop=True))
    print("  r2 untouched; r3 absorbed the full delta; exclude == freeze")


# ----------------------------------------------------------------------
# 4. Infeasible pin — floors at $0, loud, reported
# ----------------------------------------------------------------------
def test_infeasible_pin():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: pin r1 above what siblings can shed — $0 floors, "
          "unabsorbed reported, feasible=False")
    print(SEPARATOR)
    quotas = _quotas()
    # T1's Q1 pool is 1M; pin r1 to 5M total -> siblings can't absorb
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        edited, report = apply_pins(quotas, [Pin('r1', 5_000_000.0)])
    rep = report.iloc[0]
    assert not rep['feasible'] and rep['unabsorbed'] > 0
    assert (edited['base_quota'] >= -0.01).all()          # never negative
    assert any('could not be absorbed' in str(w.message) for w in wlog)
    print(f"  unabsorbed ${rep['unabsorbed']:,.2f} · min base "
          f"${edited['base_quota'].min():,.2f} (no negatives)")


# ----------------------------------------------------------------------
# 5. basis='cascaded' + scope filters
# ----------------------------------------------------------------------
def test_basis_and_scope():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: basis='cascaded' totals the hedged layer; scope pins "
          "Q1 only")
    print(SEPARATOR)
    quotas = _quotas()
    # cascaded basis: r1's hedged total across quarters == 231,000
    edited, report = apply_pins(quotas, [Pin('r1', 231_000.0,
                                             basis='cascaded')])
    r1 = edited[edited.node_id == 'r1']
    assert abs(r1['cascaded_quota'].sum() - 231_000.0) < 0.5
    assert abs(r1['base_quota'].sum() - 231_000.0 / RATIO) < 0.5
    # scope: only Q1 touched
    edited2, _ = apply_pins(quotas, [Pin('r1', 50_000.0,
                                         scope={'fiscal_quarter': 1})])
    q2_row = edited2[(edited2.node_id == 'r1')
                     & (edited2.fiscal_quarter == 2)]['base_quota'].iloc[0]
    q2_orig = quotas[(quotas.node_id == 'r1')
                     & (quotas.fiscal_quarter == 2)]['base_quota'].iloc[0]
    assert q2_row == q2_orig
    q1_row = edited2[(edited2.node_id == 'r1')
                     & (edited2.fiscal_quarter == 1)]
    assert abs(q1_row['base_quota'].iloc[0] - 50_000.0) < 0.05
    assert _depth_ok(edited2)
    print("  cascaded-basis total exact · Q2 untouched under scope")


# ----------------------------------------------------------------------
# 6. Validation + sequential pins never absorb for each other
# ----------------------------------------------------------------------
def test_validation_and_pin_interaction():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: validation errors · pinned nodes never absorb")
    print(SEPARATOR)
    quotas = _quotas()
    for bad, frag in [
        (lambda: Pin('', 1), 'node id'),
        (lambda: Pin('r1', -5), 'non-negative'),
        (lambda: Pin('r1', 1, basis='plan'), 'basis'),
    ]:
        try:
            bad(); raise AssertionError('expected ValueError')
        except ValueError as e:
            assert frag in str(e)
    try:
        apply_pins(quotas, [Pin('ghost', 1_000.0)])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'matches no rows' in str(e)
    # Two pins in T1: r1 and r2 both pinned -> only r3 absorbs both deltas
    edited, report = apply_pins(quotas, [Pin('r1', 150_000.0),
                                         Pin('r2', 300_000.0)])
    assert abs(edited[edited.node_id == 'r1']['base_quota'].sum()
               - 150_000.0) < 0.05
    assert abs(edited[edited.node_id == 'r2']['base_quota'].sum()
               - 300_000.0) < 0.05
    assert _depth_ok(edited)
    assert report['feasible'].all()
    print("  bad Pins rejected · ghost node rejected · dual pins conserve "
          "via r3 only")


if __name__ == '__main__':
    test_leaf_pin_total_across_cascades()
    test_subtree_pin()
    test_freeze_and_exclude()
    test_infeasible_pin()
    test_basis_and_scope()
    test_validation_and_pin_interaction()

    print(f"\n\n{SEPARATOR}")
    print("ALL APPLY-PINS TESTS PASSED")
    print(SEPARATOR)
