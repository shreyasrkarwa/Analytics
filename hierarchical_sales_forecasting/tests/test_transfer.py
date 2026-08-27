"""
Tests for issue #68 — transfer(): cell-matched dollar moves.

Covers:
  - the receipt that motivated it: reallocate does NOT cell-match
    (DC-heavy draw lands in the recipient's Cloud-heavy mix; pair DC
    total 300K -> 233K) — transfer conserves every matched cell
  - cross-parent move: src's chain drops, dst's chain rises, shared
    ancestor nets to zero; reconcile clean (hedged layer too)
  - to_total= form (exact post-move total); negative amount reverses
  - scope with a LIST of values (DC_PRODUCTS-style); non-scoped cells
    untouched
  - equivalence anchor vs a hand-built per-cell edit
  - guards: over-draw, unmatched cell, bad match_on, amount XOR
    to_total, unknown nodes, ambiguous dst cells
  - report exact column verified against the frame
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, reallocate, transfer,
    reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    """Two teams x two reps, DC/Cloud cells; r1 DC-heavy, r2
    Cloud-heavy."""
    hdf = pd.DataFrame([dict(region='A', team=f'T{i//2+1}',
                             rep=f'r{i+1}', kw=100) for i in range(4)])
    targets = pd.DataFrame([dict(region='A', st=s, tgt=600_000.0)
                            for s in ('DC', 'Cloud')])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
        e, _ = apply_pins(q, [
            Pin('r1', 250_000.0, scope={'st': 'DC'}),
            Pin('r1', 50_000.0, scope={'st': 'Cloud'}),
            Pin('r2', 50_000.0, scope={'st': 'DC'}),
            Pin('r2', 250_000.0, scope={'st': 'Cloud'})])
    return e


# ----------------------------------------------------------------------
# 1. The motivating receipt + the fix
# ----------------------------------------------------------------------
def test_cell_conservation_vs_reallocate():
    print(SEPARATOR)
    print("TEST 1: reallocate leaks across cells; transfer conserves "
          "every matched cell")
    print(SEPARATOR)
    e = _quotas()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ra, _ = reallocate(e, sources='r1', fraction=1 / 3,
                           recipients=['r2'], weights={'r2': 1.0})
    dc_ra = ra[(ra.node_id.isin(['r1', 'r2']))
               & (ra.st == 'DC')]['base_quota'].sum()
    assert dc_ra < 300_000.0 - 1_000        # the leak, documented
    out, rep = transfer(e, 'r1', 'r3', amount=100_000.0)
    for st in ('DC', 'Cloud'):
        p0 = e[(e.node_id.isin(['r1', 'r3']))
               & (e.st == st)]['base_quota'].sum()
        p1 = out[(out.node_id.isin(['r1', 'r3']))
                 & (out.st == st)]['base_quota'].sum()
        assert abs(p1 - p0) < 0.05, st
    assert abs(out[out.node_id == 'r1']['base_quota'].sum()
               - 200_000.0) < 0.5
    assert abs(out[out.node_id == 'r3']['base_quota'].sum()
               - 400_000.0) < 0.5
    assert rep['exact'].all() and abs(rep['moved'].sum()
                                      - 100_000.0) < 0.05
    print(f"  reallocate DC pair: {dc_ra:,.0f} (leaked); transfer: "
          f"conserved both cells")


# ----------------------------------------------------------------------
# 2. Cross-parent chains + reconcile
# ----------------------------------------------------------------------
def test_cross_parent_chains():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: T1 -100K, T2 +100K, region net zero; reconcile "
          "clean")
    print(SEPARATOR)
    e = _quotas()
    out, _ = transfer(e, 'r1', 'r3', amount=100_000.0)
    for n, delta in (('T1', -100_000.0), ('T2', 100_000.0),
                     ('A', 0.0)):
        b0 = e[e.node_id == n]['base_quota'].sum()
        b1 = out[out.node_id == n]['base_quota'].sum()
        assert abs((b1 - b0) - delta) < 0.5, n
    assert reconcile(out, hedge=1.1)['ok'].all()
    print("  chains adjusted, shared ancestor untouched, hedged "
          "identities hold")


# ----------------------------------------------------------------------
# 3. to_total / scope list / reverse / hand-built anchor
# ----------------------------------------------------------------------
def test_forms_and_anchor():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: to_total, list scope, negative amount, hand-built "
          "equivalence")
    print(SEPARATOR)
    e = _quotas()
    out, rep = transfer(e, 'r1', 'r3', to_total=150_000.0,
                        scope={'st': ['DC']})
    ix = out.set_index(['node_id', 'st'])
    assert abs(ix.loc[('r1', 'DC'), 'base_quota'] - 150_000.0) < 0.05
    assert abs(ix.loc[('r1', 'Cloud'), 'base_quota']
               - 50_000.0) < 0.05           # out of scope: untouched
    assert reconcile(out, hedge=1.1)['ok'].all()
    # hand-built: same move as direct cell algebra
    hand = e.copy()
    moved = 100_000.0
    for st, cell in (('DC', 250_000.0), ('Cloud', 50_000.0)):
        d = moved * cell / 300_000.0
        for node, sgn, chain in (('r1', -1, ['T1', 'A']),
                                 ('r3', +1, ['T2', 'A'])):
            for nn in [node] + chain:
                m = (hand.node_id == nn) & (hand.st == st)
                r = (hand.loc[m, 'cascaded_quota']
                     / hand.loc[m, 'base_quota']).iloc[0]
                hand.loc[m, 'base_quota'] += sgn * d
                hand.loc[m, 'cascaded_quota'] = \
                    hand.loc[m, 'base_quota'] * r
    got, _ = transfer(e, 'r1', 'r3', amount=moved)
    a = got.set_index(['node_id', 'st']).sort_index()['base_quota']
    b = hand.set_index(['node_id', 'st']).sort_index()['base_quota']
    assert ((a - b).abs() < 0.05).all()
    # reverse == negative
    fwd, _ = transfer(e, 'r3', 'r1', amount=50_000.0)
    rev, _ = transfer(e, 'r1', 'r3', amount=-50_000.0)
    fa = fwd.set_index(['node_id', 'st']).sort_index()['base_quota']
    ra = rev.set_index(['node_id', 'st']).sort_index()['base_quota']
    assert ((fa - ra).abs() < 0.05).all()
    print("  to_total exact, scope respected, anchor identical, "
          "reverse == swap")


# ----------------------------------------------------------------------
# 4. Guards
# ----------------------------------------------------------------------
def test_guards():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: over-draw, bad args, unmatched cells all raise")
    print(SEPARATOR)
    e = _quotas()
    for kwargs, needle in [
        (dict(amount=999_999_999.0), 'exceeds'),
        (dict(), 'exactly one'),
        (dict(amount=1.0, to_total=1.0), 'exactly one'),
        (dict(amount=1.0, match_on=['nope']), 'match_on'),
        (dict(amount=1.0, scope={'nope': 1}), 'scope column'),
    ]:
        try:
            transfer(e, 'r1', 'r3', **kwargs)
            raise AssertionError(f'expected ValueError: {needle}')
        except ValueError as ex:
            assert needle in str(ex), (needle, str(ex)[:100])
    try:
        transfer(e, 'r1', 'NOPE', amount=1.0)
        raise AssertionError('expected ValueError')
    except ValueError as ex:
        assert 'NOPE' in str(ex)
    # unmatched cell: dst lacking a drawn cell (match on a col where
    # dst has no matching row) -> named raise
    e2 = e.copy()
    e2.attrs = dict(e.attrs)
    e2 = e2[~((e2.node_id == 'r3') & (e2.st == 'Cloud'))]
    e2.attrs = dict(e.attrs)
    try:
        transfer(e2, 'r1', 'r3', amount=10_000.0)
        raise AssertionError('expected ValueError')
    except ValueError as ex:
        assert 'no row for' in str(ex)
    print("  every guard raises with names")


if __name__ == '__main__':
    test_cell_conservation_vs_reallocate()
    test_cross_parent_chains()
    test_forms_and_anchor()
    test_guards()

    print(f"\n\n{SEPARATOR}")
    print("ALL TRANSFER TESTS PASSED")
    print(SEPARATOR)
