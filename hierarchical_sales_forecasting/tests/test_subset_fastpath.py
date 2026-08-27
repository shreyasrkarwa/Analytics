"""
Tests for issue #64 — subset= fast path + order-independence receipts.

Covers:
  - apply_pins(subset=) == full-frame run, bit-for-bit (values, row
    order, index, attrs) — the library-side slice-and-stitch
  - enforce_identities(subset=) == full-frame run on violations
    confined to the subset
  - closure expansion: a leaf pin's absorption domain (parent's full
    subtree) is pulled in even when the seed IS the leaf
  - guards: pin outside the seeds' subtrees raises; unknown seed
    raises
  - attrs survival: the stitch keeps cascade_row_keys (the real
    footgun behind the filer's "row order changed my results" — a
    hand-rolled pd.concat stitch drops attrs and silently changes
    key resolution)
  - order-independence receipts: shuffled frames produce identical
    results for apply_pins (equal-dollar tied pins) and
    enforce_identities (default / scale_pins / rebalance / leaves)
"""
import warnings
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, enforce_identities,
    reconcile,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    """Two directors x two teams x four reps, two quarters."""
    hdf = pd.DataFrame([dict(region='EMEA', director=f'D{i//8+1}',
                             team=f'D{i//8+1}_T{(i % 8)//4+1}',
                             rep=f'r{i+1}', kw=100 + i)
                        for i in range(16)])
    targets = pd.DataFrame([dict(region='EMEA', fiscal_quarter=fq,
                                 tgt=4_000_000.0) for fq in (1, 2)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region'],
                            target_col='tgt',
                            taxonomy=['region', 'director', 'team',
                                      'rep'],
                            metrics=KW, hedge_multiplier=1.1)
    return q


def _same(a, b):
    ax = a.sort_index()
    bx = b.sort_index()
    return ((ax['base_quota'] - bx['base_quota']).abs().max() < 1e-9
            and (ax['cascaded_quota']
                 - bx['cascaded_quota']).abs().max() < 1e-9)


# ----------------------------------------------------------------------
# 1. apply_pins subset == full
# ----------------------------------------------------------------------
def test_apply_pins_subset_equivalence():
    print(SEPARATOR)
    print("TEST 1: apply_pins(subset=['D1']) == full run; order/index/"
          "attrs preserved")
    print(SEPARATOR)
    q = _quotas()
    pins = [Pin('r1', 500_000.0), Pin('r6', 250_000.0)]
    full, _ = apply_pins(q, pins)
    fast, rep = apply_pins(q, pins, subset=['D1'])
    assert _same(full, fast)
    assert list(fast.index) == list(q.index)
    assert fast.attrs.get('cascade_row_keys') == \
        q.attrs.get('cascade_row_keys')
    assert rep['feasible'].all()
    # untouched director bit-identical
    d2f = fast[fast.node_id.str.startswith('D2') | (fast.node_id == 'D2')]
    d2q = q[q.node_id.str.startswith('D2') | (q.node_id == 'D2')]
    assert (d2f['base_quota'].values == d2q['base_quota'].values).all()
    assert reconcile(fast, hedge=1.1)['ok'].all()
    print("  equivalent, ordered, stamped; D2 untouched")


# ----------------------------------------------------------------------
# 2. enforce subset == full; leaf-seed closure
# ----------------------------------------------------------------------
def test_enforce_subset_and_closure():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: enforce(subset=) == full; leaf seed pulls in the "
          "absorption family for pins")
    print(SEPARATOR)
    q = _quotas()
    q2 = q.copy()
    q2.attrs = dict(q.attrs)
    m = (q2.node_id == 'r1') & (q2.fiscal_quarter == 1)
    q2.loc[m, 'base_quota'] += 40_000.0
    q2.loc[m, 'cascaded_quota'] += 40_000.0 * 1.1 ** 3
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        full, _ = enforce_identities(q2)
        fast, _ = enforce_identities(q2, subset=['D1'])
    assert _same(full, fast)
    # closure: seed == the pinned LEAF still absorbs across siblings
    a, _ = apply_pins(q, [Pin('r1', 500_000.0)], subset=['r1'])
    b, _ = apply_pins(q, [Pin('r1', 500_000.0)])
    assert _same(a, b)
    print("  enforce equivalent; leaf-seed closure == full run")


# ----------------------------------------------------------------------
# 3. Guards
# ----------------------------------------------------------------------
def test_guards():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: pin outside subset raises; unknown seed raises")
    print(SEPARATOR)
    q = _quotas()
    try:
        apply_pins(q, [Pin('r9', 100.0)], subset=['D1'])   # r9 is in D2
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'outside' in str(e) and 'r9' in str(e)
    try:
        apply_pins(q, [Pin('r1', 100.0)], subset=['NOPE'])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'NOPE' in str(e)
    print("  both raise, named")


# ----------------------------------------------------------------------
# 4. Order-independence receipts (the #64 $40K claim, disproven)
# ----------------------------------------------------------------------
def test_order_independence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: shuffled frames -> identical results, all modes, "
          "equal-dollar tied pins")
    print(SEPARATOR)
    q = _quotas()
    pins = [Pin(r, 300_000.0, scope={'fiscal_quarter': 1})
            for r in ['r1', 'r2', 'r3', 'r4']]     # equal-dollar ties
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        e, _ = apply_pins(q, pins, on_overshoot='allow')

    def _run(frame, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            f, _ = enforce_identities(frame, **kw)
        return f.sort_values(['node_id', 'fiscal_quarter']) \
                .reset_index(drop=True)['base_quota']

    for kw in (dict(), dict(on_overshoot='rebalance'),
               dict(anchor='leaves')):
        base = _run(e, **kw)
        for seed in range(3):
            sh = e.sample(frac=1.0, random_state=seed) \
                  .reset_index(drop=True)
            sh.attrs = dict(e.attrs)
            assert (_run(sh, **kw) - base).abs().max() < 1e-9, kw
    # apply_pins itself
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a0, _ = apply_pins(q, pins, on_overshoot='allow')
    a0 = a0.sort_values(['node_id', 'fiscal_quarter']) \
           .reset_index(drop=True)['base_quota']
    for seed in range(3):
        sh = q.sample(frac=1.0, random_state=seed) \
              .reset_index(drop=True)
        sh.attrs = dict(q.attrs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a1, _ = apply_pins(sh, pins, on_overshoot='allow')
        a1 = a1.sort_values(['node_id', 'fiscal_quarter']) \
               .reset_index(drop=True)['base_quota']
        assert (a1 - a0).abs().max() < 1e-9
    print("  0.0 delta across shuffles — row order is NOT load-bearing")


if __name__ == '__main__':
    test_apply_pins_subset_equivalence()
    test_enforce_subset_and_closure()
    test_guards()
    test_order_independence()

    print(f"\n\n{SEPARATOR}")
    print("ALL SUBSET-FASTPATH TESTS PASSED")
    print(SEPARATOR)
