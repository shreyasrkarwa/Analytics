"""
Tests for issues #67 / #62 — zero-baseline hedge-ratio derivation, and
the receipts that pin allocation is already mix-preserving.

Covers:
  - the #67 repro FIXED: a cascaded-basis pin on a zeroed slice now
    derives base from the siblings' ratio (1.21), not base=cascaded;
    reconcile(hedge=) clean; the anchor='leaves' propagated error gone
  - base-basis pins symmetric (cascaded = base x ratio, not x1.0)
  - hedge= equivalence: float == {depth: cum} dict == HedgeByDepth ==
    sibling inference; hedge= wins over inference
  - all-siblings-zero -> same-depth fallback; nothing available ->
    1.0 + the warning names the rows and source
  - the warning fires only when a zero-baseline row RECEIVES money
    (gated 0/0 rows that stay 0 are silent)
  - #62 receipts: unscoped Pin(total) on a 70/30 node gives 770/330
    (row x total/current_total — the uniform multiplier, reading the
    CURRENT frame); the filer's 'observed' shapes reproduce only as
    overlapping-pin composition (#63), documented here
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins, enforce_identities,
    reconcile, HedgeByDepth,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _zeroed():
    """5-rep team, hedge 1.1 (reps depth 2 -> cum ratio 1.21); r4
    zeroed like a no-dc_seats Migration slice."""
    hdf = pd.DataFrame([dict(region='JP', team='T', rep=r, kw=100)
                        for r in ['r1', 'r2', 'r3', 'r4', 'r5']])
    targets = pd.DataFrame([dict(region='JP', st='Migration',
                                 tgt=900_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets, group_keys=['region', 'st'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
        z, _ = apply_pins(q, [Pin('r4', 0.0)])
    return z


# ----------------------------------------------------------------------
# 1. #67 repro fixed: sibling inference
# ----------------------------------------------------------------------
def test_sibling_inference_fixes_67():
    print(SEPARATOR)
    print("TEST 1: cascaded-basis pin on zeroed slice -> ratio from "
          "siblings; reconcile clean; leaves-anchor error gone")
    print(SEPARATOR)
    z = _zeroed()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, _ = apply_pins(z, [Pin('r4', 213_340.0, basis='cascaded')])
    ix = e.set_index('node_id')
    b, c = ix.loc['r4', 'base_quota'], ix.loc['r4', 'cascaded_quota']
    assert abs(c - 213_340.0) < 0.05
    assert abs(c / b - 1.21) < 1e-3          # not 1.0 (the old poison)
    assert abs(b - 213_340.0 / 1.21) < 0.5
    msgs = [str(x.message) for x in w
            if 'zero-baseline' in str(x.message)]
    assert msgs and 'siblings' in msgs[0] and 'r4' in msgs[0]
    assert reconcile(e, hedge=1.1)['ok'].all()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f, _ = enforce_identities(e, anchor='leaves')
    ixf = f.set_index('node_id')
    kids = sum(ixf.loc[x, 'base_quota']
               for x in ['r1', 'r2', 'r3', 'r4', 'r5'])
    assert abs(ixf.loc['T', 'base_quota'] - kids) < 0.05
    assert reconcile(f, hedge=1.1)['ok'].all()   # no 3-levels-away error
    print(f"  ratio {c/b:.3f}; reconcile ok before AND after leaves-"
          f"anchor")


# ----------------------------------------------------------------------
# 2. hedge= equivalence + precedence; base-basis symmetric
# ----------------------------------------------------------------------
def test_hedge_forms_and_base_basis():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: float == dict == HedgeByDepth == inference; "
          "base-basis pin re-hedges too")
    print(SEPARATOR)
    z = _zeroed()
    frames = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frames['infer'], _ = apply_pins(
            z, [Pin('r4', 213_340.0, basis='cascaded')])
        frames['float'], _ = apply_pins(
            z, [Pin('r4', 213_340.0, basis='cascaded')], hedge=1.1)
        frames['dict'], _ = apply_pins(
            z, [Pin('r4', 213_340.0, basis='cascaded')],
            hedge={0: 1.0, 1: 1.1, 2: 1.21})
        frames['hbd'], _ = apply_pins(
            z, [Pin('r4', 213_340.0, basis='cascaded')],
            hedge=HedgeByDepth(from_leaves={1: 1.1, 2: 1.1}))
    ref = frames['infer'].set_index('node_id')['base_quota']
    for name, fr in frames.items():
        got = fr.set_index('node_id')['base_quota']
        assert ((got - ref).abs() < 0.05).all(), name
    # base-basis: cascaded derived at 1.21, not 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g, _ = apply_pins(z, [Pin('r4', 100_000.0)])
    ixg = g.set_index('node_id')
    assert abs(ixg.loc['r4', 'cascaded_quota'] - 121_000.0) < 0.5
    print("  four ratio sources identical; base-basis casc = x1.21")


# ----------------------------------------------------------------------
# 3. Fallback chain + silence for untouched 0/0 rows
# ----------------------------------------------------------------------
def test_fallbacks_and_silence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: all-siblings-zero -> hedge= or 1.0+warning; "
          "untouched zero rows never warn")
    print(SEPARATOR)
    # hand frame: T's children BOTH zero (no sibling ratio available)
    df = pd.DataFrame([
        dict(node_id='T', parent=None, depth=0, base_quota=0.0,
             cascaded_quota=0.0),
        dict(node_id='a', parent='T', depth=1, base_quota=0.0,
             cascaded_quota=0.0),
        dict(node_id='b', parent='T', depth=1, base_quota=0.0,
             cascaded_quota=0.0)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        e, _ = apply_pins(df, [Pin('a', 100.0, basis='cascaded')],
                          hedge={1: 1.1})
    ix = e.set_index('node_id')
    assert abs(ix.loc['a', 'base_quota'] - 100.0 / 1.1) < 0.05
    # no hedge, nothing to infer from -> 1.0, warned with source
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter('always')
        e2, _ = apply_pins(df, [Pin('a', 100.0, basis='cascaded')])
    ix2 = e2.set_index('node_id')
    assert abs(ix2.loc['a', 'base_quota'] - 100.0) < 0.05
    m = [str(x.message) for x in w2 if 'zero-baseline' in str(x.message)]
    assert m and ('none' in m[0] or 'same-depth' in m[0])
    # a zeroed row that STAYS zero is silent
    z = _zeroed()
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter('always')
        _e, _ = apply_pins(z, [Pin('r1', 250_000.0)])
    assert not [x for x in w3 if 'zero-baseline' in str(x.message)]
    print("  hedge= authoritative; loud 1.0 fallback; no noise for "
          "untouched rows")


# ----------------------------------------------------------------------
# 4. #62 receipts: allocation is already mix-preserving
# ----------------------------------------------------------------------
def test_62_receipts():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: Pin(node, total) scales the CURRENT 70/30 mix -> "
          "770/330 (the #62 'want' line, pinned)")
    print(SEPARATOR)
    hdf = pd.DataFrame([dict(region='EMEA', team='T', rep=r, kw=k)
                        for r, k in [('r1', 700), ('r2', 300)]])
    targets = pd.DataFrame([dict(region='EMEA', product=p, tgt=1000.0)
                            for p in ('A', 'B')])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(hdf, targets,
                            group_keys=['region', 'product'],
                            target_col='tgt',
                            taxonomy=['region', 'team', 'rep'],
                            metrics=KW, hedge_multiplier=1.1)
        e, _ = apply_pins(q, [Pin('r1', 700.0, scope={'product': 'A'}),
                              Pin('r1', 300.0, scope={'product': 'B'})])
        f, _ = apply_pins(e, [Pin('r1', 1100.0)])
    ix = f.set_index(['node_id', 'product'])
    assert abs(ix.loc[('r1', 'A'), 'base_quota'] - 770.0) < 0.05
    assert abs(ix.loc[('r1', 'B'), 'base_quota'] - 330.0) < 0.05
    assert reconcile(f, hedge=1.1)['ok'].all()
    # the filer's 'observed' drift reproduces only via OVERLAP (#63):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g, _ = apply_pins(e, [Pin('r1', 1100.0),
                              Pin('r1', 0.0, scope={'product': 'B'})])
    ixg = g.set_index(['node_id', 'product'])
    tot = (ixg.loc[('r1', 'A'), 'base_quota']
           + ixg.loc[('r1', 'B'), 'base_quota'])
    assert abs(tot - 770.0) < 0.05      # under-delivered: overlap, not
    print("  770/330 exact; overlap composition under-delivers to 770 "
          "(that's #63's bug, documented)")


if __name__ == '__main__':
    test_sibling_inference_fixes_67()
    test_hedge_forms_and_base_basis()
    test_fallbacks_and_silence()
    test_62_receipts()

    print(f"\n\n{SEPARATOR}")
    print("ALL ZERO-BASELINE-RATIO TESTS PASSED")
    print(SEPARATOR)
