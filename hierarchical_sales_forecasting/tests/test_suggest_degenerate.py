"""
Tests for issue #8 — suggest_weights on small / degenerate slices.

Verifies (a) the long-standing graceful degradation (no exceptions,
weight 0 + rationale), and (b) the v0.10.2 courtesy warning when EVERY
candidate degrades to 0 — so the downstream equal-split fallback is
never a silent surprise.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
)

SEPARATOR = "=" * 90
CAND = [{'name': 'dc', 'column': 'dc', 'direction': 'proportional'}]


def _suggest(df):
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        specs, report = MetricSpec.suggest_weights(
            df, target_column='kw', candidate_metrics=CAND)
    heads_up = [w for w in wlog
                if 'ALL suggested weights are 0' in str(w.message)]
    return specs, report, heads_up


# ----------------------------------------------------------------------
# 1. Degenerate slices: no exception, weight 0, ONE heads-up warning
# ----------------------------------------------------------------------
def test_degenerate_slices_warn_once():
    print(SEPARATOR)
    print("TEST 1: degenerate slices never raise; all-zero result emits ONE "
          "equal-split heads-up")
    print(SEPARATOR)
    cases = {
        'single row':      pd.DataFrame({'kw': [100], 'dc': [5]}),
        'zero variance':   pd.DataFrame({'kw': [100] * 3, 'dc': [5] * 3}),
        'all-null target': pd.DataFrame({'kw': [None] * 3, 'dc': [5, 6, 7]}),
        'empty df':        pd.DataFrame({'kw': [], 'dc': []}),
    }
    for label, df in cases.items():
        specs, report, heads_up = _suggest(df)
        assert specs[0].weight == 0.0, label
        assert 'rationale' in report['dc'], label
        assert len(heads_up) == 1, f"{label}: expected 1 heads-up warning"
        assert 'EQUAL SPLIT' in str(heads_up[0].message)
        print(f"  {label:>16}: weight=0, warned once "
              f"({report['dc']['rationale'][:48]}...)")


# ----------------------------------------------------------------------
# 2. Healthy slices: no heads-up
# ----------------------------------------------------------------------
def test_healthy_slice_no_warning():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: healthy slice — usable weight, NO heads-up warning")
    print(SEPARATOR)
    df = pd.DataFrame({'kw': [100, 200, 300, 400], 'dc': [1, 2, 3, 4]})
    specs, _, heads_up = _suggest(df)
    print(f"  weight: {specs[0].weight:.3f} · heads-up warnings: {len(heads_up)}")
    assert specs[0].weight > 0
    assert heads_up == []


# ----------------------------------------------------------------------
# 3. Mixed outcome (one zero, one usable): no heads-up
# ----------------------------------------------------------------------
def test_partial_zero_no_warning():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: one candidate degrades but another survives — no heads-up")
    print(SEPARATOR)
    df = pd.DataFrame({'kw': [100, 200, 300, 400],
                       'dc': [1, 2, 3, 4],
                       'flat': [7, 7, 7, 7]})       # constant -> weight 0
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        specs, _ = MetricSpec.suggest_weights(
            df, target_column='kw',
            candidate_metrics=CAND + [{'name': 'flat', 'column': 'flat',
                                       'direction': 'proportional'}])
    weights = {s.name: s.weight for s in specs}
    heads_up = [w for w in wlog if 'ALL suggested weights' in str(w.message)]
    print(f"  weights: {weights} · heads-up: {len(heads_up)}")
    assert weights['dc'] > 0 and weights['flat'] == 0.0
    assert heads_up == []


# ----------------------------------------------------------------------
# 4. End-to-end: all-zero specs cascade as an equal split, no crash
# ----------------------------------------------------------------------
def test_all_zero_specs_equal_split():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: cascading with all-zero specs -> clean equal split")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(r='AMER', team='T1', rep='r1', kw=100, dc=5),
        dict(r='AMER', team='T1', rep='r2', kw=100, dc=5),
    ])
    specs, _, _ = _suggest(df.rename(columns={'kw': 'kw', 'dc': 'dc'})
                           .assign(kw=[100, 100]))   # zero variance
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['r', 'team', 'rep'],
                     metrics_cols=['kw', 'dc'])
    c = QuotaCascader(h)
    q = c.cascade_quota('AMER', 1_000_000.0, metrics=specs, verbose=False)
    print(f"  r1: ${q['r1']:,.2f} · r2: ${q['r2']:,.2f} (equal split)")
    assert abs(q['r1'] - 500_000.0) < 0.01
    assert abs(q['r2'] - 500_000.0) < 0.01


# ----------------------------------------------------------------------
# 5. Missing target column still raises (config error, not data)
# ----------------------------------------------------------------------
def test_missing_target_still_raises():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: missing target column still raises (typo should be loud)")
    print(SEPARATOR)
    try:
        MetricSpec.suggest_weights(pd.DataFrame({'dc': [1, 2, 3]}),
                                   target_column='kw',
                                   candidate_metrics=CAND)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'target_column' in str(e)
        print(f"  Raised as expected: {e}")


if __name__ == '__main__':
    test_degenerate_slices_warn_once()
    test_healthy_slice_no_warning()
    test_partial_zero_no_warning()
    test_all_zero_specs_equal_split()
    test_missing_target_still_raises()

    print(f"\n\n{SEPARATOR}")
    print("ALL DEGENERATE-SUGGEST TESTS PASSED")
    print(SEPARATOR)
