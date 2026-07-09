"""
Tests for issue #34 — cascade_proportional: the deterministic
"proportional-to-metric" front door (sugar over fixed-weight specs).
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    HedgeByDepth,
)

SEPARATOR = "=" * 90


def _h(rows):
    h = SalesHierarchy()
    h.from_dataframe(pd.DataFrame(rows), path_cols=['regional', 'team', 'rep'],
                     metrics_cols=[c for c in rows[0]
                                   if c not in ('regional', 'team', 'rep')])
    return h


N2 = [dict(regional='EMEA', team='T1', rep='r1', dc_seats=300, cloud=100),
      dict(regional='EMEA', team='T1', rep='r2', dc_seats=700, cloud=100)]


# ----------------------------------------------------------------------
# 1. The issue's headline: n=2, 30% of seats -> 30% of quota, one line
# ----------------------------------------------------------------------
def test_single_metric_one_liner():
    print(SEPARATOR)
    print("TEST 1: cascade_proportional(metric='dc_seats') — deterministic "
          "at n=2, no statistics, no warnings")
    print(SEPARATOR)
    c = QuotaCascader(_h(N2))
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning fails the test
        q = c.cascade_proportional('EMEA', 1_000_000.0, metric='dc_seats',
                                   verbose=False)
    print(f"  r1: ${q['r1']:,.2f} (30% of seats) · r2: ${q['r2']:,.2f}")
    assert abs(q['r1'] - 300_000.0) < 0.01
    assert abs(q['r2'] - 700_000.0) < 0.01
    # Exactly equivalent to the manual fixed-weight spec (the issue's
    # Option A, which has always worked)
    c2 = QuotaCascader(_h(N2))
    q2 = c2.cascade_quota('EMEA', 1_000_000.0, metrics=[
        MetricSpec('dc_seats', direction='proportional', weight=1.0,
                   columns=['dc_seats'])], verbose=False)
    assert q == q2


# ----------------------------------------------------------------------
# 2. Fixed blend == manual specs
# ----------------------------------------------------------------------
def test_blend_equivalence():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: metrics={'dc_seats': 1.0, 'cloud': 0.5} == manual "
          "fixed-weight specs")
    print(SEPARATOR)
    q = QuotaCascader(_h(N2)).cascade_proportional(
        'EMEA', 1_000_000.0, metrics={'dc_seats': 1.0, 'cloud': 0.5},
        verbose=False)
    q2 = QuotaCascader(_h(N2)).cascade_quota('EMEA', 1_000_000.0, metrics=[
        MetricSpec('dc_seats', direction='proportional', weight=1.0),
        MetricSpec('cloud', direction='proportional', weight=0.5)],
        verbose=False)
    assert q == q2
    # blend math: (2/3)*seat_share + (1/3)*cloud_share
    expected_r1 = 1_000_000.0 * ((2 / 3) * 0.3 + (1 / 3) * 0.5)
    print(f"  r1: ${q['r1']:,.2f} (expected ${expected_r1:,.2f})")
    assert abs(q['r1'] - expected_r1) < 0.01


# ----------------------------------------------------------------------
# 3. Everything passes through: gates, HedgeByDepth, base layer
# ----------------------------------------------------------------------
def test_kwargs_passthrough():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: gates + HedgeByDepth + reconciliation flow through")
    print(SEPARATOR)
    rows = N2 + [dict(regional='EMEA', team='T2', rep='r3', dc_seats=0,
                      cloud=500)]
    c = QuotaCascader(_h(rows))
    q = c.cascade_proportional(
        'EMEA', 1_000_000.0, metric='cloud',
        gate_metrics=[MetricSpec('dc_seats', columns=['dc_seats'])],
        hedge_multiplier=HedgeByDepth(from_leaves={1: 1.10}),
        verbose=False)
    assert q['r3'] == 0.0 and 'r3' in c.gated_nodes        # gated
    assert abs(q['r1'] - c.base_quotas['r1'] * 1.10) < 0.5  # hedged
    rep = c.reconciliation_report(c.base_quotas, target=1_000_000.0,
                                  strict=True)
    assert rep['reconciles'].all()
    print("  r3 gated · hedge applied · base reconciles at every depth")


# ----------------------------------------------------------------------
# 4. Qi_<name> convention resolves too (v0.7.1 order)
# ----------------------------------------------------------------------
def test_quarterly_convention():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: quarterly Qi_<name> columns work without explicit "
          "columns=")
    print(SEPARATOR)
    rows = [dict(regional='EMEA', team='T1', rep='r1',
                 Q1_acv=10, Q2_acv=10, Q3_acv=10, Q4_acv=10),
            dict(regional='EMEA', team='T1', rep='r2',
                 Q1_acv=30, Q2_acv=30, Q3_acv=30, Q4_acv=30)]
    q = QuotaCascader(_h(rows)).cascade_proportional(
        'EMEA', 400_000.0, metric='acv', verbose=False)
    print(f"  r1: ${q['r1']:,.2f} (expected $100,000 via Q1..Q4 sums)")
    assert abs(q['r1'] - 100_000.0) < 0.01
    assert abs(q['r2'] - 300_000.0) < 0.01


# ----------------------------------------------------------------------
# 5. direction='inverse' + validation
# ----------------------------------------------------------------------
def test_inverse_and_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: inverse direction · argument validation")
    print(SEPARATOR)
    c = QuotaCascader(_h(N2))
    q = c.cascade_proportional('EMEA', 1_000_000.0, metric='dc_seats',
                               direction='inverse', verbose=False)
    assert q['r1'] > q['r2']            # fewer seats -> bigger share
    print(f"  inverse: r1 ${q['r1']:,.2f} > r2 ${q['r2']:,.2f}")
    for kwargs, frag in [
        (dict(), 'exactly one'),
        (dict(metric='a', metrics={'b': 1.0}), 'exactly one'),
        (dict(metrics={}), 'empty'),
        (dict(metrics={'a': 0}), 'positive'),
        (dict(metric='a', direction='up'), 'direction'),
    ]:
        try:
            c.cascade_proportional('EMEA', 1.0, verbose=False, **kwargs)
            raise AssertionError(f'expected ValueError for {kwargs}')
        except ValueError as e:
            assert frag in str(e), (kwargs, str(e))
            print(f"  {kwargs} -> rejected")


if __name__ == '__main__':
    test_single_metric_one_liner()
    test_blend_equivalence()
    test_kwargs_passthrough()
    test_quarterly_convention()
    test_inverse_and_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL CASCADE-PROPORTIONAL TESTS PASSED")
    print(SEPARATOR)
