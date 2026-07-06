"""
Tests for issue #6 — MetricSpec.name vs columns coupling.

Covers:
  - specs returned by suggest_weights are directly usable when the metric
    name IS the data column (no `spec.columns = [spec.name]` boilerplate)
  - explicit columns= always wins (fallback never interferes)
  - the Qi_<name> convention is preferred over the fallback when present
  - tree-wide zero-signal metrics warn loudly (no more silent no-ops)
  - gate metrics benefit from the same fallback
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
TAXONOMY = ['regional', 'sub_region', 'team', 'territory']


def _flat_df():
    """Single-column metrics named exactly like the data columns."""
    return pd.DataFrame([
        dict(regional='AMER', sub_region='W', team='W1', territory='r1',
             knowledge_workers=100, dc_seats=10),
        dict(regional='AMER', sub_region='W', team='W2', territory='r2',
             knowledge_workers=300, dc_seats=0),
        dict(regional='AMER', sub_region='E', team='E1', territory='r3',
             knowledge_workers=600, dc_seats=5),
    ])


# ----------------------------------------------------------------------
# 1. Issue #6 repro — suggest_weights output cascades correctly as-is
# ----------------------------------------------------------------------
def test_suggested_specs_directly_usable():
    print(SEPARATOR)
    print("TEST 1: Issue #6 — suggest_weights output usable WITHOUT "
          "spec.columns = [spec.name]")
    print(SEPARATOR)
    df = _flat_df()
    suggested, _ = MetricSpec.suggest_weights(
        df,
        target_column='knowledge_workers',
        candidate_metrics=[
            {'name': 'dc_seats', 'column': 'dc_seats',
             'direction': 'proportional'},
        ],
        warn_on_direction_mismatch=False,
    )
    # The returned spec has columns=None (unset) — the old failure setup
    assert suggested[0].columns is None
    specs = [MetricSpec('knowledge_workers', direction='proportional',
                        weight=1.0),   # columns unset too: relies on fallback
             *suggested]

    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY,
                     metrics_cols=['knowledge_workers', 'dc_seats'])
    c = QuotaCascader(h)
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        q = c.cascade_quota('AMER', 1_000_000.0, metrics=specs, verbose=False)
    # No zero-signal warnings — both metrics found their columns via fallback
    assert not any('ZERO signal' in str(w.message) for w in wlog), \
        [str(w.message) for w in wlog]
    # knowledge_workers signal actually drove shares (r3 has 60% of kw)
    print(f"  r1: ${q['r1']:,.2f} · r2: ${q['r2']:,.2f} · r3: ${q['r3']:,.2f}")
    assert q['r3'] > q['r2'] > q['r1']
    # Pre-fix, no signal -> equal split at every level; assert NOT equal
    assert abs(q['r1'] - q['r3']) > 1_000


# ----------------------------------------------------------------------
# 2. Explicit columns= always wins — fallback never interferes
# ----------------------------------------------------------------------
def test_explicit_columns_win():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: explicit columns= wins even when an attribute named "
          "<name> also exists")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('Root')
    # 'kw' attr says 999 (decoy); the configured column says 10 vs 30
    h.add_node('a', {'kw': 999, 'kw_clean': 10})
    h.add_node('b', {'kw': 999, 'kw_clean': 30})
    h.add_edge('Root', 'a'); h.add_edge('Root', 'b')
    c = QuotaCascader(h)
    spec = MetricSpec('kw', direction='proportional', weight=1.0,
                      columns=['kw_clean'])
    assert c._aggregate_node_metric('a', spec) == 10.0
    q = c.cascade_quota('Root', 400_000.0, metrics=[spec], verbose=False)
    print(f"  a: ${q['a']:,.2f} (expected $100,000) · b: ${q['b']:,.2f}")
    assert abs(q['a'] - 100_000.0) < 0.01
    assert abs(q['b'] - 300_000.0) < 0.01


# ----------------------------------------------------------------------
# 3. Qi_<name> convention preferred over the plain-<name> fallback
# ----------------------------------------------------------------------
def test_qi_convention_preferred():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: when Qi_<name> columns exist, the fallback does NOT fire")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('Root')
    h.add_node('leaf', {'Q1_kw': 5, 'Q2_kw': 5, 'kw': 999})
    h.add_edge('Root', 'leaf')
    c = QuotaCascader(h)
    spec = MetricSpec('kw', direction='proportional', weight=1.0, lookback=2)
    val = c._aggregate_node_metric('leaf', spec)
    print(f"  aggregated: {val} (expected 10.0 from Q1+Q2, not 999)")
    assert val == 10.0


# ----------------------------------------------------------------------
# 4. Tree-wide zero signal warns loudly, naming the columns tried
# ----------------------------------------------------------------------
def test_zero_signal_warns():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: a typo'd metric name warns instead of silently "
          "contributing nothing")
    print(SEPARATOR)
    df = _flat_df()
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY,
                     metrics_cols=['knowledge_workers', 'dc_seats'])
    c = QuotaCascader(h)
    specs = [
        MetricSpec('knowledge_workers', direction='proportional', weight=1.0),
        MetricSpec('knowledge_wrokers',  # typo — no such column anywhere
                   direction='proportional', weight=0.5),
    ]
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        c.cascade_quota('AMER', 1_000_000.0, metrics=specs, verbose=False)
    msgs = [str(w.message) for w in wlog if 'ZERO signal' in str(w.message)]
    assert len(msgs) == 1 and 'knowledge_wrokers' in msgs[0]
    print(f"  warning: {msgs[0][:100]}...")


# ----------------------------------------------------------------------
# 5. Gate metrics get the same fallback
# ----------------------------------------------------------------------
def test_gate_metric_fallback():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: gate spec with columns unset resolves via the "
          "plain-<name> fallback")
    print(SEPARATOR)
    df = _flat_df()
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY,
                     metrics_cols=['knowledge_workers', 'dc_seats'])
    c = QuotaCascader(h)
    q = c.cascade_quota(
        'AMER', 1_000_000.0,
        metrics=[MetricSpec('knowledge_workers', direction='proportional',
                            weight=1.0)],
        gate_metrics=[MetricSpec('dc_seats')],   # columns unset — fallback
        verbose=False,
    )
    print(f"  r2 (dc_seats=0): ${q['r2']:,.2f} (expected $0, gated)")
    assert q['r2'] == 0.0 and 'r2' in c.gated_nodes
    assert q['r1'] > 0 and q['r3'] > 0


if __name__ == '__main__':
    test_suggested_specs_directly_usable()
    test_explicit_columns_win()
    test_qi_convention_preferred()
    test_zero_signal_warns()
    test_gate_metric_fallback()

    print(f"\n\n{SEPARATOR}")
    print("ALL METRIC-COLUMN-FALLBACK TESTS PASSED")
    print(SEPARATOR)
