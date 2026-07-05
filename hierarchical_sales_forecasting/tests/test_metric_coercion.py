"""
Tests for issue #3 — non-numeric metric columns silently aggregate to 0.

Covers:
  - boolean / numpy-scalar / string gate columns now carry signal
  - numeric strings ("1,200", "$500", "12.5%") coerced on ingest
  - uncoercible values -> warning + treated as missing (never a silent 0)
  - aggregation-side warn-once for garbage stored via add_node()
  - legacy '_Attainment' path applies the same coercion
  - boolean auto-detection (no zero-imputation) still works post-coercion
"""
import warnings
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
)

SEPARATOR = "=" * 90
TAXONOMY = ['regional', 'node_3_region', 'node_4_team', 'node_5_rep_no']
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['knowledge_workers'])]
DC_GATE = [MetricSpec('dc', columns=['dc_seats'])]


def _cascade(h):
    c = QuotaCascader(h)
    q = c.cascade_quota('S_AMER', 1_000_000.0, metrics=KW,
                        gate_metrics=DC_GATE, verbose=False)
    return c, q


# ----------------------------------------------------------------------
# 1. Issue #3 repro — string bools in the gate column
# ----------------------------------------------------------------------
def test_issue3_string_bool_gate_column():
    print(SEPARATOR)
    print("TEST 1: Issue #3 — gate column with 'true'/'false' strings carries "
          "signal (was silent $0 for everyone)")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(regional='S_AMER', node_3_region='E', node_4_team='T1',
             node_5_rep_no='r1', knowledge_workers=100, dc_seats='true'),
        dict(regional='S_AMER', node_3_region='W', node_4_team='T2',
             node_5_rep_no='r2', knowledge_workers=80, dc_seats='false'),
    ])
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY,
                     metrics_cols=['knowledge_workers', 'dc_seats'])
    # Stored as real bools now
    assert h.graph.nodes['r1']['dc_seats'] is True
    assert h.graph.nodes['r2']['dc_seats'] is False
    c, q = _cascade(h)
    print(f"  r1 (entitled):     ${q['r1']:,.2f} (expected $1,000,000)")
    print(f"  r2 (not entitled): ${q['r2']:,.2f} (expected $0, gated)")
    assert abs(q['r1'] - 1_000_000.0) < 0.01
    assert q['r2'] == 0.0
    assert 'r2' in c.gated_nodes and not c.gate_relaxed_nodes


# ----------------------------------------------------------------------
# 2. numpy scalars (np.bool_, np.int64) — the DB/manual-ingest path
# ----------------------------------------------------------------------
def test_numpy_scalars_coerced():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: numpy scalars (np.bool_, np.int64) coerced instead of "
          "reading as 0")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('S_AMER'); h.add_node('E'); h.add_node('W')
    h.add_node('r1', {'knowledge_workers': np.int64(100),
                      'dc_seats': np.bool_(True)})
    h.add_node('r2', {'knowledge_workers': np.float32(80.0),
                      'dc_seats': np.bool_(False)})
    h.add_edge('S_AMER', 'E'); h.add_edge('S_AMER', 'W')
    h.add_edge('E', 'r1'); h.add_edge('W', 'r2')
    c, q = _cascade(h)
    print(f"  r1: ${q['r1']:,.2f} (np.bool_(True) gate passed)")
    print(f"  r2: ${q['r2']:,.2f} (np.bool_(False) -> gated $0)")
    assert abs(q['r1'] - 1_000_000.0) < 0.01
    assert q['r2'] == 0.0
    # np.int64 knowledge_workers aggregated correctly (not 0)
    assert c._aggregate_node_metric('r1', KW[0]) == 100.0


# ----------------------------------------------------------------------
# 3. Numeric strings with formatting — "1,200", "$500", "12.5%"
# ----------------------------------------------------------------------
def test_formatted_numeric_strings():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: '1,200' / '$500' / '12.5%' strings coerced to numbers on "
          "ingest")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(regional='S_AMER', node_3_region='E', node_4_team='T1',
             node_5_rep_no='r1', knowledge_workers='1,200', dc_seats='$500'),
        dict(regional='S_AMER', node_3_region='W', node_4_team='T2',
             node_5_rep_no='r2', knowledge_workers='12.5%', dc_seats='0'),
    ])
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=TAXONOMY,
                     metrics_cols=['knowledge_workers', 'dc_seats'])
    assert h.graph.nodes['r1']['knowledge_workers'] == 1200.0
    assert h.graph.nodes['r1']['dc_seats'] == 500.0
    assert h.graph.nodes['r2']['knowledge_workers'] == 12.5
    print("  '1,200' -> 1200.0 · '$500' -> 500.0 · '12.5%' -> 12.5")


# ----------------------------------------------------------------------
# 4. Uncoercible values — warning + missing, never silent
# ----------------------------------------------------------------------
def test_uncoercible_warns_and_treated_missing():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: uncoercible cell ('N/A - pending') warns and is treated "
          "as missing")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(regional='S_AMER', node_3_region='E', node_4_team='T1',
             node_5_rep_no='r1', knowledge_workers=100, dc_seats='N/A - pending'),
        dict(regional='S_AMER', node_3_region='W', node_4_team='T2',
             node_5_rep_no='r2', knowledge_workers=80, dc_seats=40),
    ])
    h = SalesHierarchy()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        h.from_dataframe(df, path_cols=TAXONOMY,
                         metrics_cols=['knowledge_workers', 'dc_seats'])
    msgs = [str(w.message) for w in wlog]
    assert any('could not be coerced' in m and 'dc_seats' in m for m in msgs)
    # Value is MISSING (attribute absent), not stored as garbage or 0
    assert 'dc_seats' not in h.graph.nodes['r1']
    assert h.graph.nodes['r2']['dc_seats'] == 40
    print(f"  warning emitted: {[m for m in msgs if 'coerced' in m][0][:90]}...")


# ----------------------------------------------------------------------
# 5. Aggregation-side guard — garbage stored via add_node warns ONCE/column
# ----------------------------------------------------------------------
def test_aggregation_warns_once_per_column():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: garbage stored via add_node() warns once per column at "
          "aggregation time")
    print(SEPARATOR)
    h = SalesHierarchy()
    h.add_node('Root')
    h.add_node('x1', {'dc_seats': 'high'})
    h.add_node('x2', {'dc_seats': 'medium', 'knowledge_workers': 10})
    h.add_edge('Root', 'x1'); h.add_edge('Root', 'x2')
    c = QuotaCascader(h)
    spec = MetricSpec('dc', columns=['dc_seats'])
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        total = c._aggregate_node_metric('Root', spec)
        c._aggregate_node_metric('Root', spec)  # second call — no new warning
    dc_warnings = [w for w in wlog if 'dc_seats' in str(w.message)]
    print(f"  aggregate: {total} · warnings for dc_seats: {len(dc_warnings)} "
          f"(expected exactly 1)")
    assert total == 0.0
    assert len(dc_warnings) == 1


# ----------------------------------------------------------------------
# 6. Legacy '_Attainment' path applies the same coercion
# ----------------------------------------------------------------------
def test_legacy_attainment_coercion():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: legacy path — '_Attainment' strings like '1,000' coerced")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC': ['IC_A', 'IC_B'],
        'Q1_Attainment': ['1,000', '3,000'],   # strings with commas
    })
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'IC'],
                     metrics_cols=['Q1_Attainment'])
    c = QuotaCascader(h)
    q = c.cascade_quota('Corp', 400_000.0)     # legacy single-metric path
    print(f"  IC_A: ${q['IC_A']:,.2f} (expected $100,000)")
    print(f"  IC_B: ${q['IC_B']:,.2f} (expected $300,000)")
    assert abs(q['IC_A'] - 100_000.0) < 0.01
    assert abs(q['IC_B'] - 300_000.0) < 0.01


# ----------------------------------------------------------------------
# 7. Boolean auto-detection survives coercion (no zero-imputation)
# ----------------------------------------------------------------------
def test_boolean_autodetect_after_coercion():
    print(f"\n\n{SEPARATOR}")
    print("TEST 7: 'true'/'false' strings become bools, so boolean "
          "auto-detection still skips zero-imputation")
    print(SEPARATOR)
    df = pd.DataFrame([
        dict(Global='Corp', IC='IC_A', Q1_Cert='true', Q2_Cert='false',
             Q3_Cert='false', Q4_Cert='false'),
        dict(Global='Corp', IC='IC_B', Q1_Cert='true', Q2_Cert='true',
             Q3_Cert='true', Q4_Cert='true'),
    ])
    cols = ['Q1_Cert', 'Q2_Cert', 'Q3_Cert', 'Q4_Cert']
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'IC'], metrics_cols=cols)
    c = QuotaCascader(h)
    spec = MetricSpec('Cert', direction='proportional', weight=1.0,
                      lookback=4, impute_zeros=True)
    # If imputation wrongly kicked in, IC_A's three False quarters would be
    # inflated to 1 each (sum 4 == IC_B). Boolean detection must keep 1 vs 4.
    a = c._aggregate_node_metric('IC_A', spec)
    b = c._aggregate_node_metric('IC_B', spec)
    print(f"  IC_A cert sum: {a} (expected 1.0) · IC_B: {b} (expected 4.0)")
    assert a == 1.0 and b == 4.0


if __name__ == '__main__':
    test_issue3_string_bool_gate_column()
    test_numpy_scalars_coerced()
    test_formatted_numeric_strings()
    test_uncoercible_warns_and_treated_missing()
    test_aggregation_warns_once_per_column()
    test_legacy_attainment_coercion()
    test_boolean_autodetect_after_coercion()

    print(f"\n\n{SEPARATOR}")
    print("ALL METRIC-COERCION TESTS PASSED")
    print(SEPARATOR)
