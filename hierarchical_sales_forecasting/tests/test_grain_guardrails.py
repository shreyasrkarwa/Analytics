"""
Tests for issue #36 — grain-mismatch guardrail: warn when a metric's
values are identical across (nearly) every leaf-sibling group, the
signature of an ancestor-level value repeated onto leaf rows.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import SalesHierarchy, QuotaCascader, MetricSpec

SEPARATOR = "=" * 90


def _build(rows):
    h = SalesHierarchy()
    cols = [c for c in rows[0] if c not in ('regional', 'team', 'rep')]
    h.from_dataframe(pd.DataFrame(rows), path_cols=['regional', 'team', 'rep'],
                     metrics_cols=cols)
    return QuotaCascader(h)


def _rows(dc_values):
    """4 reps in 2 teams; kw varies (healthy), dc per the given list."""
    kws = [100, 200, 300, 400]
    return [dict(regional='EMEA', team=f'T{i//2+1}', rep=f'r{i+1}',
                 kw=kws[i], dc_seats=dc_values[i]) for i in range(4)]


def _cascade(c, **kw):
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        c.cascade_quota('EMEA', 1_000_000.0, verbose=False, **kw)
    return [str(w.message) for w in wlog
            if 'coarser grain' in str(w.message)]


DC = [MetricSpec('dc_seats', direction='proportional', weight=1.0,
                 columns=['dc_seats'])]
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


# ----------------------------------------------------------------------
# 1. Region-level value repeated onto leaves -> warning + equal split
# ----------------------------------------------------------------------
def test_repeated_ancestor_value_warns():
    print(SEPARATOR)
    print("TEST 1: region-level dc_seats repeated per rep — detector fires; "
          "sibling shares collapse to equal split")
    print(SEPARATOR)
    # Same value within each team (repeated from team/region level),
    # different across teams — the issue's exact signature.
    c = _build(_rows([500, 500, 800, 800]))
    msgs = _cascade(c, metrics=DC)
    assert len(msgs) == 1 and 'dc_seats' in msgs[0]
    # The symptom the warning prevents: identical siblings
    q = c.last_quotas
    assert abs(q['r1'] - q['r2']) < 0.01     # equal within T1
    print(f"  warned: {msgs[0][:90]}...")
    print(f"  symptom shown: r1 == r2 == ${q['r1']:,.2f}")


# ----------------------------------------------------------------------
# 2. True leaf-grain metric -> no warning
# ----------------------------------------------------------------------
def test_leaf_grain_no_warning():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: per-leaf values — no warning")
    print(SEPARATOR)
    c = _build(_rows([10, 40, 25, 75]))
    msgs = _cascade(c, metrics=DC)
    assert msgs == []
    print("  varied per leaf -> silent, as it should be")


# ----------------------------------------------------------------------
# 3. Boolean / 0-1 metrics exempt
# ----------------------------------------------------------------------
def test_boolean_exempt():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: boolean flags constant within teams — exempt")
    print(SEPARATOR)
    rows = _rows([1, 1, 1, 1])
    c = _build(rows)
    msgs = _cascade(c, metrics=KW, gate_metrics=[
        MetricSpec('dc_seats', columns=['dc_seats'], gate_mode='truthy')])
    assert msgs == []
    print("  all-1 flag gate -> no grain warning")


# ----------------------------------------------------------------------
# 4. Gate metrics are checked too (the dc_seats incident was a gate)
# ----------------------------------------------------------------------
def test_gate_metric_checked():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: repeated-grain GATE metric also warns")
    print(SEPARATOR)
    c = _build(_rows([500, 500, 800, 800]))
    msgs = _cascade(c, metrics=KW, gate_metrics=[
        MetricSpec('dc_seats', columns=['dc_seats'])])
    assert len(msgs) == 1 and 'dc_seats' in msgs[0]
    print("  gate spec flagged")


# ----------------------------------------------------------------------
# 5. Warn once per metric per cascader; healthy metric never named
# ----------------------------------------------------------------------
def test_warn_once_and_scoped():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: once per metric per cascader; kw (healthy) never named")
    print(SEPARATOR)
    c = _build(_rows([500, 500, 800, 800]))
    msgs1 = _cascade(c, metrics=DC + KW)
    msgs2 = _cascade(c, metrics=DC + KW)     # second cascade, same cascader
    assert len(msgs1) == 1 and 'dc_seats' in msgs1[0]
    assert 'kw' not in msgs1[0]
    assert msgs2 == []
    print("  1 warning total across 2 cascades; only dc_seats named")


# ----------------------------------------------------------------------
# 6. Tiny trees (fewer than 2 sibling groups) stay silent
# ----------------------------------------------------------------------
def test_tiny_tree_silent():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: a single sibling group can't establish a pattern — "
          "silent")
    print(SEPARATOR)
    rows = [dict(regional='EMEA', team='T1', rep=f'r{i}', kw=100,
                 dc_seats=500) for i in range(3)]
    c = _build(rows)
    msgs = _cascade(c, metrics=DC)
    assert msgs == []
    print("  1 group of constant values -> no verdict, no warning")


if __name__ == '__main__':
    test_repeated_ancestor_value_warns()
    test_leaf_grain_no_warning()
    test_boolean_exempt()
    test_gate_metric_checked()
    test_warn_once_and_scoped()
    test_tiny_tree_silent()

    print(f"\n\n{SEPARATOR}")
    print("ALL GRAIN-GUARDRAIL TESTS PASSED")
    print(SEPARATOR)
