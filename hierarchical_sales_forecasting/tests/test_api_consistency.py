"""
Tests for issue #5 — consistent graph accessor (.graph everywhere) and
read-only hierarchy helpers.
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import SalesHierarchy, QuotaCascader

SEPARATOR = "=" * 90


def _h():
    df = pd.DataFrame([
        dict(Global='Corp', Region='NA', Mgr='M1', IC='a', Q1_Attainment=10),
        dict(Global='Corp', Region='NA', Mgr='M1', IC='b', Q1_Attainment=30),
        dict(Global='Corp', Region='EMEA', Mgr='M2', IC='c', Q1_Attainment=60),
    ])
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'Region', 'Mgr', 'IC'],
                     metrics_cols=['Q1_Attainment'])
    return h


# ----------------------------------------------------------------------
# 1. Issue #5 repro — hierarchy.hierarchy no longer AttributeErrors
# ----------------------------------------------------------------------
def test_graph_accessor_consistency():
    print(SEPARATOR)
    print("TEST 1: .graph is canonical everywhere; .hierarchy aliases work")
    print(SEPARATOR)
    h = _h()
    c = QuotaCascader(h)
    # The exact confusion from the issue: both names now resolve, to the
    # SAME underlying nx.DiGraph object.
    assert h.hierarchy is h.graph                 # was AttributeError
    assert c.graph is h.graph                     # canonical name on cascader
    assert c.hierarchy is h.graph                 # pre-0.7.2 name still works
    print("  h.graph is h.hierarchy is c.graph is c.hierarchy — one object")


# ----------------------------------------------------------------------
# 2. Read-only helpers
# ----------------------------------------------------------------------
def test_readonly_helpers():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: roots() / leaves() / managers() / node_depths()")
    print(SEPARATOR)
    h = _h()
    assert h.roots() == ['Corp']
    assert sorted(h.leaves()) == ['a', 'b', 'c']
    assert sorted(h.leaves('M1')) == ['a', 'b']
    assert sorted(h.managers()) == ['Corp', 'EMEA', 'M1', 'M2', 'NA']
    assert sorted(h.managers('NA')) == ['M1']
    depths = h.node_depths()
    assert depths['Corp'] == 0 and depths['NA'] == 1
    assert depths['M2'] == 2 and depths['a'] == 3
    print(f"  roots={h.roots()} · leaves={sorted(h.leaves())} · "
          f"managers under NA={h.managers('NA')}")
    print(f"  depths: {depths}")


# ----------------------------------------------------------------------
# 3. Cascade behavior unchanged after the rename
# ----------------------------------------------------------------------
def test_cascade_unchanged():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: legacy cascade still exact after internal rename")
    print(SEPARATOR)
    h = _h()
    c = QuotaCascader(h)
    q = c.cascade_quota('Corp', 1_000_000.0)
    print(f"  a: ${q['a']:,.2f} · b: ${q['b']:,.2f} · c: ${q['c']:,.2f}")
    assert abs(q['a'] - 100_000.0) < 0.01
    assert abs(q['b'] - 300_000.0) < 0.01
    assert abs(q['c'] - 600_000.0) < 0.01


if __name__ == '__main__':
    test_graph_accessor_consistency()
    test_readonly_helpers()
    test_cascade_unchanged()

    print(f"\n\n{SEPARATOR}")
    print("ALL API-CONSISTENCY TESTS PASSED")
    print(SEPARATOR)
