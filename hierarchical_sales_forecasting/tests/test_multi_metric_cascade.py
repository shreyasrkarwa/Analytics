"""
Tests for multi-metric cascading, MetricSpec helpers, brand-new IC
handling via CSV column, boolean / arbitrary-name metric support, and
the multi-column PipelineAdjuster extension.
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    PipelineAdjuster,
    MetricSpec,
    GateAllocationError,
)

SEPARATOR = "=" * 90


def _build_simple_hierarchy(df, path_cols, metrics_cols, brand_new_col=None):
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=path_cols, metrics_cols=metrics_cols,
                     brand_new_col=brand_new_col)
    return h


# ----------------------------------------------------------------------
# 1. Backward compat
# ----------------------------------------------------------------------
def test_backward_compat():
    print(SEPARATOR)
    print("TEST 1: Backward compat — metrics=None matches legacy path")
    print(SEPARATOR)

    df = pd.DataFrame({
        'Global': ['Corp'] * 3,
        'IC':     ['IC_A', 'IC_B', 'IC_C'],
        'Q1_Attainment': [100.0, 200.0, 300.0],
        'Q2_Attainment': [110.0, 210.0, 310.0],
        'Q3_Attainment': [120.0, 220.0, 320.0],
        'Q4_Attainment': [130.0, 230.0, 330.0],
    })
    h = _build_simple_hierarchy(df, ['Global', 'IC'],
                                ['Q1_Attainment', 'Q2_Attainment',
                                 'Q3_Attainment', 'Q4_Attainment'])
    cascader = QuotaCascader(h)
    quotas = cascader.cascade_quota('Corp', 1000.0)

    expected_a = 1000.0 * 460.0 / 2580.0
    expected_b = 1000.0 * 860.0 / 2580.0
    expected_c = 1000.0 * 1260.0 / 2580.0
    ok = (abs(quotas['IC_A'] - expected_a) < 0.01
          and abs(quotas['IC_B'] - expected_b) < 0.01
          and abs(quotas['IC_C'] - expected_c) < 0.01)
    print(f"  Legacy path unchanged? {'OK' if ok else 'FAIL'}")
    assert ok


# ----------------------------------------------------------------------
# 2. Single proportional MetricSpec matches legacy
# ----------------------------------------------------------------------
def test_single_proportional_metric_matches_legacy():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: Single proportional MetricSpec ≡ legacy single-metric path")
    print(SEPARATOR)

    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        'Q1_NetNewACV': [100.0, 400.0],
        'Q2_NetNewACV': [200.0, 300.0],
        'Q3_NetNewACV': [300.0, 200.0],
        'Q4_NetNewACV': [400.0, 100.0],
    })
    h = _build_simple_hierarchy(df, ['Global', 'IC'],
                                ['Q1_NetNewACV', 'Q2_NetNewACV',
                                 'Q3_NetNewACV', 'Q4_NetNewACV'])
    cascader = QuotaCascader(h)
    metrics = [MetricSpec(name='NetNewACV', direction='proportional',
                          weight=1.0, lookback=4)]
    quotas = cascader.cascade_quota('Corp', 1000.0, metrics=metrics)

    ok = abs(quotas['IC_A'] - 500.0) < 0.01 and abs(quotas['IC_B'] - 500.0) < 0.01
    print(f"  Even split when 4Q sums tie? {'OK' if ok else 'FAIL'}")
    assert ok


# ----------------------------------------------------------------------
# 3. Multi-metric blend with inverse direction (worked example)
# ----------------------------------------------------------------------
def test_blend_with_inverse_direction():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: Multi-metric blend with inverse direction (worked example)")
    print(SEPARATOR)

    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        'Q1_NetNewACV': [250_000, 125_000],
        'Q2_NetNewACV': [250_000, 125_000],
        'Q3_NetNewACV': [250_000, 125_000],
        'Q4_NetNewACV': [250_000, 125_000],
        'Q1_CloudSeats': [12, 25],
        'Q2_CloudSeats': [12, 25],
        'Q3_CloudSeats': [13, 25],
        'Q4_CloudSeats': [13, 25],
        'LTM_ExpansionSpent': [200_000, 50_000],
    })
    metrics_cols = [
        'Q1_NetNewACV', 'Q2_NetNewACV', 'Q3_NetNewACV', 'Q4_NetNewACV',
        'Q1_CloudSeats', 'Q2_CloudSeats', 'Q3_CloudSeats', 'Q4_CloudSeats',
        'LTM_ExpansionSpent',
    ]
    h = _build_simple_hierarchy(df, ['Global', 'IC'], metrics_cols)
    cascader = QuotaCascader(h)

    metrics = [
        MetricSpec('NetNewACV',  direction='proportional', weight=1.0, lookback=4),
        MetricSpec('CloudSeats', direction='proportional', weight=0.5, lookback=4),
        MetricSpec('ExpansionSpent', direction='inverse', weight=0.7,
                   columns=['LTM_ExpansionSpent']),
    ]
    quotas = cascader.cascade_quota('Corp', 1_000_000.0, metrics=metrics)

    w_acv, w_cloud, w_exp = 1.0/2.2, 0.5/2.2, 0.7/2.2
    exp_share_a = (w_acv * (1.0/1.5)
                   + w_cloud * (50.0/150.0)
                   + w_exp * (1.0/(200_000 + 2_000) /
                              (1.0/(200_000 + 2_000) + 1.0/(50_000 + 2_000))))
    expected_a = 1_000_000.0 * exp_share_a

    ok_order = quotas['IC_B'] > quotas['IC_A']
    ok_math = abs(quotas['IC_A'] - expected_a) < 1.0
    print(f"  IC_B > IC_A?         {'OK' if ok_order else 'FAIL'}")
    print(f"  Math matches manual? {'OK' if ok_math else 'FAIL'}")
    assert ok_order and ok_math


# ----------------------------------------------------------------------
# 4. Brand-new IC handling — explicit-list / CSV-column / rule, with mutex
# ----------------------------------------------------------------------
def test_brand_new_ic_csv_column():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: Brand-new IC via CSV column (brand_new_col)")
    print(SEPARATOR)

    df = pd.DataFrame({
        'Global': ['Corp'] * 4,
        'Mgr':    ['Mgr'] * 4,
        'IC':     ['IC_Veteran_1', 'IC_Veteran_2', 'IC_NewHire', 'IC_AlsoNew'],
        'Q1_NetNewACV': [100_000, 200_000, 0, 50_000],
        'Q2_NetNewACV': [100_000, 200_000, 0, 50_000],
        'Q3_NetNewACV': [100_000, 200_000, 0, 50_000],
        'Q4_NetNewACV': [100_000, 200_000, 0, 50_000],
        # Brand-new flag in the CSV. Note IC_AlsoNew has $200K of history
        # but is STILL marked brand-new by the analyst (maybe transferred
        # in from another team) — the explicit flag wins.
        'Is_Brand_New': [False, False, True, 'yes'],
    })
    h = _build_simple_hierarchy(
        df,
        path_cols=['Global', 'Mgr', 'IC'],
        metrics_cols=['Q1_NetNewACV', 'Q2_NetNewACV',
                      'Q3_NetNewACV', 'Q4_NetNewACV'],
        brand_new_col='Is_Brand_New',
    )
    cascader = QuotaCascader(h)
    metrics = [MetricSpec('NetNewACV', direction='proportional',
                          weight=1.0, lookback=4)]

    # Use the CSV-column path; rule must NOT be passed simultaneously.
    quotas = cascader.cascade_quota(
        'Corp', 100_000.0, metrics=metrics, new_ic_attr='_is_brand_new'
    )
    equal_share = 100_000.0 / 4
    print(f"  IC_NewHire   (zero ACV, flagged):     "
          f"${quotas['IC_NewHire']:,.2f}  (expected ${equal_share:,.2f})")
    print(f"  IC_AlsoNew   (has $200K, but flagged): "
          f"${quotas['IC_AlsoNew']:,.2f}  (expected ${equal_share:,.2f})")
    print(f"  IC_Veteran_1 (not flagged):           "
          f"${quotas['IC_Veteran_1']:,.2f}")
    assert abs(quotas['IC_NewHire'] - equal_share) < 1.0
    assert abs(quotas['IC_AlsoNew'] - equal_share) < 1.0
    # Veterans should NOT get the equal share — they split the remainder
    # proportionally by their NetNewACV.
    assert abs(quotas['IC_Veteran_1'] - equal_share) > 1.0


def test_brand_new_rule_mutex():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: Either-or mutex — explicit path + rule together raises")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        'Q1_NetNewACV': [100.0, 0.0],
        'Q2_NetNewACV': [100.0, 0.0],
        'Q3_NetNewACV': [100.0, 0.0],
        'Q4_NetNewACV': [100.0, 0.0],
    })
    h = _build_simple_hierarchy(df, ['Global', 'IC'],
                                ['Q1_NetNewACV', 'Q2_NetNewACV',
                                 'Q3_NetNewACV', 'Q4_NetNewACV'])
    cascader = QuotaCascader(h)
    metrics = [MetricSpec('NetNewACV', direction='proportional',
                          weight=1.0, lookback=4)]

    raised = False
    try:
        cascader.cascade_quota('Corp', 1000.0, metrics=metrics,
                               new_ic_ids=['IC_B'],
                               new_ic_rule='all_metrics_zero')
    except ValueError as e:
        raised = True
        print(f"  Raised ValueError as expected: {str(e)[:100]}...")
    print(f"  Either-or enforced? {'OK' if raised else 'FAIL'}")
    assert raised


def test_brand_new_rule_default_when_no_explicit():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: Default rule (all_metrics_zero) when no explicit path")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 3,
        'Mgr':    ['Mgr'] * 3,
        'IC':     ['IC_A', 'IC_B', 'IC_New'],
        'Q1_NetNewACV': [100.0, 200.0, 0.0],
        'Q2_NetNewACV': [100.0, 200.0, 0.0],
        'Q3_NetNewACV': [100.0, 200.0, 0.0],
        'Q4_NetNewACV': [100.0, 200.0, 0.0],
    })
    h = _build_simple_hierarchy(df, ['Global', 'Mgr', 'IC'],
                                ['Q1_NetNewACV', 'Q2_NetNewACV',
                                 'Q3_NetNewACV', 'Q4_NetNewACV'])
    cascader = QuotaCascader(h)
    metrics = [MetricSpec('NetNewACV', direction='proportional',
                          weight=1.0, lookback=4)]
    quotas = cascader.cascade_quota('Corp', 300.0, metrics=metrics)
    # IC_New (all zero) gets equal share of 100; remainder 200 split 1:2
    print(f"  IC_New gets ${quotas['IC_New']:.2f} (expected $100.00)")
    print(f"  IC_A   gets ${quotas['IC_A']:.2f}")
    print(f"  IC_B   gets ${quotas['IC_B']:.2f}")
    assert abs(quotas['IC_New'] - 100.0) < 0.01


# ----------------------------------------------------------------------
# 7. Arbitrary metric names — handle any string, not just NetNewACV / CloudSeats
# ----------------------------------------------------------------------
def test_arbitrary_metric_names():
    print(f"\n\n{SEPARATOR}")
    print("TEST 7: Any metric name (Customer_Sat, MQLs_Sourced, ...) works")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        # Bizarre names with spaces-replaced, mixed case, dashes
        'Q1_Customer_Sat_Score':       [70, 90],
        'Q2_Customer_Sat_Score':       [72, 88],
        'Q3_Customer_Sat_Score':       [75, 85],
        'Q4_Customer_Sat_Score':       [78, 82],
        'Q1_MQLs_Sourced_via_Outbound':[120, 30],
        'Q2_MQLs_Sourced_via_Outbound':[130, 28],
        'Q3_MQLs_Sourced_via_Outbound':[125, 32],
        'Q4_MQLs_Sourced_via_Outbound':[140, 30],
    })
    cols = [c for c in df.columns if c.startswith('Q')]
    h = _build_simple_hierarchy(df, ['Global', 'IC'], cols)
    cascader = QuotaCascader(h)

    metrics = [
        MetricSpec('Customer_Sat_Score',        direction='proportional',
                   weight=1.0, lookback=4),
        MetricSpec('MQLs_Sourced_via_Outbound', direction='proportional',
                   weight=1.0, lookback=4),
    ]
    quotas = cascader.cascade_quota('Corp', 1000.0, metrics=metrics)
    print(f"  IC_A gets ${quotas['IC_A']:.2f}")
    print(f"  IC_B gets ${quotas['IC_B']:.2f}")
    print(f"  Sum  = ${quotas['IC_A'] + quotas['IC_B']:.2f} (expected $1000.00)")
    assert abs(quotas['IC_A'] + quotas['IC_B'] - 1000.0) < 0.01


# ----------------------------------------------------------------------
# 8. Boolean / 0-1 metric data type
# ----------------------------------------------------------------------
def test_boolean_metric():
    print(f"\n\n{SEPARATOR}")
    print("TEST 8: Boolean metric (e.g., Has_Active_Cert): no zero-imputation")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        # IC_A had certs in 2 of 4 quarters; IC_B in 4 of 4.
        # With imputation enabled by default, IC_A would falsely become
        # [1, 1, 1, 1] (avg of its non-zero = 1) -> sum 4 instead of 2.
        'Q1_Has_Active_Cert': [True,  True],
        'Q2_Has_Active_Cert': [False, True],
        'Q3_Has_Active_Cert': [True,  True],
        'Q4_Has_Active_Cert': [False, True],
        # A proportional dollar metric so the blend isn't trivial
        'Q1_NetNewACV': [100_000, 100_000],
        'Q2_NetNewACV': [100_000, 100_000],
        'Q3_NetNewACV': [100_000, 100_000],
        'Q4_NetNewACV': [100_000, 100_000],
    })
    cols = [c for c in df.columns if c.startswith('Q')]
    h = _build_simple_hierarchy(df, ['Global', 'IC'], cols)
    cascader = QuotaCascader(h)

    metrics = [
        MetricSpec('NetNewACV',       direction='proportional',
                   weight=1.0, lookback=4),
        MetricSpec('Has_Active_Cert', direction='proportional',
                   weight=1.0, lookback=4),
    ]
    quotas = cascader.cascade_quota('Corp', 1000.0, metrics=metrics)
    # Expected: NetNewACV ties (50/50). Has_Active_Cert: IC_A=2, IC_B=4 ->
    # shares 1/3 and 2/3. Blend w/equal weights -> A = (0.5 + 1/3)/2 =
    # 0.4167; B = (0.5 + 2/3)/2 = 0.5833. NOT imputed; if imputation had
    # fired, A would also be 0.5 and result would be 50/50.
    print(f"  IC_A gets ${quotas['IC_A']:.2f} (expected ~$416.67)")
    print(f"  IC_B gets ${quotas['IC_B']:.2f} (expected ~$583.33)")
    expected_a = (0.5 + 1.0/3.0) / 2 * 1000.0
    assert abs(quotas['IC_A'] - expected_a) < 1.0


# ----------------------------------------------------------------------
# 9. suggest_weights: direction is user-input; sign-mismatch warning fires
# ----------------------------------------------------------------------
def test_suggest_weights_keeps_user_direction():
    print(f"\n\n{SEPARATOR}")
    print("TEST 9: suggest_weights preserves user direction; warns on mismatch")
    print(SEPARATOR)
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    cloud = rng.uniform(10, 200, n)
    dc    = rng.uniform(0, 100, n)
    exp_  = rng.uniform(0, 500_000, n)
    acv = 2000 * cloud - 1500 * dc - 0.3 * exp_ + rng.normal(0, 50_000, n)

    df = pd.DataFrame({
        'NetNewACV_4Q_sum':  acv,
        'CloudSeats_4Q_sum': cloud,
        'DCSeats_4Q_sum':    dc,
        'LTM_ExpansionSpent': exp_,
    })

    # User says DC seats are PROPORTIONAL even though data clearly disagrees.
    # The package should warn and KEEP the user's direction.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        suggestions, report = MetricSpec.suggest_weights(
            df,
            target_column='NetNewACV_4Q_sum',
            candidate_metrics=[
                {'name': 'CloudSeats', 'column': 'CloudSeats_4Q_sum',
                 'direction': 'proportional', 'lookback': 4},
                {'name': 'DCSeats', 'column': 'DCSeats_4Q_sum',
                 'direction': 'proportional', 'lookback': 4},  # user is "wrong"
                {'name': 'ExpansionSpent', 'column': 'LTM_ExpansionSpent',
                 'columns': ['LTM_ExpansionSpent'],
                 'direction': 'inverse', 'lookback': 1},
            ],
        )
        dc_warning = any(
            'DCSeats' in str(warning.message) for warning in w
        )

    by_name = {s.name: s for s in suggestions}
    print(f"  CloudSeats direction = {by_name['CloudSeats'].direction} "
          f"(user said proportional, data agrees)")
    print(f"  DCSeats    direction = {by_name['DCSeats'].direction} "
          f"(user said proportional, data DISAGREES; warning fired? "
          f"{'yes' if dc_warning else 'no'})")
    print(f"  ExpansionSpent direction = {by_name['ExpansionSpent'].direction} "
          f"(user said inverse, data agrees)")

    assert by_name['CloudSeats'].direction == 'proportional'
    assert by_name['DCSeats'].direction == 'proportional'  # preserved!
    assert by_name['ExpansionSpent'].direction == 'inverse'
    assert dc_warning  # warning fired


def test_suggest_weights_requires_direction():
    print(f"\n\n{SEPARATOR}")
    print("TEST 10: suggest_weights rejects missing direction")
    print(SEPARATOR)
    df = pd.DataFrame({'target': [1, 2, 3], 'X': [1, 2, 3]})
    raised = False
    try:
        MetricSpec.suggest_weights(
            df, target_column='target',
            candidate_metrics=[{'name': 'X', 'column': 'X'}],  # no direction
        )
    except ValueError as e:
        raised = True
        print(f"  Raised ValueError: {str(e)[:90]}...")
    assert raised, "Missing direction should raise"


def test_suggest_directions_and_weights_still_available():
    print(f"\n\n{SEPARATOR}")
    print("TEST 11: suggest_directions_and_weights (exploratory) infers both")
    print(SEPARATOR)
    import numpy as np
    rng = np.random.default_rng(42)
    n = 100
    cloud = rng.uniform(10, 200, n)
    dc    = rng.uniform(0, 100, n)
    acv = 2000 * cloud - 1500 * dc + rng.normal(0, 50_000, n)
    df = pd.DataFrame({
        'target': acv,
        'CloudSeats_4Q_sum': cloud,
        'DCSeats_4Q_sum': dc,
    })
    sugg, _ = MetricSpec.suggest_directions_and_weights(
        df, target_column='target',
        candidate_metrics=[
            {'name': 'CloudSeats', 'column': 'CloudSeats_4Q_sum'},
            {'name': 'DCSeats',    'column': 'DCSeats_4Q_sum'},
        ],
    )
    by = {s.name: s for s in sugg}
    print(f"  CloudSeats inferred direction: {by['CloudSeats'].direction}")
    print(f"  DCSeats    inferred direction: {by['DCSeats'].direction}")
    assert by['CloudSeats'].direction == 'proportional'
    assert by['DCSeats'].direction == 'inverse'


# ----------------------------------------------------------------------
# 12. PipelineAdjuster: pipeline_attr accepts a list of columns
# ----------------------------------------------------------------------
def test_pipeline_adjuster_multi_column():
    print(f"\n\n{SEPARATOR}")
    print("TEST 12: PipelineAdjuster sums multiple pipeline columns")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'Mgr':    ['Mgr'] * 2,
        'IC':     ['IC_A', 'IC_B'],
        'Q1_Attainment': [100, 200],
        'Q2_Attainment': [100, 200],
        'Q3_Attainment': [100, 200],
        'Q4_Attainment': [100, 200],
        # Multi-source pipeline: open + late-stage commit + best-case
        'Open_Pipeline':       [500_000, 1_000_000],
        'Late_Stage_Commit':   [200_000, 300_000],
        'Best_Case_Adds':      [100_000, 200_000],
    })
    cols = [c for c in df.columns if c not in ('Global', 'Mgr', 'IC')]
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'Mgr', 'IC'], metrics_cols=cols)

    cascader = QuotaCascader(h)
    quotas = cascader.cascade_quota('Corp', 900_000.0)

    # Single column (backward compat)
    adj_single = PipelineAdjuster(h, quotas, pipeline_attr='Open_Pipeline')
    pipe_a_single = adj_single._get_node_pipeline('IC_A')
    print(f"  Single-col Open_Pipeline for IC_A:  "
          f"${pipe_a_single:,.2f} (expected $500,000)")
    assert pipe_a_single == 500_000.0

    # Multi-column list — sums the three
    adj_multi = PipelineAdjuster(
        h, quotas,
        pipeline_attr=['Open_Pipeline', 'Late_Stage_Commit', 'Best_Case_Adds'],
    )
    pipe_a_multi = adj_multi._get_node_pipeline('IC_A')
    pipe_b_multi = adj_multi._get_node_pipeline('IC_B')
    pipe_mgr_multi = adj_multi._get_node_pipeline('Mgr')
    print(f"  Multi-col total for IC_A:  ${pipe_a_multi:,.2f} (expected $800,000)")
    print(f"  Multi-col total for IC_B:  ${pipe_b_multi:,.2f} (expected $1,500,000)")
    print(f"  Multi-col total for Mgr:   ${pipe_mgr_multi:,.2f} (expected $2,300,000 — rolls up)")
    assert pipe_a_multi == 800_000.0
    assert pipe_b_multi == 1_500_000.0
    assert pipe_mgr_multi == 2_300_000.0


# ----------------------------------------------------------------------
# 13. End-to-end on synthetic_multi_metric.csv (refreshed)
# ----------------------------------------------------------------------
def test_end_to_end_with_synthetic_csv():
    print(f"\n\n{SEPARATOR}")
    print("TEST 13: End-to-end on synthetic_multi_metric.csv (user directions)")
    print(SEPARATOR)
    try:
        df = pd.read_csv('tests/data/synthetic_multi_metric.csv', keep_default_na=False)
    except FileNotFoundError:
        print("  Skipping — generate the dataset first.")
        return

    taxonomy = ['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC']
    metrics_cols = [
        'Q1_NetNewACV', 'Q2_NetNewACV', 'Q3_NetNewACV', 'Q4_NetNewACV',
        'Q1_CloudSeats', 'Q2_CloudSeats', 'Q3_CloudSeats', 'Q4_CloudSeats',
        'Q1_DCSeats', 'Q2_DCSeats', 'Q3_DCSeats', 'Q4_DCSeats',
        'LTM_ExpansionSpent', 'Current_Pipeline',
    ]
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=taxonomy, metrics_cols=metrics_cols)

    cascader = QuotaCascader(h)

    # The user declares directions up front; the package only suggests weights.
    suggestions, report = MetricSpec.suggest_weights(
        df,
        target_column='NetNewACV_4Q_sum',
        candidate_metrics=[
            {'name': 'CloudSeats', 'column': 'CloudSeats_4Q_sum',
             'direction': 'proportional', 'lookback': 4},
            {'name': 'DCSeats', 'column': 'DCSeats_4Q_sum',
             'direction': 'inverse', 'lookback': 4},
            {'name': 'ExpansionSpent', 'column': 'LTM_ExpansionSpent',
             'columns': ['LTM_ExpansionSpent'],
             'direction': 'inverse', 'lookback': 1},
        ],
    )
    suggestions.insert(0, MetricSpec('NetNewACV',
                                     direction='proportional',
                                     weight=1.0, lookback=4))

    print("  Suggested weights (directions are user-input):")
    for s in suggestions:
        if s.name in report:
            r = report[s.name]
            print(f"    {s.name:>16s}  direction={s.direction:<13s} "
                  f"weight={s.weight:.3f}  (corr={r['correlation']:+.3f})")
        else:
            print(f"    {s.name:>16s}  direction={s.direction:<13s} "
                  f"weight={s.weight:.3f}  (user-pinned)")

    quotas = cascader.cascade_quota(
        'Global_Corp', 50_000_000.0,
        hedge_multiplier=1.05,
        metrics=suggestions,
    )
    leaves = [n for n in h.graph.nodes if h.graph.out_degree(n) == 0]
    ic_total = sum(quotas[n] for n in leaves)
    print(f"\n  $50M cascaded to {len(leaves)} ICs, IC sum = ${ic_total:,.2f}")
    assert ic_total > 50_000_000.0
    assert all(quotas[n] >= 0 for n in leaves)


# ----------------------------------------------------------------------
# 14. Gate metric: a single gate zeroes a node and redistributes share
# ----------------------------------------------------------------------
def test_gate_metric_basic():
    print(f"\n\n{SEPARATOR}")
    print("TEST 14: Gate metric zeroes a node; siblings absorb the share")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 3,
        'Mgr':    ['Mgr'] * 3,
        'IC':     ['IC_A', 'IC_B', 'IC_Gated'],
        'Q1_NetNewACV': [100_000, 100_000, 100_000],
        'Q2_NetNewACV': [100_000, 100_000, 100_000],
        'Q3_NetNewACV': [100_000, 100_000, 100_000],
        'Q4_NetNewACV': [100_000, 100_000, 100_000],
        'Unmigrated_Seats': [50, 50, 0],   # IC_Gated has no white space
    })
    cols = [c for c in df.columns if c.startswith('Q')] + ['Unmigrated_Seats']
    h = _build_simple_hierarchy(df, ['Global', 'Mgr', 'IC'], cols)
    cascader = QuotaCascader(h)

    quotas = cascader.cascade_quota(
        'Corp', 300_000.0,
        metrics=[MetricSpec('NetNewACV', direction='proportional',
                            weight=1.0, lookback=4)],
        gate_metrics=[MetricSpec('Unmigrated_Seats',
                                  columns=['Unmigrated_Seats'])],
        verbose=False,
    )
    print(f"  IC_A:     ${quotas['IC_A']:,.2f} (expected $150,000)")
    print(f"  IC_B:     ${quotas['IC_B']:,.2f} (expected $150,000)")
    print(f"  IC_Gated: ${quotas['IC_Gated']:,.2f} (expected $0)")
    print(f"  Cascader gated_nodes: {cascader.gated_nodes}")
    assert quotas['IC_Gated'] == 0.0
    # IC_A and IC_B split the full $300K (gated share goes to them)
    assert abs(quotas['IC_A'] - 150_000.0) < 0.01
    assert abs(quotas['IC_B'] - 150_000.0) < 0.01
    assert 'IC_Gated' in cascader.gated_nodes


# ----------------------------------------------------------------------
# 15. Gate propagates upward — whole subtree gets $0
# ----------------------------------------------------------------------
def test_gate_propagates_upward():
    print(f"\n\n{SEPARATOR}")
    print("TEST 15: Gate propagates — whole subtree (Mgr + ICs) gets $0")
    print(SEPARATOR)
    # Two managers; Mgr_Empty's ICs ALL have 0 unmigrated seats, so the
    # manager itself should also be gated and get $0.
    df = pd.DataFrame({
        'Global': ['Corp'] * 4,
        'Mgr':    ['Mgr_Full', 'Mgr_Full', 'Mgr_Empty', 'Mgr_Empty'],
        'IC':     ['IC_F1', 'IC_F2', 'IC_E1', 'IC_E2'],
        'Q1_NetNewACV': [100_000] * 4,
        'Q2_NetNewACV': [100_000] * 4,
        'Q3_NetNewACV': [100_000] * 4,
        'Q4_NetNewACV': [100_000] * 4,
        'Unmigrated_Seats': [50, 80, 0, 0],
    })
    cols = [c for c in df.columns if c.startswith('Q')] + ['Unmigrated_Seats']
    h = _build_simple_hierarchy(df, ['Global', 'Mgr', 'IC'], cols)
    cascader = QuotaCascader(h)

    quotas = cascader.cascade_quota(
        'Corp', 1_000_000.0,
        metrics=[MetricSpec('NetNewACV', direction='proportional',
                            weight=1.0, lookback=4)],
        gate_metrics=[MetricSpec('Unmigrated_Seats',
                                  columns=['Unmigrated_Seats'])],
        verbose=False,
    )
    print(f"  Mgr_Full:  ${quotas['Mgr_Full']:,.2f} (expected $1,000,000)")
    print(f"  Mgr_Empty: ${quotas['Mgr_Empty']:,.2f} (expected $0 — whole team gated)")
    print(f"  IC_F1:     ${quotas['IC_F1']:,.2f}")
    print(f"  IC_F2:     ${quotas['IC_F2']:,.2f}")
    print(f"  IC_E1:     ${quotas['IC_E1']:,.2f} (expected $0)")
    print(f"  IC_E2:     ${quotas['IC_E2']:,.2f} (expected $0)")
    assert quotas['Mgr_Empty'] == 0.0
    assert quotas['IC_E1'] == 0.0
    assert quotas['IC_E2'] == 0.0
    # Mgr_Full absorbs Mgr_Empty's share too
    assert abs(quotas['Mgr_Full'] - 1_000_000.0) < 0.01
    assert 'Mgr_Empty' in cascader.gated_nodes


# ----------------------------------------------------------------------
# 16. Multiple gates compose with AND
# ----------------------------------------------------------------------
def test_gates_compose_with_and():
    print(f"\n\n{SEPARATOR}")
    print("TEST 16: Multiple gates compose with AND")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 4,
        'Mgr':    ['Mgr'] * 4,
        'IC':     ['IC_PassBoth', 'IC_FailA', 'IC_FailB', 'IC_FailBoth'],
        'Q1_NetNewACV': [100_000] * 4,
        'Q2_NetNewACV': [100_000] * 4,
        'Q3_NetNewACV': [100_000] * 4,
        'Q4_NetNewACV': [100_000] * 4,
        'Gate_A': [10, 0, 10,  0],
        'Gate_B': [10, 10, 0,  0],
    })
    cols = [c for c in df.columns if c.startswith('Q')] + ['Gate_A', 'Gate_B']
    h = _build_simple_hierarchy(df, ['Global', 'Mgr', 'IC'], cols)
    cascader = QuotaCascader(h)

    quotas = cascader.cascade_quota(
        'Corp', 400_000.0,
        metrics=[MetricSpec('NetNewACV', direction='proportional',
                            weight=1.0, lookback=4)],
        gate_metrics=[
            MetricSpec('Gate_A', columns=['Gate_A']),
            MetricSpec('Gate_B', columns=['Gate_B']),
        ],
        verbose=False,
    )
    # Only IC_PassBoth survives -> gets full $400K
    print(f"  IC_PassBoth: ${quotas['IC_PassBoth']:,.2f} (expected $400,000)")
    print(f"  IC_FailA:    ${quotas['IC_FailA']:,.2f} (expected $0)")
    print(f"  IC_FailB:    ${quotas['IC_FailB']:,.2f} (expected $0)")
    print(f"  IC_FailBoth: ${quotas['IC_FailBoth']:,.2f} (expected $0)")
    assert abs(quotas['IC_PassBoth'] - 400_000.0) < 0.01
    assert quotas['IC_FailA'] == 0.0
    assert quotas['IC_FailB'] == 0.0
    assert quotas['IC_FailBoth'] == 0.0


# ----------------------------------------------------------------------
# 17. CRO override wins over gate
# ----------------------------------------------------------------------
def test_cro_override_wins_over_gate():
    print(f"\n\n{SEPARATOR}")
    print("TEST 17: CRO override wins over gate (explicit business override)")
    print(SEPARATOR)
    df = pd.DataFrame({
        'Global': ['Corp'] * 2,
        'Mgr':    ['Mgr'] * 2,
        'IC':     ['IC_Normal', 'IC_Gated_But_Override'],
        'Q1_NetNewACV': [100_000, 100_000],
        'Q2_NetNewACV': [100_000, 100_000],
        'Q3_NetNewACV': [100_000, 100_000],
        'Q4_NetNewACV': [100_000, 100_000],
        'Unmigrated_Seats': [50, 0],  # the second IC fails the gate
    })
    cols = [c for c in df.columns if c.startswith('Q')] + ['Unmigrated_Seats']
    h = _build_simple_hierarchy(df, ['Global', 'Mgr', 'IC'], cols)
    cascader = QuotaCascader(h)

    quotas = cascader.cascade_quota(
        'Corp', 500_000.0,
        metrics=[MetricSpec('NetNewACV', direction='proportional',
                            weight=1.0, lookback=4)],
        gate_metrics=[MetricSpec('Unmigrated_Seats',
                                  columns=['Unmigrated_Seats'])],
        new_ic_overrides={'IC_Gated_But_Override': 200_000.0},
        verbose=False,
    )
    # CRO pinned IC_Gated_But_Override at $200K; IC_Normal gets the rest
    print(f"  IC_Gated_But_Override: ${quotas['IC_Gated_But_Override']:,.2f} (expected $200,000 — CRO wins)")
    print(f"  IC_Normal:             ${quotas['IC_Normal']:,.2f} (expected $300,000)")
    assert abs(quotas['IC_Gated_But_Override'] - 200_000.0) < 0.01
    assert abs(quotas['IC_Normal'] - 300_000.0) < 0.01


# ----------------------------------------------------------------------
# Issue #12 fixtures — 4-level hierarchy (regional > sub-region > team >
# territory) mirroring the Enterprise_AMER report, with one fully-gated
# team (West1: all territories have 0 DC seats).
# ----------------------------------------------------------------------
def _issue12_hierarchy(all_gated=False):
    dc = 0 if all_gated else 1
    df = pd.DataFrame({
        'Regional':  ['Enterprise_AMER'] * 8,
        'SubRegion': ['West'] * 5 + ['East'] * 3,
        'Team':      ['West1', 'West1', 'West2', 'West3', 'West3',
                      'East2', 'East2', 'East6'],
        'Territory': ['West1_1', 'West1_2', 'West2_1', 'West3_1', 'West3_2',
                      'East2_1', 'East2_6', 'East6_2'],
        'Q1_Revenue': [100_000] * 8,
        'Q2_Revenue': [100_000] * 8,
        'DC_Seats':   [0, 0, 40 * dc, 25 * dc, 25 * dc,
                       30 * dc, 30 * dc, 50 * dc],  # West1 fully gated
    })
    cols = ['Q1_Revenue', 'Q2_Revenue', 'DC_Seats']
    return _build_simple_hierarchy(
        df, ['Regional', 'SubRegion', 'Team', 'Territory'], cols)


def _depth_sums(cascader, quotas):
    depths = cascader._node_depths()
    sums = {}
    for node, q in quotas.items():
        sums[depths[node]] = sums.get(depths[node], 0.0) + q
    return sums


_ISSUE12_METRICS = [MetricSpec('Revenue', direction='proportional',
                               weight=1.0, lookback=2)]
_ISSUE12_GATES = [MetricSpec('DC_Seats', columns=['DC_Seats'])]


# ----------------------------------------------------------------------
# 18. Issue #12: gated subtree redistributes; every depth sums to target
# ----------------------------------------------------------------------
def test_issue12_gated_subtree_reconciles_at_every_depth():
    print(f"\n\n{SEPARATOR}")
    print("TEST 18: Issue #12 — gated team's share redistributes; "
          "sum(depth d) == target for all d")
    print(SEPARATOR)
    h = _issue12_hierarchy()
    cascader = QuotaCascader(h)
    target = 1_000_000.0
    quotas = cascader.cascade_quota(
        'Enterprise_AMER', target,
        metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
        verbose=False,
    )
    sums = _depth_sums(cascader, quotas)
    for d in range(4):
        print(f"  depth {d} sum: ${sums[d]:,.2f}")
        assert abs(sums[d] - target) < 0.01, f"depth {d} does not reconcile"
    # West1 (fully gated) gets $0; its share went to sibling teams
    assert quotas['West1'] == 0.0
    assert quotas['West1_1'] == 0.0 and quotas['West1_2'] == 0.0
    assert quotas['West2'] > 0 and quotas['West3'] > 0
    # reconciliation_report agrees (strict mode must not raise)
    report = cascader.reconciliation_report(quotas, target=target, strict=True)
    assert report['reconciles'].all()
    print("  reconciliation_report: all depths reconcile")


# ----------------------------------------------------------------------
# 19. Issue #12: fully-gated ROOT no longer strands the target
# ----------------------------------------------------------------------
def test_issue12_fully_gated_root_redistributes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 19: Issue #12 — fully-gated root (0 eligible reps anywhere) "
          "still lands the target on ICs")
    print(SEPARATOR)
    h = _issue12_hierarchy(all_gated=True)  # NO territory has DC seats
    cascader = QuotaCascader(h)
    target = 100_000.0
    quotas = cascader.cascade_quota(
        'Enterprise_AMER', target,
        metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
        verbose=False,
    )
    sums = _depth_sums(cascader, quotas)
    for d in range(4):
        print(f"  depth {d} sum: ${sums[d]:,.2f}")
        assert abs(sums[d] - target) < 0.01, f"depth {d} short (stranded target)"
    # Root was gated but never zeroed
    assert quotas['Enterprise_AMER'] == target
    # Fallback was recorded — nothing silent
    assert cascader.gate_relaxed_nodes
    assert cascader.unallocated == 0.0
    print(f"  gate_relaxed_nodes: {len(cascader.gate_relaxed_nodes)} node(s)")


# ----------------------------------------------------------------------
# 20. Issue #12: gate_fallback='strand_at_root' reports unallocated
# ----------------------------------------------------------------------
def test_issue12_strand_at_root_reports_unallocated():
    print(f"\n\n{SEPARATOR}")
    print("TEST 20: Issue #12 — gate_fallback='strand_at_root' keeps target "
          "at root and exposes unallocated")
    print(SEPARATOR)
    h = _issue12_hierarchy(all_gated=True)
    cascader = QuotaCascader(h)
    target = 100_000.0
    quotas = cascader.cascade_quota(
        'Enterprise_AMER', target,
        metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
        gate_fallback='strand_at_root',
        verbose=False,
    )
    # Root is never gated to $0 — it holds the target explicitly...
    assert quotas['Enterprise_AMER'] == target
    # ...and the stranded amount is reported, not silent.
    assert abs(cascader.unallocated - target) < 0.01
    assert cascader.unallocated_nodes == {'Enterprise_AMER': target}
    df = cascader.quotas_to_dataframe(quotas)
    assert bool(df.loc[df.node_id == 'Enterprise_AMER', 'is_unallocated'].iloc[0])
    print(f"  unallocated: ${cascader.unallocated:,.2f} at root; "
          f"is_unallocated flagged in DataFrame")


# ----------------------------------------------------------------------
# 21. Issue #12: gate_fallback='error' raises GateAllocationError
# ----------------------------------------------------------------------
def test_issue12_gate_fallback_error_raises():
    print(f"\n\n{SEPARATOR}")
    print("TEST 21: Issue #12 — gate_fallback='error' raises "
          "GateAllocationError")
    print(SEPARATOR)
    h = _issue12_hierarchy(all_gated=True)
    cascader = QuotaCascader(h)
    try:
        cascader.cascade_quota(
            'Enterprise_AMER', 100_000.0,
            metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
            gate_fallback='error',
            verbose=False,
        )
        raise AssertionError("expected GateAllocationError")
    except GateAllocationError as e:
        print(f"  Raised as expected: {e}")


# ----------------------------------------------------------------------
# 22. Issue #12: base_quotas — hedged = base x hedge, base reconciles
# ----------------------------------------------------------------------
def test_issue12_base_quotas_single_call():
    print(f"\n\n{SEPARATOR}")
    print("TEST 22: Issue #12 — single call yields base + hedged layers; "
          "base reconciles at every depth")
    print(SEPARATOR)
    h = _issue12_hierarchy()
    cascader = QuotaCascader(h)
    target = 1_000_000.0
    hedge = 1.05
    quotas = cascader.cascade_quota(
        'Enterprise_AMER', target,
        hedge_multiplier=hedge,
        metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
        verbose=False,
    )
    base = cascader.base_quotas
    # Base layer reconciles at every depth despite the gate + hedge
    base_sums = _depth_sums(cascader, base)
    for d in range(4):
        print(f"  base depth {d} sum: ${base_sums[d]:,.2f}")
        assert abs(base_sums[d] - target) < 0.01
    # Hedge compounds per level: hedged / base == hedge**depth
    depths = cascader._node_depths()
    for node, q in quotas.items():
        if base[node] > 0:
            assert abs(q / base[node] - hedge ** depths[node]) < 1e-9
    # quotas_to_dataframe(unhedged_quotas='auto') needs no second run
    df = cascader.quotas_to_dataframe(quotas, unhedged_quotas='auto')
    assert {'unhedged_quota', 'hedge_buffer', 'overassignment_pct'} <= set(df.columns)
    print("  hedged = base x hedge^depth verified; 'auto' audit columns OK")


# ----------------------------------------------------------------------
# 23. Issue #12: reconciliation_report strict mode catches shortfalls
# ----------------------------------------------------------------------
def test_issue12_reconciliation_report_strict():
    print(f"\n\n{SEPARATOR}")
    print("TEST 23: Issue #12 — reconciliation_report(strict=True) catches "
          "a stranded cascade")
    print(SEPARATOR)
    h = _issue12_hierarchy(all_gated=True)
    cascader = QuotaCascader(h)
    target = 100_000.0
    quotas = cascader.cascade_quota(
        'Enterprise_AMER', target,
        metrics=_ISSUE12_METRICS, gate_metrics=_ISSUE12_GATES,
        gate_fallback='strand_at_root',
        verbose=False,
    )
    report = cascader.reconciliation_report(quotas, target=target)
    assert bool(report.loc[report.depth == 0, 'reconciles'].iloc[0])
    assert not bool(report.loc[report.depth == 1, 'reconciles'].iloc[0])
    try:
        cascader.reconciliation_report(quotas, target=target, strict=True)
        raise AssertionError("expected AssertionError from strict mode")
    except AssertionError as e:
        if "does not reconcile" not in str(e):
            raise
        print(f"  strict mode raised as expected: {e}")


if __name__ == '__main__':
    test_backward_compat()
    test_single_proportional_metric_matches_legacy()
    test_blend_with_inverse_direction()
    test_brand_new_ic_csv_column()
    test_brand_new_rule_mutex()
    test_brand_new_rule_default_when_no_explicit()
    test_arbitrary_metric_names()
    test_boolean_metric()
    test_suggest_weights_keeps_user_direction()
    test_suggest_weights_requires_direction()
    test_suggest_directions_and_weights_still_available()
    test_pipeline_adjuster_multi_column()
    test_end_to_end_with_synthetic_csv()
    test_gate_metric_basic()
    test_gate_propagates_upward()
    test_gates_compose_with_and()
    test_cro_override_wins_over_gate()
    test_issue12_gated_subtree_reconciles_at_every_depth()
    test_issue12_fully_gated_root_redistributes()
    test_issue12_strand_at_root_reports_unallocated()
    test_issue12_gate_fallback_error_raises()
    test_issue12_base_quotas_single_call()
    test_issue12_reconciliation_report_strict()

    print(f"\n\n{SEPARATOR}")
    print("ALL MULTI-METRIC TESTS PASSED")
    print(SEPARATOR)
