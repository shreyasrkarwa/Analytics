"""
End-to-end test of the PipelineAdjuster using synthetic_hierarchy.csv.
Also validates flexible quarter support and new IC handling in QuotaCascader.
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting.hierarchy import SalesHierarchy
from b2b_revenue_forecasting.quota_cascader import QuotaCascader
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster

SEPARATOR = "=" * 90

def test_flexible_quarters():
    """Test that _calculate_node_historical_capacity works with any number of quarters."""
    print(SEPARATOR)
    print("TEST 1: Flexible Quarter Support")
    print(SEPARATOR)
    
    # Build a tiny hierarchy with 8 quarters of data
    df = pd.DataFrame({
        'Global': ['Corp', 'Corp'],
        'IC': ['IC_A', 'IC_B'],
        'Q1_Attainment': [100.0, 200.0],
        'Q2_Attainment': [110.0, 210.0],
        'Q3_Attainment': [120.0, 220.0],
        'Q4_Attainment': [130.0, 230.0],
        'Q5_Attainment': [140.0, 240.0],
        'Q6_Attainment': [150.0, 250.0],
        'Q7_Attainment': [160.0, 260.0],
        'Q8_Attainment': [170.0, 270.0],
    })
    
    metrics = [f'Q{i}_Attainment' for i in range(1, 9)]
    
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'IC'], metrics_cols=metrics)
    
    cascader = QuotaCascader(h)
    
    cap_a = cascader._calculate_node_historical_capacity('IC_A')
    cap_b = cascader._calculate_node_historical_capacity('IC_B')
    
    expected_a = sum(range(100, 180, 10))  # 100+110+...+170 = 1080
    expected_b = sum(range(200, 280, 10))  # 200+210+...+270 = 1880
    
    print(f"  IC_A capacity (8 quarters): {cap_a:.2f} (expected: {expected_a:.2f}) {'✅' if abs(cap_a - expected_a) < 0.01 else '❌'}")
    print(f"  IC_B capacity (8 quarters): {cap_b:.2f} (expected: {expected_b:.2f}) {'✅' if abs(cap_b - expected_b) < 0.01 else '❌'}")
    
    # Cascade and verify proportional distribution
    quotas = cascader.cascade_quota('Corp', 1000.0)
    print(f"  IC_A quota: ${quotas['IC_A']:.2f} (weight: {cap_a/(cap_a+cap_b):.2%})")
    print(f"  IC_B quota: ${quotas['IC_B']:.2f} (weight: {cap_b/(cap_a+cap_b):.2%})")
    total = quotas['IC_A'] + quotas['IC_B']
    print(f"  Total: ${total:.2f} {'✅' if abs(total - 1000.0) < 0.01 else '❌'}")


def test_new_ic_handling():
    """Test partial history imputation and brand-new IC equal-share."""
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: New IC Handling (Partial History + Brand New)")
    print(SEPARATOR)
    
    # Manager with 5 ICs: 3 tenured, 1 partial history, 1 brand new
    df = pd.DataFrame({
        'Global': ['Corp'] * 5,
        'Manager': ['Mgr_A'] * 5,
        'IC': ['IC_Tenured_1', 'IC_Tenured_2', 'IC_Tenured_3', 'IC_Partial', 'IC_New'],
        'Q1_Attainment': [100000, 200000, 150000, 0, 0],
        'Q2_Attainment': [120000, 180000, 160000, 0, 0],
        'Q3_Attainment': [110000, 190000, 170000, 0, 0],
        'Q4_Attainment': [130000, 210000, 140000, 150000, 0],  # Partial hire has Q4 only
    })
    
    metrics = ['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment']
    
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'Manager', 'IC'], metrics_cols=metrics)
    
    cascader = QuotaCascader(h)
    
    # Check partial-history imputation
    cap_partial = cascader._calculate_node_historical_capacity('IC_Partial')
    print(f"\n  IC_Partial (Q1=0, Q2=0, Q3=0, Q4=$150K):")
    print(f"    → Zeros imputed with avg of non-zero = $150K")
    print(f"    → Imputed capacity: ${cap_partial:,.2f} (expected: $600,000) {'✅' if abs(cap_partial - 600000) < 0.01 else '❌'}")
    
    # Check brand-new IC
    cap_new = cascader._calculate_node_historical_capacity('IC_New')
    print(f"\n  IC_New (all zeros):")
    print(f"    → Capacity: ${cap_new:,.2f} (expected: $0) {'✅' if cap_new == 0.0 else '❌'}")
    
    # Cascade $100K through the manager
    quotas = cascader.cascade_quota('Corp', 100000.0)
    
    print(f"\n  Cascading $100,000 to Mgr_A's 5 ICs:")
    print(f"    IC_New (brand new) gets equal share: ${quotas['IC_New']:,.2f} (expected: $20,000) {'✅' if abs(quotas['IC_New'] - 20000) < 1.0 else '❌'}")
    
    print(f"\n  Remaining $80,000 distributed proportionally by capacity:")
    for ic in ['IC_Tenured_1', 'IC_Tenured_2', 'IC_Tenured_3', 'IC_Partial']:
        cap = cascader._calculate_node_historical_capacity(ic)
        print(f"    {ic}: ${quotas[ic]:,.2f} (capacity: ${cap:,.2f})")
    
    total_ic = sum(quotas[ic] for ic in ['IC_Tenured_1', 'IC_Tenured_2', 'IC_Tenured_3', 'IC_Partial', 'IC_New'])
    mgr_quota = quotas['Mgr_A']
    print(f"\n  Sum of IC quotas: ${total_ic:,.2f}")
    print(f"  Manager quota: ${mgr_quota:,.2f}")
    print(f"  ICs sum matches manager? {'✅' if abs(total_ic - mgr_quota) < 1.0 else '❌'}")


def test_pipeline_adjuster():
    """End-to-end test with synthetic_hierarchy.csv."""
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: Pipeline Adjuster (using synthetic_hierarchy.csv)")
    print(SEPARATOR)
    
    # Load the dataset
    df = pd.read_csv('tests/data/synthetic_hierarchy.csv', keep_default_na=False)
    
    taxonomy = ['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC']
    metrics = ['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment', 'Current_Pipeline']
    
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=taxonomy, metrics_cols=metrics)
    
    # Cascade $100M
    cascader = QuotaCascader(h)
    quotas = cascader.cascade_quota('Global_Corp', 100_000_000.0, hedge_multiplier=1.05)
    
    leaves = [n for n in h.graph.nodes if h.graph.out_degree(n) == 0]
    print(f"\n  Cascaded $100M with 5% hedge → {len(leaves)} ICs")
    
    # --- Test diagnose() ---
    print(f"\n  --- diagnose() with per-region thresholds ---")
    
    coverage_thresholds = {
        'NA':       {'healthy': 1.5, 'at_risk': 0.8},
        'EMEA':     {'healthy': 2.5, 'at_risk': 1.2},
        'APAC':     {'healthy': 3.0, 'at_risk': 1.5},
        '_default': {'healthy': 2.0, 'at_risk': 1.0}
    }
    
    adjuster = PipelineAdjuster(h, quotas)
    diagnosis = adjuster.diagnose(coverage_thresholds)
    
    # Show summary by risk status
    risk_summary = diagnosis.groupby('Risk_Status').agg(
        Count=('Node', 'count'),
        Total_Quota=('Cascaded_Quota', 'sum'),
        Total_Pipeline=('Pipeline', 'sum')
    ).reset_index()
    
    print(f"\n  Risk status summary:")
    for _, row in risk_summary.iterrows():
        print(f"    {row['Risk_Status']:>10s}: {row['Count']:>3d} nodes | Quota: ${row['Total_Quota']:>14,.2f} | Pipeline: ${row['Total_Pipeline']:>14,.2f}")
    
    # Show a few IC-level rows
    ic_rows = diagnosis[diagnosis['Level'] == diagnosis['Level'].max()].head(5)
    print(f"\n  Sample IC diagnoses:")
    for _, row in ic_rows.iterrows():
        print(f"    {row['Node']:>40s} | Coverage: {row['Coverage_Ratio']:.2f}x | Status: {row['Risk_Status']}")
    
    # --- Test flag_only mode ---
    print(f"\n  --- adjust(mode='flag_only') ---")
    flagged = adjuster.adjust(mode='flag_only', coverage_thresholds=coverage_thresholds)
    
    # Verify quotas are unchanged
    unchanged = all(abs(flagged[n] - quotas[n]) < 0.01 for n in quotas)
    print(f"  All quotas unchanged? {'✅' if unchanged else '❌'}")
    
    # --- Test redistribute mode ---
    print(f"\n  --- adjust(mode='redistribute', max_adjustment_pct=0.20) ---")
    redistributed = adjuster.adjust(
        mode='redistribute', 
        coverage_thresholds=coverage_thresholds,
        max_adjustment_pct=0.20
    )
    
    # Verify manager totals are preserved (zero-sum within each manager)
    manager_nodes = [n for n in h.graph.nodes 
                     if list(h.graph.successors(n)) and 
                     all(h.graph.out_degree(c) == 0 for c in h.graph.successors(n))]
    
    all_zero_sum = True
    adjustments_made = 0
    for mgr in manager_nodes:
        ics = list(h.graph.successors(mgr))
        original_total = sum(quotas.get(ic, 0) for ic in ics)
        adjusted_total = sum(redistributed.get(ic, 0) for ic in ics)
        
        if abs(original_total - adjusted_total) > 1.0:
            all_zero_sum = False
            print(f"    ❌ {mgr}: original={original_total:.2f}, adjusted={adjusted_total:.2f}")
        
        for ic in ics:
            if abs(redistributed.get(ic, 0) - quotas.get(ic, 0)) > 0.01:
                adjustments_made += 1
    
    print(f"  Manager totals preserved (zero-sum)? {'✅' if all_zero_sum else '❌'}")
    print(f"  Number of IC quotas adjusted: {adjustments_made}")
    
    # Verify max_adjustment_pct cap
    cap_respected = True
    for ic in leaves:
        orig = quotas.get(ic, 0)
        adj = redistributed.get(ic, 0)
        if orig > 0:
            pct_change = abs(adj - orig) / orig
            if pct_change > 0.201:  # tiny tolerance for floating point
                cap_respected = False
                print(f"    ❌ {ic}: {pct_change:.2%} exceeds 20% cap")
    
    print(f"  Max adjustment cap (20%) respected? {'✅' if cap_respected else '❌'}")
    
    # Show a few ICs that were adjusted
    print(f"\n  Sample adjusted IC quotas:")
    print(f"  {'IC':<42s} {'Original':>14s} {'Adjusted':>14s} {'Change':>10s}")
    print(f"  {'-'*84}")
    shown = 0
    for ic in leaves:
        orig = quotas.get(ic, 0)
        adj = redistributed.get(ic, 0)
        if abs(adj - orig) > 1.0 and shown < 6:
            change_pct = ((adj - orig) / orig) * 100
            print(f"  {ic:<42s} ${orig:>12,.2f} ${adj:>12,.2f} {change_pct:>+8.1f}%")
            shown += 1
    
    # --- Test locked_nodes ---
    print(f"\n  --- adjust() with locked_nodes ---")
    sample_ic = leaves[0]
    locked = {sample_ic: 999999.99}
    
    adj_locked = adjuster.adjust(
        mode='redistribute',
        coverage_thresholds=coverage_thresholds,
        locked_nodes=locked
    )
    
    locked_ok = abs(adj_locked[sample_ic] - 999999.99) < 0.01
    print(f"  Locked node {sample_ic[:30]}... = ${adj_locked[sample_ic]:,.2f} {'✅' if locked_ok else '❌'}")


def test_cro_override():
    """Test new_ic_overrides for CRO-mandated quotas."""
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: CRO Override (new_ic_overrides)")
    print(SEPARATOR)
    
    df = pd.DataFrame({
        'Global': ['Corp'] * 3,
        'Manager': ['Mgr_A'] * 3,
        'IC': ['IC_1', 'IC_2', 'IC_Strategic'],
        'Q1_Attainment': [100000, 200000, 50000],
        'Q2_Attainment': [100000, 200000, 50000],
        'Q3_Attainment': [100000, 200000, 50000],
        'Q4_Attainment': [100000, 200000, 50000],
    })
    
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['Global', 'Manager', 'IC'],
                     metrics_cols=['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment'])
    
    cascader = QuotaCascader(h)
    
    # CRO says IC_Strategic must carry exactly $500K regardless of history
    quotas = cascader.cascade_quota('Corp', 1000000.0, new_ic_overrides={'IC_Strategic': 500000.0})
    
    print(f"\n  Cascading $1M with CRO override: IC_Strategic = $500K")
    print(f"  IC_Strategic: ${quotas['IC_Strategic']:,.2f} (expected: $500,000) {'✅' if abs(quotas['IC_Strategic'] - 500000) < 1 else '❌'}")
    
    remaining = 1000000.0 - 500000.0
    print(f"  Remaining $500K split between IC_1 and IC_2:")
    print(f"    IC_1: ${quotas['IC_1']:,.2f}")
    print(f"    IC_2: ${quotas['IC_2']:,.2f}")
    total = quotas['IC_1'] + quotas['IC_2'] + quotas['IC_Strategic']
    print(f"  Total: ${total:,.2f} {'✅' if abs(total - 1000000) < 1 else '❌'}")


if __name__ == '__main__':
    test_flexible_quarters()
    test_new_ic_handling()
    test_pipeline_adjuster()
    test_cro_override()
    
    print(f"\n\n{SEPARATOR}")
    print("ALL TESTS COMPLETED")
    print(SEPARATOR)
