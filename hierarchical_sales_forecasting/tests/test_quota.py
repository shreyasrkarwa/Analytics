import pandas as pd
from core.hierarchy import SalesHierarchy
from core.quota_cascader import QuotaCascader

def main():
    # 1. Load the synthetic B2B dataset
    df = pd.read_csv('tests/data/synthetic_hierarchy.csv')

    # 2. Build the flexible org chart
    taxonomy = ['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC']
    metrics = ['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment', 'Current_Pipeline']

    hierarchy = SalesHierarchy()
    hierarchy.from_dataframe(df, path_cols=taxonomy, metrics_cols=metrics)

    print(f"Total Nodes in DAG Hierarchy: {len(hierarchy.graph.nodes)}")
    print(f"Total Individual Contributors (Leaves): {len(hierarchy.get_leaves('Global_Corp'))}")

    # 3. Cascade a $100M Global Quota with a 5% safety hedge at every manager level
    cascader = QuotaCascader(hierarchy)
    GLOBAL_MACRO_TARGET = 100_000_000.0

    print(f"\n[Algorithm Event] Distributing ${GLOBAL_MACRO_TARGET:,.2f} Global Target...")
    print(f"[Algorithm Event] Managerial Hedge Applied: 1.05 (5% Overassignment per branch)")
    assigned_quotas = cascader.cascade_quota('Global_Corp', GLOBAL_MACRO_TARGET, hedge_multiplier=1.05)

    # 4. Verification and Mathematics Check
    total_ic_quota = sum([assigned_quotas[node] for node in hierarchy.get_leaves('Global_Corp')])
    
    print(f"\n--- Output Analysis ---")
    print(f"Sum of all assigned IC Quotas (Bottom-up limit): ${total_ic_quota:,.2f}")
    
    # Mathematical compounding check (5 levels of hedging: Region -> RVP -> Director -> Manager -> IC)
    # The actual depth of hedging in the taxonomy is 5 edges: 1.05^5 = 1.27628 (approx 27.6% over-assignment overall)
    actual_overassignment = ((total_ic_quota / GLOBAL_MACRO_TARGET) - 1) * 100
    print(f"Total Compounded Overassignment (Hedge Buffer): {actual_overassignment:.2f}%\n")
    
    # Let's peek at a random sample of 3 IC quotas based on their 4-quarter history
    print("Sample IC Quota Distributions (Weighted by 4-Quarter specific capacity):")
    sample_ics = hierarchy.get_leaves('Global_Corp')[:3]
    for ic in sample_ics:
        historical_4q_sum = cascader._calculate_node_historical_capacity(ic)
        assigned = assigned_quotas[ic]
        print(f" - {ic} | Past 4 Quarters: ${historical_4q_sum:,.2f} | Newly Assigned Quota: ${assigned:,.2f}")

if __name__ == "__main__":
    main()
