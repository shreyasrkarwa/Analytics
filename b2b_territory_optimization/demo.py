import pandas as pd
from data_generator import B2BDataGenerator
from b2b_territory_optimization.taxonomy import TaxonomySchema
from b2b_territory_optimization.allocator import TerritoryAllocator
from b2b_territory_optimization.assignment import SellerAssignmentMatrix
from b2b_territory_optimization.reassignment import ReassignmentEngine

def run_demo():
    print("="*60)
    print(" B2B TERRITORY OPTIMIZATION & MANAGEMENT - END-TO-END DEMO")
    print("="*60)
    
    # ---------------------------------------------------------
    # Step 1: Generate Synthetic CRM Data
    # ---------------------------------------------------------
    print("\n[1] Generating Synthetic B2B Account Data (1,000 Accounts)...")
    generator = B2BDataGenerator(random_seed=42)
    df = generator.generate_accounts(1000)
    print(f"Total Accounts: {len(df)}")
    print(f"Total Global TAM: ${df['Estimated_TAM'].sum():,.2f}")
    
    # ---------------------------------------------------------
    # Step 2: Define Taxonomy (Strict Constraints)
    # ---------------------------------------------------------
    print("\n[2] Defining Territory Taxonomy Boundaries (Region + Segment)...")
    taxonomy_cols = ['Region', 'Account_Segment']
    schema = TaxonomySchema(df, taxonomy_cols)
    
    print("\nTaxonomy Buckets (Top 5 by Account Count):")
    print(schema.get_bucket_summary().head())
    
    # ---------------------------------------------------------
    # Step 3: Algorithmic Allocation
    # ---------------------------------------------------------
    print("\n[3] Allocating Accounts to Territories via Greedy Optimization...")
    # Let's say we decide on headcount for each bucket:
    k_mapping = {
        ('AMER', 'SMB'): 4,
        ('AMER', 'Mid-Market'): 3,
        ('AMER', 'Enterprise'): 2,
        ('EMEA', 'SMB'): 3,
        ('EMEA', 'Mid-Market'): 2,
        ('EMEA', 'Enterprise'): 1,
        ('APAC', 'SMB'): 2,
        ('APAC', 'Mid-Market'): 1,
        ('APAC', 'Enterprise'): 1,
    }
    
    allocator = TerritoryAllocator(target_metric='Estimated_TAM')
    allocated_df = allocator.allocate_all_taxonomies(schema, k_mapping)
    
    # Show balance for AMER Mid-Market
    amer_mm = allocated_df[allocated_df['Territory_ID'].str.startswith("AMER_Mid-Market")]
    summary = amer_mm.groupby('Territory_ID').agg(
        Account_Count=('Account_ID', 'count'),
        Total_TAM=('Estimated_TAM', 'sum')
    ).reset_index()
    
    mean_tam = summary['Total_TAM'].mean()
    summary['Variance_Pct'] = (summary['Total_TAM'] - mean_tam) / mean_tam * 100
    
    print("\nOptimized Balance for AMER Mid-Market Territories:")
    print(summary.to_string())
    
    # ---------------------------------------------------------
    # Step 4: Seller Role Assignment
    # ---------------------------------------------------------
    print("\n[4] Generating Seller Assignment Matrix...")
    unique_territories = allocated_df['Territory_ID'].unique().tolist()
    matrix = SellerAssignmentMatrix(unique_territories)
    
    matrix.add_role('Account_Executive', 1.0) # 1:1
    matrix.add_role('Sales_Engineer', 3.0)    # 1 SE covers 3 Territories
    matrix.add_role('BDR', 0.5)               # 2 BDRs per Territory
    
    roster = matrix.get_territory_roster()
    print("\nSample Staffing Roster (First 3 Territories):")
    print(roster.head(3).to_string())
    
    # ---------------------------------------------------------
    # Step 5: Manager Overrides & Rebalancing
    # ---------------------------------------------------------
    print("\n[5] Simulating Manager Overrides (Relationship Mapping)...")
    
    reassign_engine = ReassignmentEngine(allocated_df, target_metric='Estimated_TAM')
    
    # Get two AMER Mid-Market territories
    t1 = "AMER_Mid-Market_Territory_1"
    t2 = "AMER_Mid-Market_Territory_2"
    
    # Pick a large account from T1
    t1_accounts = allocated_df[allocated_df['Territory_ID'] == t1].sort_values('Estimated_TAM', ascending=False)
    whale_acc = t1_accounts.iloc[0]
    
    print(f"Manager manually moves '{whale_acc['Account_Name']}' (${whale_acc['Estimated_TAM']:,.0f})")
    print(f"from {t1} to {t2} due to an existing seller relationship.")
    
    reassign_engine.move_account(whale_acc['Account_ID'], t2)
    
    print("\nNew Imbalance Detected:")
    imbalance = reassign_engine.get_imbalance(taxonomy_bucket="AMER_Mid-Market")
    print(imbalance[['Territory_ID', 'Variance_From_Mean', 'Variance_Pct']])
    
    print("\nAuto-Rebalancing Suggestions (Accounts to move back):")
    suggestions = reassign_engine.suggest_rebalance(over_territory=t2, under_territory=t1)
    
    for s in suggestions:
        print(f"- Suggest Move: {s['Account_ID']} ({s['Account_Name']}) | Value: ${s['Value']:,.0f}")
        print(f"  Improves variance delta by: ${s['Delta_Improvement']:,.0f}")
        print("-" * 40)
        
    print("\n=== DEMO COMPLETE ===")

if __name__ == "__main__":
    run_demo()
