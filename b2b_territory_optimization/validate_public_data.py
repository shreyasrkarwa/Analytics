import os
import pandas as pd
import requests
from b2b_territory_optimization.taxonomy import TaxonomySchema
from b2b_territory_optimization.allocator import TerritoryAllocator
from b2b_territory_optimization.reassignment import ReassignmentEngine
from b2b_territory_optimization.assignment import SellerAssignmentMatrix

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

def test_on_telco():
    print("="*60)
    print(" PUBLIC DATASET VALIDATION (IBM Telco Churn)")
    print("="*60)
    
    # 1. Load Data
    print(f"\n[1] Downloading Public Dataset from IBM GitHub...")
    try:
        df = pd.read_csv(TELCO_URL)
    except Exception as e:
        print(f"Failed to download: {e}")
        return
        
    # Clean up empty strings in TotalCharges and cast to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Rename columns to map to our B2B concepts
    df = df.rename(columns={
        'customerID': 'Account_ID',
        'TotalCharges': 'Estimated_TAM', # Using LTV/TotalCharges as TAM
        'MonthlyCharges': 'Annual_Revenue' 
    })
    
    print(f"Loaded {len(df):,} accounts.")
    print(f"Total Network TAM (TotalCharges): ${df['Estimated_TAM'].sum():,.2f}")
    
    # 2. Define Taxonomy
    # Using Contract (Month-to-month, 1yr, 2yr) and InternetService (Fiber, DSL, No)
    print("\n[2] Defining Taxonomy Schema: [Contract, InternetService]...")
    taxonomy_cols = ['Contract', 'InternetService']
    schema = TaxonomySchema(df, taxonomy_cols)
    
    print("\nTop 3 Taxonomy Buckets:")
    summary = schema.get_bucket_summary()
    print(summary.head(3).to_string())
    
    # 3. Allocating 
    print("\n[3] Allocating the largest bucket (Month-to-month, Fiber optic)...")
    # Let's allocate the biggest bucket into 5 territories
    bucket_key = ('Month-to-month', 'Fiber optic')
    
    # We only allocate this one bucket for demonstration
    k_mapping = {
        bucket_key: 5
    }
    
    allocator = TerritoryAllocator(target_metric='Estimated_TAM')
    allocated_df = allocator.allocate_bucket(
        df_bucket=schema.get_bucket(bucket_key),
        num_territories=5,
        taxonomy_name="MTM_Fiber"
    )
    
    # 4. Results
    results = allocated_df.groupby('Territory_ID').agg(
        Account_Count=('Account_ID', 'count'),
        Total_TAM=('Estimated_TAM', 'sum')
    ).reset_index()
    
    mean_tam = results['Total_TAM'].mean()
    results['Variance_From_Mean'] = results['Total_TAM'] - mean_tam
    results['Variance_Pct'] = (results['Variance_From_Mean'] / mean_tam) * 100
    
    print("\nAllocation Results for MTM_Fiber:")
    print(results.to_string())
    
    max_variance = results['Variance_Pct'].abs().max()
    print(f"\nSUCCESS: Balanced {len(allocated_df)} accounts across 5 territories.")
    print(f"Maximum TAM Variance from mean: {max_variance:.3f}%\n")

    # 4. Test Human Resource Mapping
    print("[4] Testing Human Resource Mapping on Public Data...")
    unique_territories = allocated_df['Territory_ID'].unique().tolist()
    matrix = SellerAssignmentMatrix(unique_territories)
    
    # Let's say for this specific bucket, we want:
    # 1 Account Executive per territory
    # 2 BDRs per territory (ratio 0.5)
    # 1 Manager for all 5 territories (ratio 5.0)
    matrix.add_role('Account_Executive', 1.0)
    matrix.add_role('BDR', 0.5)
    matrix.add_role('Manager', 5.0)
    
    roster = matrix.get_territory_roster()
    print("\nGenerated Staffing Roster for MTM_Fiber Territories:")
    print(roster.to_string())
    print("\n")

    # 5. Test Manager Override
    print("[5] Testing Manager Override on Public Data...")
    reassign_engine = ReassignmentEngine(allocated_df, target_metric='Estimated_TAM')
    
    t1 = "MTM_Fiber_Territory_1"
    t2 = "MTM_Fiber_Territory_2"
    
    # Pick a large account from T1 (highest TotalCharges)
    t1_accounts = allocated_df[allocated_df['Territory_ID'] == t1].sort_values('Estimated_TAM', ascending=False)
    whale_acc = t1_accounts.iloc[0]
    
    print(f"Manager manually moves Account '{whale_acc['Account_ID']}' (TAM: ${whale_acc['Estimated_TAM']:,.2f})")
    print(f"from {t1} to {t2}.")
    
    reassign_engine.move_account(whale_acc['Account_ID'], t2)
    
    print("\nNew Imbalance Detected:")
    imbalance = reassign_engine.get_imbalance()
    print(imbalance[['Territory_ID', 'Total_Metric', 'Variance_From_Mean', 'Variance_Pct']])
    
    print("\nAuto-Rebalancing Suggestions (Accounts to move back to restore balance):")
    suggestions = reassign_engine.suggest_rebalance(over_territory=t2, under_territory=t1)
    
    for s in suggestions:
        acc_names = [f"{a['Account_ID']}" for a in s['Accounts']]
        print(f"- Suggest Move ({s['Type']}): {', '.join(acc_names)} | Total Value: ${s['Total_Value']:,.2f}")
        print(f"  Improves variance delta by: ${s['Delta_Improvement']:,.2f}")
        print("-" * 40)
        
    print("\n--- What if the manager refuses to move the best single account? ---")
    best_single_id = suggestions[0]['Accounts'][0]['Account_ID']
    print(f"Manager locks account: {best_single_id}")
    
    suggestions_excluded = reassign_engine.suggest_rebalance(
        over_territory=t2, 
        under_territory=t1, 
        excluded_account_ids=[best_single_id]
    )
    
    print("\nNew Auto-Rebalancing Suggestions:")
    for s in suggestions_excluded:
        acc_names = [f"{a['Account_ID']}" for a in s['Accounts']]
        print(f"- Suggest Move ({s['Type']}): {', '.join(acc_names)} | Total Value: ${s['Total_Value']:,.2f}")
        print(f"  Improves variance delta by: ${s['Delta_Improvement']:,.2f}")
        print("-" * 40)

if __name__ == "__main__":
    test_on_telco()
