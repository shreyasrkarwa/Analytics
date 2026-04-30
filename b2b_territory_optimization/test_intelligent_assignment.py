from data_generator import B2BDataGenerator
from b2b_territory_optimization.taxonomy import TaxonomySchema
from b2b_territory_optimization.allocator import TerritoryAllocator
from b2b_territory_optimization.intelligent_assignment import IntelligentAssigner
import pandas as pd

def test_intelligent_assignment():
    print("="*60)
    print(" INTELLIGENT SELLER ASSIGNMENT DEMO")
    print("="*60)
    
    # 1. Generate Accounts & Sellers
    print("\n[1] Generating 500 accounts and 5 sellers...")
    gen = B2BDataGenerator(random_seed=42)
    accounts_df = gen.generate_accounts(500)
    sellers_df = gen.generate_synthetic_sellers(5)
    
    print("\nGenerated Sellers:")
    print(sellers_df[['Seller_ID', 'Region', 'Account_Segment', 'Seniority', 'Domain_Expertise']].to_string())
    
    # 2. Carve Territories
    print("\n[2] Carving Territories (Strict bounds: Region + Account_Segment)...")
    tax_cols = ['Region', 'Account_Segment']
    schema = TaxonomySchema(accounts_df, tax_cols)
    allocator = TerritoryAllocator()
    
    # Let's allocate one territory per bucket for simplicity
    k_mapping = {k: 1 for k in schema.get_all_bucket_keys()}
    allocated_df = allocator.allocate_all_taxonomies(schema, k_mapping)
    
    # 3. Intelligent Assignment
    print("\n[3] Running Bipartite Matching (Hungarian Algorithm)...")
    assigner = IntelligentAssigner(sellers_df, allocated_df, taxonomy_columns=tax_cols)
    
    print("\nExtracted Territory Profiles (Top 5):")
    print(assigner.territory_profiles[['Territory_ID', 'Region', 'Account_Segment', 'Dominant_Industry', 'Territory_Tier']].head().to_string())
    
    assignments = assigner.assign_sellers()
    
    # Merge back to show the logic
    results = pd.merge(assignments, sellers_df, on='Seller_ID', how='left')
    results = pd.merge(results, assigner.territory_profiles, on='Territory_ID', how='left')
    
    print("\n[4] Optimization Results:")
    for _, row in results.iterrows():
        print("-" * 50)
        print(f"SELLER: {row['Seller_ID']} (Region: {row['Region_x']}, Segment: {row['Account_Segment_x']}, Domain: {row['Domain_Expertise']}, Seniority: {row['Seniority']})")
        print(f"TERRITORY: {row['Territory_ID']} (Region: {row['Region_y']}, Segment: {row['Account_Segment_y']}, Dominant: {row['Dominant_Industry']}, Tier: {row['Territory_Tier']})")
        print(f"FIT COST: {row['Fit_Cost']} | VALID MATCH: {row['Is_Valid_Match']}")
        if not row['Is_Valid_Match']:
            print("WARNING: Forced match due to lack of supply (Hard constraint violated!)")

if __name__ == "__main__":
    test_intelligent_assignment()
