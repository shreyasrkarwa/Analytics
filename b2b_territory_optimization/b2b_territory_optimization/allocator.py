import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class TerritoryAllocator:
    """
    Allocates accounts into k balanced territories within a specific Taxonomy Bucket.
    It uses a Greedy heuristic (LPT-inspired) to balance continuous metrics (like TAM)
    while respecting hard locks.
    """
    
    def __init__(self, 
                 target_metric: str = 'Estimated_TAM',
                 territory_prefix: str = 'Territory'):
        """
        Args:
            target_metric: The continuous column to balance (e.g. 'Estimated_TAM').
            territory_prefix: Prefix for naming the output territories.
        """
        self.target_metric = target_metric
        self.territory_prefix = territory_prefix
        
    def allocate_bucket(self, df_bucket: pd.DataFrame, num_territories: int, 
                        taxonomy_name: str, locked_assignments: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Allocates a single bucket into `num_territories`.
        
        Args:
            df_bucket: The accounts in this taxonomy bucket.
            num_territories: How many territories to carve this bucket into.
            taxonomy_name: String representation of the bucket (e.g., "AMER_Enterprise").
            locked_assignments: Dict mapping Account_ID -> target Territory_ID.
                                These accounts will NOT be moved.
        Returns:
            DataFrame with a new column 'Territory_ID'.
        """
        if len(df_bucket) == 0:
            return df_bucket
            
        if num_territories <= 1:
            df_out = df_bucket.copy()
            df_out['Territory_ID'] = f"{taxonomy_name}_{self.territory_prefix}_1"
            return df_out
            
        df = df_bucket.copy()
        locked_assignments = locked_assignments or {}
        
        # Initialize territory totals
        territory_names = [f"{taxonomy_name}_{self.territory_prefix}_{i+1}" for i in range(num_territories)]
        territory_totals = {t: 0.0 for t in territory_names}
        
        df['Territory_ID'] = None
        
        # 1. Process locked assignments first
        locked_mask = df['Account_ID'].isin(locked_assignments.keys())
        if locked_mask.any():
            for idx, row in df[locked_mask].iterrows():
                acc_id = row['Account_ID']
                assigned_t = locked_assignments[acc_id]
                # Ensure the locked territory is within our generated names (or add it if custom)
                if assigned_t not in territory_totals:
                    territory_totals[assigned_t] = 0.0
                
                df.at[idx, 'Territory_ID'] = assigned_t
                val = float(row[self.target_metric]) if pd.notna(row[self.target_metric]) else 0.0
                territory_totals[assigned_t] += val
                
        # 2. Sort remaining accounts descending by the target metric (Greedy/LPT approach)
        unlocked_df = df[~locked_mask].sort_values(by=self.target_metric, ascending=False)
        
        # 3. Allocate greedily to the currently "poorest" territory
        for idx, row in unlocked_df.iterrows():
            val = float(row[self.target_metric]) if pd.notna(row[self.target_metric]) else 0.0
            
            # Find territory with minimum current total
            poorest_territory = min(territory_totals.items(), key=lambda x: x[1])[0]
            
            df.at[idx, 'Territory_ID'] = poorest_territory
            territory_totals[poorest_territory] += val
            
        return df

    def allocate_all_taxonomies(self, schema, 
                                num_territories_per_bucket: Dict[Tuple, int],
                                locked_assignments: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Applies allocation across all buckets in a TaxonomySchema.
        
        Args:
            schema: An initialized TaxonomySchema object.
            num_territories_per_bucket: Dict mapping Taxonomy Key -> number of territories.
        """
        allocated_chunks = []
        
        for key in schema.get_all_bucket_keys():
            df_bucket = schema.get_bucket(key)
            k = num_territories_per_bucket.get(key, 1) # Default to 1 if not specified
            tax_name = "_".join([str(x) for x in key])
            
            df_allocated = self.allocate_bucket(df_bucket, k, tax_name, locked_assignments)
            allocated_chunks.append(df_allocated)
            
        return pd.concat(allocated_chunks, ignore_index=True)


if __name__ == "__main__":
    from data_generator import B2BDataGenerator
    from taxonomy import TaxonomySchema
    
    # 1. Generate Data
    df = B2BDataGenerator(random_seed=42).generate_accounts(500)
    
    # 2. Define strict taxonomy boundaries (Region + Segment)
    schema = TaxonomySchema(df, ['Region', 'Account_Segment'])
    
    # 3. Decide how many territories per bucket (e.g. based on headcount)
    # Let's say we have 3 AMER Enterprise reps, 2 AMER Mid-Market reps, 1 EMEA Enterprise rep, etc.
    k_mapping = {
        ('AMER', 'Enterprise'): 3,
        ('AMER', 'Mid-Market'): 2,
        ('EMEA', 'Enterprise'): 2,
    }
    # Any key not in mapping gets 1 territory by default.
    
    # 4. Allocate
    allocator = TerritoryAllocator(target_metric='Estimated_TAM')
    final_df = allocator.allocate_all_taxonomies(schema, k_mapping)
    
    # 5. Review results for a specific bucket to see balancing
    print("--- AMER Enterprise Territories ---")
    amer_ent = final_df[final_df['Territory_ID'].str.startswith("AMER_Enterprise")]
    summary = amer_ent.groupby('Territory_ID').agg(
        Account_Count=('Account_ID', 'count'),
        Total_TAM=('Estimated_TAM', 'sum')
    ).reset_index()
    
    print(summary)
    
    # Check variance
    mean_tam = summary['Total_TAM'].mean()
    summary['Variance_Pct'] = (summary['Total_TAM'] - mean_tam) / mean_tam * 100
    print("\nVariance from Mean TAM:")
    print(summary[['Territory_ID', 'Variance_Pct']])
