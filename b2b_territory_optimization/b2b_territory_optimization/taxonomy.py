import pandas as pd
from typing import List, Dict, Tuple

class TaxonomySchema:
    """
    Defines the strict hierarchical boundaries for territories.
    Accounts are grouped into "Taxonomy Buckets" based on the specified columns.
    Territory optimization happens strictly WITHIN these buckets, ensuring
    hard constraints (e.g. an SMB account cannot end up in an Enterprise territory).
    """
    
    def __init__(self, df: pd.DataFrame, taxonomy_columns: List[str]):
        """
        Args:
            df: The complete accounts DataFrame.
            taxonomy_columns: List of columns that define the strict territory 
                              boundaries (e.g. ['Region', 'Account_Segment']).
        """
        self.taxonomy_columns = taxonomy_columns
        
        # Validate columns exist
        missing = [col for col in taxonomy_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Taxonomy columns not found in DataFrame: {missing}")
            
        self.df = df
        self.buckets = self._create_buckets()
        
    def _create_buckets(self) -> Dict[Tuple, pd.DataFrame]:
        """
        Groups the dataframe into strict buckets based on the taxonomy columns.
        Returns a dictionary mapping the taxonomy tuple to a sub-dataframe.
        """
        buckets = {}
        grouped = self.df.groupby(self.taxonomy_columns)
        
        for name, group in grouped:
            # name is a tuple if multiple columns, or a single value if one column
            if not isinstance(name, tuple):
                name = (name,)
            buckets[name] = group.copy()
            
        return buckets
        
    def get_bucket(self, taxonomy_values: Tuple) -> pd.DataFrame:
        """Returns the accounts in a specific taxonomy bucket."""
        if not isinstance(taxonomy_values, tuple):
            taxonomy_values = (taxonomy_values,)
        return self.buckets.get(taxonomy_values, pd.DataFrame())
    
    def get_all_bucket_keys(self) -> List[Tuple]:
        """Returns a list of all unique taxonomy combinations."""
        return list(self.buckets.keys())

    def get_bucket_summary(self) -> pd.DataFrame:
        """Returns a summary of account counts and total TAM per bucket."""
        summary = []
        for key, df_bucket in self.buckets.items():
            name = "_".join([str(k) for k in key])
            summary.append({
                'Taxonomy_Key': key,
                'Taxonomy_Name': name,
                'Account_Count': len(df_bucket),
                'Total_TAM': df_bucket['Estimated_TAM'].sum() if 'Estimated_TAM' in df_bucket.columns else 0
            })
        return pd.DataFrame(summary).sort_values('Account_Count', ascending=False)

if __name__ == "__main__":
    from data_generator import B2BDataGenerator
    df = B2BDataGenerator().generate_accounts(100)
    
    # Example 1: 1-Level Taxonomy
    print("--- 1-Level Taxonomy (Region) ---")
    schema1 = TaxonomySchema(df, ['Region'])
    print(schema1.get_bucket_summary())
    
    # Example 2: 2-Level Taxonomy
    print("\n--- 2-Level Taxonomy (Region + Segment) ---")
    schema2 = TaxonomySchema(df, ['Region', 'Account_Segment'])
    print(schema2.get_bucket_summary())
