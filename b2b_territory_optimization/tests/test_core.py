import pytest
import pandas as pd
from b2b_territory_optimization.allocator import TerritoryAllocator
from b2b_territory_optimization.assignment import SellerAssignmentMatrix
from b2b_territory_optimization.taxonomy import TaxonomySchema

def test_taxonomy_bucketing():
    df = pd.DataFrame({
        'Account_ID': [1, 2, 3, 4],
        'Region': ['AMER', 'AMER', 'EMEA', 'EMEA'],
        'Segment': ['SMB', 'SMB', 'SMB', 'Enterprise']
    })
    
    schema = TaxonomySchema(df, ['Region', 'Segment'])
    buckets = schema.get_all_bucket_keys()
    
    assert len(buckets) == 3
    assert ('AMER', 'SMB') in buckets
    assert ('EMEA', 'Enterprise') in buckets

def test_allocator_balancing():
    df = pd.DataFrame({
        'Account_ID': [1, 2, 3, 4, 5, 6],
        'Estimated_TAM': [100, 100, 100, 100, 100, 100]
    })
    
    allocator = TerritoryAllocator(target_metric='Estimated_TAM')
    allocated = allocator.allocate_bucket(df, num_territories=2, taxonomy_name="Test")
    
    # 6 accounts into 2 territories = 3 per territory
    counts = allocated.groupby('Territory_ID').size()
    assert len(counts) == 2
    assert counts.iloc[0] == 3
    assert counts.iloc[1] == 3
    
def test_export_to_hierarchy():
    territories = ['T1', 'T2']
    matrix = SellerAssignmentMatrix(territories)
    matrix.add_role('Manager', 2.0) # 1 manager for 2 terrs
    matrix.add_role('AE', 1.0) # 1 AE per terr
    
    df_hierarchy = matrix.export_to_hierarchy(['Manager', 'AE'])
    
    assert 'Global' in df_hierarchy.columns
    assert 'Manager' in df_hierarchy.columns
    assert 'AE' in df_hierarchy.columns
    assert 'Territory_ID' in df_hierarchy.columns
    
    # Both territories should have the same manager ID
    mgrs = df_hierarchy['Manager'].unique()
    assert len(mgrs) == 1
