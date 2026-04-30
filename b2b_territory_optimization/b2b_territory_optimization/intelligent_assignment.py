import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple

class IntelligentAssigner:
    """
    Uses Bipartite Matching (the Hungarian Algorithm) to mathematically 
    assign real human sellers to optimized territories based on fit criteria
    and strict dimension matching.
    """
    
    def __init__(self, sellers_df: pd.DataFrame, allocated_df: pd.DataFrame, 
                 taxonomy_columns: List[str]):
        """
        Args:
            sellers_df: DataFrame of human sellers with profiles.
            allocated_df: Output DataFrame from TerritoryAllocator containing Account assignments.
            taxonomy_columns: List of columns used to create the taxonomy (e.g. ['Region', 'Account_Segment']).
        """
        self.sellers_df = sellers_df.copy()
        self.allocated_df = allocated_df.copy()
        self.taxonomy_columns = taxonomy_columns
        
        # Build Territory Profiles
        self.territory_profiles = self._build_territory_profiles()
        
    def _build_territory_profiles(self) -> pd.DataFrame:
        """
        Extracts the profile of each territory from the allocated accounts.
        """
        # Group by territory to get totals and taxonomy metadata
        profiles = []
        
        grouped = self.allocated_df.groupby('Territory_ID')
        for terr_id, group in grouped:
            # Taxonomy data should be uniform across all accounts in a territory
            tax_data = {col: group[col].iloc[0] for col in self.taxonomy_columns}
            
            # Find dominant industry in the territory
            if 'Industry' in group.columns:
                dominant_industry = group['Industry'].value_counts().index[0]
            else:
                dominant_industry = None
                
            total_tam = group['Estimated_TAM'].sum() if 'Estimated_TAM' in group.columns else 0.0
            
            profile = {
                'Territory_ID': terr_id,
                'Total_TAM': total_tam,
                'Dominant_Industry': dominant_industry,
                'Account_Count': len(group)
            }
            profile.update(tax_data)
            profiles.append(profile)
            
        profiles_df = pd.DataFrame(profiles)
        
        # Rank territories by TAM into tiers (1 = Highest TAM, 3 = Lowest TAM)
        if len(profiles_df) > 0 and 'Total_TAM' in profiles_df.columns:
            # Use qcut if we have enough variance, otherwise fallback
            try:
                profiles_df['Territory_Tier'] = pd.qcut(profiles_df['Total_TAM'], q=3, labels=[3, 2, 1]).astype(int)
            except ValueError:
                # If all TAMs are identical (perfectly balanced), everyone is Tier 2
                profiles_df['Territory_Tier'] = 2
                
        return profiles_df
        
    def build_cost_matrix(self) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Builds the cost matrix where rows are Sellers and columns are Territories.
        Returns:
            Cost Matrix, List of Seller IDs, List of Territory IDs
        """
        sellers = self.sellers_df.to_dict('records')
        territories = self.territory_profiles.to_dict('records')
        
        n_sellers = len(sellers)
        n_territories = len(territories)
        
        # Create a massive cost matrix (initialized to high values)
        # We use Cost instead of Fit because linear_sum_assignment MINIMIZES cost
        cost_matrix = np.full((n_sellers, n_territories), 1000000.0)
        
        seller_ids = []
        territory_ids = [t['Territory_ID'] for t in territories]
        
        for i, seller in enumerate(sellers):
            seller_ids.append(seller['Seller_ID'])
            for j, terr in enumerate(territories):
                
                # 1. HARD CONSTRAINTS (Taxonomy Matching)
                # If a seller has a Region/Segment dimension, it MUST match the territory
                is_valid = True
                for col in self.taxonomy_columns:
                    if col in seller and pd.notna(seller[col]):
                        if seller[col] != terr.get(col):
                            is_valid = False
                            break
                            
                if not is_valid:
                    # Incompatible match - leave cost at 1,000,000 (Infinity)
                    continue
                    
                # 2. SOFT CONSTRAINTS (Calculating Fit)
                # We start with a base cost of 100. A perfect fit will drive the cost down to 0.
                cost = 100.0
                
                # Seniority vs Territory Tier match (both 1 to 3)
                if 'Seniority' in seller and 'Territory_Tier' in terr:
                    tier_diff = abs(seller['Seniority'] - terr['Territory_Tier'])
                    # Penalty for mismatch
                    cost += (tier_diff * 20.0)
                
                # Domain Expertise match
                if 'Domain_Expertise' in seller and 'Dominant_Industry' in terr:
                    if seller['Domain_Expertise'] == terr['Dominant_Industry']:
                        cost -= 30.0 # Reward for matching domain
                        
                cost_matrix[i, j] = cost
                
        return cost_matrix, seller_ids, territory_ids

    def assign_sellers(self) -> pd.DataFrame:
        """
        Runs the Bipartite Matching Algorithm to optimally assign sellers.
        Returns a DataFrame mapping Seller_ID to Territory_ID.
        """
        cost_matrix, seller_ids, territory_ids = self.build_cost_matrix()
        
        # Determine if we have a square, wide, or tall matrix
        # The algorithm will assign min(n_sellers, n_territories) pairings
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            
            # If cost is near infinity, it's an invalid match forced by lack of supply
            is_valid = cost < 500000.0
            
            assignments.append({
                'Seller_ID': seller_ids[r],
                'Territory_ID': territory_ids[c],
                'Fit_Cost': cost if is_valid else np.nan,
                'Is_Valid_Match': is_valid
            })
            
        return pd.DataFrame(assignments)
