import pandas as pd
from typing import List, Dict, Tuple

class ReassignmentEngine:
    """
    Handles manual manager overrides (moving an account from one territory to another).
    Automatically calculates the resulting imbalance and suggests counter-moves
    to restore equilibrium.
    """
    
    def __init__(self, allocated_df: pd.DataFrame, target_metric: str = 'Estimated_TAM'):
        self.df = allocated_df.copy()
        self.target_metric = target_metric
        self.history = [] # Tracks manual moves
        
    def move_account(self, account_id: str, new_territory_id: str):
        """Manually moves an account to a new territory."""
        mask = self.df['Account_ID'] == account_id
        if not mask.any():
            raise ValueError(f"Account {account_id} not found.")
            
        old_territory_id = self.df.loc[mask, 'Territory_ID'].values[0]
        val = self.df.loc[mask, self.target_metric].values[0]
        
        self.df.loc[mask, 'Territory_ID'] = new_territory_id
        
        self.history.append({
            'Account_ID': account_id,
            'From': old_territory_id,
            'To': new_territory_id,
            'Value': val
        })
        
    def get_imbalance(self, taxonomy_bucket: str = None) -> pd.DataFrame:
        """
        Calculates the current total metric per territory.
        If taxonomy_bucket is provided, limits analysis to territories starting with that string.
        """
        df_view = self.df
        if taxonomy_bucket:
            df_view = self.df[self.df['Territory_ID'].str.startswith(taxonomy_bucket)]
            
        summary = df_view.groupby('Territory_ID').agg(
            Account_Count=('Account_ID', 'count'),
            Total_Metric=(self.target_metric, 'sum')
        ).reset_index()
        
        mean_val = summary['Total_Metric'].mean()
        summary['Variance_From_Mean'] = summary['Total_Metric'] - mean_val
        summary['Variance_Pct'] = (summary['Variance_From_Mean'] / mean_val) * 100
        
        return summary.sort_values('Variance_From_Mean', ascending=False)
        
    def suggest_rebalance(self, over_territory: str, under_territory: str, 
                          excluded_account_ids: List[str] = None, top_n: int = 5) -> List[Dict]:
        """
        Suggests accounts (or combinations of accounts) to move from the over-allocated 
        territory to the under-allocated territory to minimize their difference.
        
        Args:
            over_territory: Territory to take accounts from.
            under_territory: Territory to give accounts to.
            excluded_account_ids: List of account IDs the manager refuses to move.
            top_n: Number of suggestions to return.
        """
        import itertools
        
        summary = self.get_imbalance()
        
        val_over = summary[summary['Territory_ID'] == over_territory]['Total_Metric'].values[0]
        val_under = summary[summary['Territory_ID'] == under_territory]['Total_Metric'].values[0]
        
        difference = val_over - val_under
        if difference <= 0:
            return [] # No need to move from over to under if it's not actually over
            
        target_move_value = difference / 2.0
        
        excluded_account_ids = excluded_account_ids or []
        recently_moved = [m['Account_ID'] for m in self.history]
        all_exclusions = set(excluded_account_ids + recently_moved)
        
        candidates = self.df[
            (self.df['Territory_ID'] == over_territory) & 
            (~self.df['Account_ID'].isin(all_exclusions))
        ].copy()
        
        suggestions = []
        
        # 1. Single Account Swaps
        candidates['Distance_To_Target'] = abs(candidates[self.target_metric] - target_move_value)
        single_best = candidates.sort_values('Distance_To_Target').head(5)
        
        for _, row in single_best.iterrows():
            suggestions.append({
                'Action': 'SUGGESTED_MOVE',
                'Type': 'Single Account',
                'Accounts': [{'Account_ID': row['Account_ID'], 'Account_Name': row.get('Account_Name', '')}],
                'Total_Value': row[self.target_metric],
                'From': over_territory,
                'To': under_territory,
                'Delta_Improvement': abs(difference) - abs(difference - 2 * row[self.target_metric])
            })
            
        # 2. Two-Account Combinations
        # Only evaluate combinations for a subset of the data to avoid combinatorial explosion
        subset = candidates.sort_values(self.target_metric).head(50) # Take 50 smaller accounts
        if len(subset) >= 2:
            records = subset.to_dict('records')
            combos = []
            for a, b in itertools.combinations(records, 2):
                combined_val = a[self.target_metric] + b[self.target_metric]
                dist = abs(combined_val - target_move_value)
                combos.append((dist, combined_val, a, b))
                
            # Sort combos by distance to target and take the best ones
            combos.sort(key=lambda x: x[0])
            for dist, combined_val, a, b in combos[:5]:
                suggestions.append({
                    'Action': 'SUGGESTED_MOVE',
                    'Type': 'Combination (2 Accounts)',
                    'Accounts': [
                        {'Account_ID': a['Account_ID'], 'Account_Name': a.get('Account_Name', '')},
                        {'Account_ID': b['Account_ID'], 'Account_Name': b.get('Account_Name', '')}
                    ],
                    'Total_Value': combined_val,
                    'From': over_territory,
                    'To': under_territory,
                    'Delta_Improvement': abs(difference) - abs(difference - 2 * combined_val)
                })
                
        # Sort all suggestions (single and combinations) by improvement and return top_n
        suggestions.sort(key=lambda x: x['Delta_Improvement'], reverse=True)
        return suggestions[:top_n]

if __name__ == "__main__":
    # Test script
    df = pd.DataFrame({
        'Account_ID': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
        'Territory_ID': ['T1', 'T1', 'T1', 'T2', 'T2', 'T2'],
        'Estimated_TAM': [100, 100, 100, 100, 100, 100]
    })
    
    engine = ReassignmentEngine(df, 'Estimated_TAM')
    
    # Both start at 300 TAM
    print("Initial Imbalance:")
    print(engine.get_imbalance())
    
    # Manager moves A1 from T1 to T2 because of a relationship
    print("\n--- Moving A1 from T1 to T2 ---")
    engine.move_account('A1', 'T2')
    
    print("\nNew Imbalance:")
    print(engine.get_imbalance())
    
    print("\nRebalancing Suggestions (Including Combinations):")
    suggestions = engine.suggest_rebalance(over_territory='T2', under_territory='T1')
    for s in suggestions:
        print(s)
        
    print("\n--- What if manager rejects A4? ---")
    print("Rebalancing Suggestions (Excluding A4):")
    suggestions2 = engine.suggest_rebalance(over_territory='T2', under_territory='T1', excluded_account_ids=['A4'])
    for s in suggestions2:
        print(s)
