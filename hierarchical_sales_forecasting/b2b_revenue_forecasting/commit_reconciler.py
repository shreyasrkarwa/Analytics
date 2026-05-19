import pandas as pd
from typing import Dict, Any

class CommitReconciler:
    """
    A mathematical engine to reconcile bottom-up CRM pipeline coverage 
    with human-in-the-loop managerial commits by quantifying historical bias.
    """
    def __init__(self, historical_data: pd.DataFrame):
        """
        historical_data requires columns:
        ['Manager_ID', 'Historical_Commit', 'Historical_Actual_Closed']
        """
        self.bias_weights = self._calculate_managerial_bias(historical_data)
        
    def _calculate_managerial_bias(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates a 'sandbagging' or 'happy ears' coefficient for each manager.
        Bias > 1.0 means the manager is a sandbagger (they close MORE than they commit).
        Bias < 1.0 means the manager is overly optimistic (they close LESS than they commit).
        """
        bias_dict = {}
        # Group by manager to get their historical aggregate bias
        if not df.empty and 'Manager_ID' in df.columns:
            grouped = df.groupby('Manager_ID')[['Historical_Commit', 'Historical_Actual_Closed']].sum()
            
            for manager, row in grouped.iterrows():
                if row['Historical_Commit'] > 0:
                    bias_dict[manager] = row['Historical_Actual_Closed'] / row['Historical_Commit']
                else:
                    bias_dict[manager] = 1.0
                    
        return bias_dict
        
    def reconcile_forecast(self, manager_id: str, current_commit: float, machine_forecast: float = None) -> float:
        """
        Reconciles the current quarter's commit by adjusting for the manager's historical bias.
        Optionally blends the result with a pure machine-learning baseline.
        """
        # 1. Get their historical truth quotient (default to 1.0 if new manager with no history)
        manager_bias = self.bias_weights.get(manager_id, 1.0)

        # 2. Adjust their human commit based on their historical behavior
        # (Sandbaggers get their number inflated mathematically; overly optimistic managers get deflated)
        adjusted_human_forecast = current_commit * manager_bias

        # 3. If a machine forecast is provided, blend them
        if machine_forecast is not None:
            # We can expand this in future versions to penalize wild machine variance,
            # but for V1, we take a straight average of Adjusted Human vs Machine Baseline
            final_reconciled_forecast = (adjusted_human_forecast + machine_forecast) / 2.0
            return final_reconciled_forecast

        return adjusted_human_forecast

    def reconcile_all(
        self,
        commits_df: pd.DataFrame,
        manager_col: str = "Manager_ID",
        commit_col: str = "Current_Commit",
        machine_col: str = None,
    ) -> pd.DataFrame:
        """
        Batch-reconcile a DataFrame of current-quarter commits and return
        a tidy DataFrame ready for CSV export
        (`.to_csv('commit_reconciliation.csv', index=False)`).

        Columns in the returned DataFrame:
          manager_id, current_commit, bias_quotient, bias_label,
          bias_adjusted_commit, machine_forecast (if machine_col provided),
          blended_forecast (if machine_col provided)

        Parameters
        ----------
        commits_df : pd.DataFrame
            One row per manager for the current quarter. Must contain
            manager_col and commit_col; optionally machine_col.
        manager_col, commit_col, machine_col : str
            Column names in commits_df. Defaults match the documented
            convention used elsewhere in the package.
        """
        rows = []
        for _, row in commits_df.iterrows():
            mgr = row[manager_col]
            commit = float(row[commit_col])
            bias = self.bias_weights.get(mgr, 1.0)
            adjusted = commit * bias
            label = ("sandbagger" if bias > 1.1
                     else "happy_ears" if bias < 0.9
                     else "truth_teller")

            out_row = {
                "manager_id": mgr,
                "current_commit": round(commit, 2),
                "bias_quotient": round(bias, 4),
                "bias_label": label,
                "bias_adjusted_commit": round(adjusted, 2),
            }
            if machine_col is not None and machine_col in commits_df.columns:
                ml = float(row[machine_col])
                blended = (adjusted + ml) / 2.0
                out_row["machine_forecast"] = round(ml, 2)
                out_row["blended_forecast"] = round(blended, 2)
            rows.append(out_row)

        return pd.DataFrame(rows)
