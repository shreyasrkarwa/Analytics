import pandas as pd
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler

def main():
    # Synthetic Managerial History (Past 4 quarters)
    # Manager A: Sandbagger (Commits 100k but repeatedly Closes 150k) -> Bias Ratio = 1.5
    # Manager B: Happy Ears (Commits 200k but only Closes 100k) -> Bias Ratio = 0.5
    # Manager C: Truth Teller (Commits 100k and precisely Closes 100k) -> Bias Ratio = 1.0
    
    data = {
        'Manager_ID': ['Manager_A_(Sandbagger)', 'Manager_B_(Optimist)', 'Manager_C_(Truth_Teller)'],
        'Historical_Commit': [400000, 800000, 400000],          # Past 4 quarters sum
        'Historical_Actual_Closed': [600000, 400000, 400000]    # Past 4 quarters sum
    }
    df_history = pd.DataFrame(data)
    
    reconciler = CommitReconciler(df_history)
    
    print("--- Calculating Managerial Bias Coefficients ---")
    for mgr, bias in reconciler.bias_weights.items():
        print(f"{mgr[:12]} | Historical Bias Quotient: {bias:.2f}")
        
    print("\n--- Current Quarter (Q1) Reconciliation ---")
    print("Scenario: All three managers submit a manual pipeline commit of $100k.")
    print("The statistical Machine Learning Baseline independently predicts $120k pipeline closure.\n")
    
    current_commit = 100000.0
    machine_forecast = 120000.0
    
    for mgr in df_history['Manager_ID']:
        reconciled = reconciler.reconcile_forecast(mgr, current_commit, machine_forecast)
        
        # We also calculate the raw "adjusted human" side to see what the math did to their human commit 
        human_adjusted = current_commit * reconciler.bias_weights.get(mgr, 1.0)
        
        print(f"[{mgr}]")
        print(f"   -> Raw Human Commit:         ${current_commit:,.2f}")
        print(f"   -> Bias-Adjusted Human:      ${human_adjusted:,.2f}")
        print(f"   -> Final BLENDED Forecast:   ${reconciled:,.2f}\n")

if __name__ == "__main__":
    main()
