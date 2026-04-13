import uuid
import numpy as np
import pandas as pd
from typing import Tuple

def generate_synthetic_b2b_data(n_accounts: int = 1000, max_months: int = 36, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic B2B SaaS account data incorporating static firmographics
    and time-varying continuous telemetry metrics for survival analysis.
    
    Returns:
        static_df: DataFrame containing static firmographic details.
        telemetry_df: DataFrame containing time-varying attributes per month per account.
    """
    np.random.seed(random_seed)
    
    static_records = []
    telemetry_records = []
    
    segments = ['SMB', 'Mid-Market', 'Enterprise']
    regions = ['AMER', 'APAC', 'EMEA']
    industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']
    
    for _ in range(n_accounts):
        account_id = str(uuid.uuid4())
        
        # 1. Firmographic Static Generation
        segment = np.random.choice(segments, p=[0.5, 0.3, 0.2])
        region = np.random.choice(regions, p=[0.6, 0.15, 0.25])
        industry = np.random.choice(industries)
        has_channel_partner = np.random.binomial(1, 0.25)
        
        if segment == 'SMB':
            arr = np.random.normal(15000, 3000)
            contract_length = np.random.choice([12], p=[1.0])
            onboarding_base = np.random.normal(15, 5)
            # Higher base churn risk for SMB
            base_churn_prob = 0.04  
        elif segment == 'Mid-Market':
            arr = np.random.normal(75000, 15000)
            contract_length = np.random.choice([12, 24], p=[0.7, 0.3])
            onboarding_base = np.random.normal(40, 10)
            base_churn_prob = 0.02
        else: # Enterprise
            arr = np.random.normal(250000, 50000)
            contract_length = np.random.choice([12, 24, 36], p=[0.2, 0.5, 0.3])
            onboarding_base = np.random.normal(90, 20)
            base_churn_prob = 0.01
            
        initial_arr = max(5000, arr)
        onboarding_days = max(1, onboarding_base)
        
        # Partner attachment lowers risk
        if has_channel_partner:
            base_churn_prob *= 0.8
            
        static_record = {
            'account_id': account_id,
            'industry': industry,
            'account_segment': segment,
            'account_region': region,
            'has_channel_partner': has_channel_partner,
            'initial_arr': round(initial_arr, 2),
            'contract_length_months': contract_length,
            'onboarding_duration_days': round(onboarding_days)
        }
        
        # 2. Telemetry Time-Varying Simulation
        current_mau = np.random.normal(0.85, 0.1)  # starting healthy
        has_churned = False
        months_survived = 0
        
        for month in range(1, max_months + 1):
            if has_churned:
                break
                
            months_survived = month
            
            # Simulate shifting metrics
            # MAU drifts over time
            current_mau = max(0.1, min(1.0, current_mau + np.random.normal(0, 0.05)))
            feature_adoption = max(0, min(100, current_mau * 100 + np.random.normal(0, 10)))
            support_tickets = max(0, int(np.random.normal((1.0 - current_mau) * 5, 2)))
            csat = min(5, max(1, np.random.normal(current_mau * 5, 0.5)))
            exec_turnover = np.random.binomial(1, 0.02) # 2% chance per month
            overdue_invoices = np.random.binomial(1, 0.01 + (1 - current_mau) * 0.05)
            
            telemetry_records.append({
                'account_id': account_id,
                'time_period': month,
                'monthly_active_users_pct': round(current_mau, 3),
                'feature_adoption_score': round(feature_adoption, 1),
                'support_tickets_created': support_tickets,
                'csat_score': round(csat, 1),
                'executive_sponsor_turnover': exec_turnover,
                'overdue_invoices': overdue_invoices
            })
            
            # Predict dynamic churn threshold based on telemetry
            # If metrics drop, churn probability spikes, especially at renewal boundaries
            monthly_risk = base_churn_prob 
            if exec_turnover:
                monthly_risk += 0.05
            if overdue_invoices:
                monthly_risk += 0.03
            if current_mau < 0.4:
                monthly_risk += 0.04
                
            # Renewal cliff effect: Huge spike in churn risk exactly on the renewal month
            if month % contract_length == 0:
                monthly_risk *= 3.0
                
            if np.random.random() < monthly_risk:
                has_churned = True
                
        # Update static record with survival status (right censoring check)
        static_record['time_to_event'] = months_survived
        static_record['event_observed'] = 1 if has_churned else 0
        
        static_records.append(static_record)
        
    static_df = pd.DataFrame(static_records)
    telemetry_df = pd.DataFrame(telemetry_records)
    
    return static_df, telemetry_df

if __name__ == "__main__":
    print("Generating synthetic B2B dataset...")
    static, telemetry = generate_synthetic_b2b_data(n_accounts=2000)
    print(f"Generated {len(static)} accounts with {len(telemetry)} telemetry records.")
    
    churn_rate = static['event_observed'].mean() * 100
    print(f"Dataset Churn Rate (Uncensored): {churn_rate:.1f}%")
    
    static.to_csv("static_data.csv", index=False)
    telemetry.to_csv("telemetry_data.csv", index=False)
    print("Saved outputs to static_data.csv and telemetry_data.csv")
