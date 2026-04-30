import pandas as pd
import numpy as np
import random
import uuid

class B2BDataGenerator:
    """
    Generates synthetic B2B account data with rich firmographic, technographic, 
    financial, and intent dimensions for territory optimization testing.
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.segments = ['Enterprise', 'Mid-Market', 'SMB']
        self.regions = ['AMER', 'EMEA', 'APAC']
        self.sub_regions = {
            'AMER': ['North', 'South', 'East', 'West'],
            'EMEA': ['UK', 'DACH', 'Nordics', 'South'],
            'APAC': ['ANZ', 'ASEAN', 'Japan', 'India']
        }
        self.industries = ['Technology', 'Financial Services', 'Healthcare', 'Retail', 'Manufacturing']
        
    def generate_accounts(self, num_accounts: int = 1000) -> pd.DataFrame:
        data = []
        
        for _ in range(num_accounts):
            account_id = f"ACC-{str(uuid.uuid4())[:8].upper()}"
            segment = np.random.choice(self.segments, p=[0.1, 0.3, 0.6]) # Mostly SMB
            region = np.random.choice(self.regions, p=[0.5, 0.3, 0.2]) # Mostly AMER
            sub_region = np.random.choice(self.sub_regions[region])
            industry = np.random.choice(self.industries)
            
            # Firmographics based on segment
            if segment == 'Enterprise':
                employee_count = int(np.random.lognormal(mean=9.2, sigma=1.0)) # ~10k mean
                annual_revenue = np.random.lognormal(mean=21.0, sigma=1.0) # ~$1.3B
                tam_base = 500_000
            elif segment == 'Mid-Market':
                employee_count = int(np.random.lognormal(mean=6.2, sigma=0.8)) # ~500 mean
                annual_revenue = np.random.lognormal(mean=17.5, sigma=0.8) # ~$40M
                tam_base = 100_000
            else: # SMB
                employee_count = int(np.random.lognormal(mean=3.9, sigma=0.8)) # ~50 mean
                annual_revenue = np.random.lognormal(mean=14.5, sigma=0.8) # ~$2M
                tam_base = 20_000
                
            # Ensure logical mins
            employee_count = max(10, employee_count)
            annual_revenue = max(500_000, annual_revenue)
            
            # Financials
            estimated_tam = tam_base * np.random.uniform(0.5, 2.5)
            # 40% are existing customers
            is_customer = np.random.rand() < 0.4
            historical_spend = estimated_tam * np.random.uniform(0.1, 0.8) if is_customer else 0.0
            
            # Intent & Activity
            g2_intent_score = np.clip(np.random.normal(50, 20), 0, 100)
            crm_activity_score = np.clip(np.random.normal(30, 25), 0, 100)
            
            # Existing Relationships (for override testing)
            # 10% of accounts have a strong locked-in relationship with a specific rep
            existing_rep = None
            if np.random.rand() < 0.1:
                existing_rep = f"REP-{random.randint(100, 999)}"
                
            data.append({
                'Account_ID': account_id,
                'Account_Name': f"{industry} Corp {str(uuid.uuid4())[:4].upper()}",
                'Account_Segment': segment,
                'Region': region,
                'Sub_Region': sub_region,
                'Industry': industry,
                'Employee_Count': employee_count,
                'Annual_Revenue': annual_revenue,
                'Estimated_TAM': estimated_tam,
                'Historical_Spend': historical_spend,
                'G2_Intent_Score': round(g2_intent_score, 1),
                'CRM_Activity_Score': round(crm_activity_score, 1),
                'Existing_Relationship_Rep_ID': existing_rep
            })
            
        return pd.DataFrame(data)

if __name__ == "__main__":
    generator = B2BDataGenerator()
    df = generator.generate_accounts(10)
    print("Generated 10 sample accounts:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
