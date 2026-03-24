import pandas as pd
import numpy as np
import random
import os

def generate_hierarchy():
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    regions = ['NA', 'EMEA', 'APAC']
    data = []
    
    for r_idx, region in enumerate(regions):
        # 1-2 RVPs per region
        num_rvps = random.randint(1, 2)
        for rvp_idx in range(num_rvps):
            rvp_name = f"RVP_{region}_{rvp_idx+1}"
            
            # 2-3 Directors per RVP
            num_dirs = random.randint(2, 3)
            for d_idx in range(num_dirs):
                dir_name = f"Dir_{rvp_name}_{d_idx+1}"
                
                # 3-5 Managers per Director
                num_mgrs = random.randint(3, 5)
                for m_idx in range(num_mgrs):
                    mgr_name = f"Mgr_{dir_name}_{m_idx+1}"
                    
                    # 5-8 ICs per Manager
                    num_ics = random.randint(5, 8)
                    for ic_idx in range(num_ics):
                        ic_name = f"IC_{mgr_name}_{ic_idx+1}"
                        
                        # Generate random past 4 quarters performance (e.g. baseline of $100k-$500k a quarter output)
                        baseline = random.uniform(100000, 500000)
                        
                        # Add variance and standard enterprise tech seasonality (Q4 Hockey Stick)
                        q1 = baseline * random.uniform(0.7, 1.2)
                        q2 = baseline * random.uniform(0.8, 1.3)
                        q3 = baseline * random.uniform(0.9, 1.4) 
                        q4 = baseline * random.uniform(1.2, 2.0) 
                        
                        # Generating current pipeline coverage (usually 1.5x up to 4.0x of baseline)
                        current_pipe = (q1+q2+q3+q4)/4 * random.uniform(1.5, 4.0)
                        
                        data.append({
                            'Global': 'Global_Corp',
                            'Region': region,
                            'RVP': rvp_name,
                            'Director': dir_name,
                            'Manager': mgr_name,
                            'IC': ic_name,
                            'Q1_Attainment': round(q1, 2),
                            'Q2_Attainment': round(q2, 2),
                            'Q3_Attainment': round(q3, 2),
                            'Q4_Attainment': round(q4, 2),
                            'Current_Pipeline': round(current_pipe, 2)
                        })
                        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs('tests/data', exist_ok=True)
    out_path = 'tests/data/synthetic_hierarchy.csv'
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic hierarchy with {len(df)} ICs at {out_path}.")
    return df

if __name__ == '__main__':
    generate_hierarchy()
