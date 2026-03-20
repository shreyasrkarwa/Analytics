import requests
import pandas as pd
import time
import os

API_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"

# Targeting core federal IT-related NAICS codes
# 541511: Custom Computer Programming
# 541512: Computer Systems Design
# 511210: Software Publishers
# 518210: Data Processing/Hosting
TARGET_NAICS = ["541511", "541512", "511210", "518210"]

# Generate Fiscal Years 2018 to 2024 definitions
fiscal_years = []
for year in range(2018, 2025):
    fiscal_years.append({
        "year": year,
        # Default US Federal Fiscal Year: Oct 1 of previous year to Sep 30 of target year
        "start_date": f"{year-1}-10-01",
        "end_date": f"{year}-09-30"
    })

def fetch_top_contracts_for_period(start_date, end_date, max_pages=10):
    all_results = []
    
    # max_pages limits how many contracts we pull per FY to avoid memory/rate-limit issues
    # max_pages=10 with limit=100 means we pull the top 1,000 highly-funded contracts per year.
    for page in range(1, max_pages + 1):
        print(f"  Fetching page {page} for period {start_date} to {end_date}...")
        payload = {
            "filters": {
                "award_type_codes": ["A", "B", "C", "D"], 
                "naics_codes": {"require": TARGET_NAICS},
                "time_period": [{"start_date": start_date, "end_date": end_date}]
            },
            "fields": [
                "Award ID", "Recipient Name", "Start Date", "End Date", 
                "Award Amount", "Description", "Awarding Agency",
                "Subaward Count", "Subaward Total Amount" 
            ],
            "page": page,
            "limit": 100,
            "sort": "Award Amount",
            "order": "desc"
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                # No more results on this page
                break
                
            all_results.extend(results)
            
            # Simple rate limiting/politeness throttle to respect the government API
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching page {page}: {e}")
            break
            
    return all_results

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    all_contracts = []
    
    print("Initiating longitudinal USASpending data extraction (FY2018 - FY2024)...")
    
    for fy in fiscal_years:
        print(f"\nProcessing FY {fy['year']}...")
        # Collecting the top 1,000 highly-funded critical IT contracts for each year
        fy_results = fetch_top_contracts_for_period(fy['start_date'], fy['end_date'], max_pages=10) 
        all_contracts.extend(fy_results)
        print(f"Successfully gathered {len(fy_results)} contracts for FY {fy['year']}.")
        
    if all_contracts:
        df = pd.DataFrame(all_contracts)
        # Drop duplicates just in case an award ID spanned overlapping API calls
        df.drop_duplicates(subset=["Award ID"], inplace=True)
        
        output_file = "data/longitudinal_it_contracts_fy18_fy24.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData extraction complete! Total unique records: {len(df)}")
        print(f"Dataset successfully saved to: {os.path.abspath(output_file)}")
    else:
        print("Failed to pull dataset.")
