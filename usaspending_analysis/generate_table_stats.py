import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

# Unique vendors
unique_vendors = df['Recipient Name'].nunique()
print(f"UNIQUE VENDORS in 1730 Dataset: {unique_vendors}")

# Top 10 Table Data
top_10_names = df.groupby('Recipient Name')['Award Amount'].sum().nlargest(10).index.tolist()

table_data = []
for name in top_10_names:
    vendor_df = df[df['Recipient Name'] == name]
    total_amount = vendor_df['Award Amount'].sum() / 1e9
    count = len(vendor_df)
    min_date = vendor_df['Start Date'].min().strftime('%Y-%m-%d')
    max_date = vendor_df['Start Date'].max().strftime('%Y-%m-%d')
    table_data.append(f"| {name} | ${total_amount:.2f}B | {count} | {min_date} | {max_date} |")

print("\nTABLE ROWS:")
print("| Vendor | Total Funding | Contract Count | First Contract | Last Contract |")
print("|---|---|---|---|---|")
for row in table_data:
    print(row)

# Hypothesis test: DoD vs Civilian isolated isolation
def is_dod(agency_str):
    upper = str(agency_str).upper()
    return any(a in upper for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE'])

df['IS_DOD_FLAG'] = df['Awarding Agency'].apply(is_dod)
df_dod = df[df['IS_DOD_FLAG'] == True].copy()
df_civ = df[df['IS_DOD_FLAG'] == False].copy()

def get_rf_importance(data, name):
    data = data.dropna(subset=['Duration (Days)', 'Award Amount'])
    data = data[data['Duration (Days)'] >= 0]
    
    vendor_sums = data.groupby('Recipient Name')['Award Amount'].sum().sort_values(ascending=False)
    top_10 = vendor_sums.head(10).index.tolist()
    data['Is_Top_10'] = data['Recipient Name'].apply(lambda x: 1 if x in top_10 else 0)
    
    y = (data['Duration (Days)'] >= 1825).astype(int)
    X = data[['Award Amount', 'Is_Top_10']]
    
    if y.sum() < 5 or y.sum() == len(y):
        print(f"[{name}] Not enough variance to test.")
        return
        
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
    clf.fit(X, y)
    print(f"[{name}] Scope Importance: {clf.feature_importances_[0]*100:.1f}%, Mega-Vendor Scale Importance: {clf.feature_importances_[1]*100:.1f}%")

print("\n--- DoD vs CIVILIAN SEPARATED HYPOTHESIS TEST (5-Year Threshold) ---")
get_rf_importance(df_dod, "Department of Defense (DoD)")
get_rf_importance(df_civ, "Civilian Agencies (HHS, DHS, etc)")
