import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

sns.set_theme(style="whitegrid")

df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

df = df.dropna(subset=['Duration (Days)', 'Award Amount'])
df = df[df['Duration (Days)'] >= 0]

vendor_sums = df.groupby('Recipient Name')['Award Amount'].sum().sort_values(ascending=False)
df['Top_10'] = df['Recipient Name'].apply(lambda x: 1 if x in vendor_sums.head(10).index else 0)
df['Top_25'] = df['Recipient Name'].apply(lambda x: 1 if x in vendor_sums.head(25).index else 0)
df['Top_50'] = df['Recipient Name'].apply(lambda x: 1 if x in vendor_sums.head(50).index else 0)

def is_dod(agency_str):
    upper = str(agency_str).upper()
    return 1 if any(a in upper for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE']) else 0

df['Is_DoD'] = df['Awarding Agency'].apply(is_dod)

times = {
    '1-Year Base': 365,
    '3-Year Options': 1095,
    '5-Year Ceiling': 1825,
    '10-Year IDIQ': 3650
}

tiers = {
    'Top 10 Mega-Vendors': 'Top_10',
    'Top 25 Integrators': 'Top_25',
    'Top 50 Major Vendors': 'Top_50'
}

results = []

for time_label, days in times.items():
    y = (df['Duration (Days)'] >= days).astype(int)
    if y.sum() < 10 or y.sum() == len(y): continue
        
    for tier_label, tier_feature in tiers.items():
        X = df[['Award Amount', tier_feature, 'Is_DoD']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
        clf.fit(X_train, y_train)
        
        results.append({
            'Threshold': time_label,
            'Vendor Category': tier_label,
            'Predictive Importance (%)': clf.feature_importances_[1] * 100
        })

res_df = pd.DataFrame(results)

artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"
plt.figure(figsize=(10, 5))

# Creating a beautiful Line Plot highlighting the flatline
sns.lineplot(data=res_df, x='Threshold', y='Predictive Importance (%)', hue='Vendor Category', marker='o', linewidth=3, palette='magma')
plt.title('The Flatline of Incumbency: Corporate Scale Importance Over Time')
plt.ylabel('Predictive Power (%)')
plt.xlabel('Contract Survival Boundary')
plt.ylim(0, 15) # Zooming in on the Y-Axis to emphasize the insignificance

plt.savefig(os.path.join(artifact_dir, 'temporal_incumbency_tracking.png'), bbox_inches='tight')
plt.close()

print("SUCCESS: Temporal Tracking Plot Generated.")
