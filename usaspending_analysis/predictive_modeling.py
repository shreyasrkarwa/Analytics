import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

sns.set_theme(style="whitegrid")

print("Loading data for Top-N Corporate Scaling pipeline...")
df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

df = df.dropna(subset=['Duration (Days)', 'Award Amount'])
df = df[df['Duration (Days)'] >= 0]

# --- Top-N Corporate Scaling ---
# Calculate the total funding secured by each vendor over the entire 7-year dataset
vendor_sums = df.groupby('Recipient Name')['Award Amount'].sum().sort_values(ascending=False)

top_10 = vendor_sums.head(10).index.tolist()
top_25 = vendor_sums.head(25).index.tolist()
top_50 = vendor_sums.head(50).index.tolist()

df['Is_Top_10'] = df['Recipient Name'].apply(lambda x: 1 if x in top_10 else 0)
df['Is_Top_25'] = df['Recipient Name'].apply(lambda x: 1 if x in top_25 else 0)
df['Is_Top_50'] = df['Recipient Name'].apply(lambda x: 1 if x in top_50 else 0)

def is_dod(agency_str):
    upper = str(agency_str).upper()
    return 1 if any(a in upper for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE']) else 0

df['Is_DoD'] = df['Awarding Agency'].apply(is_dod)

# Target Variable: 5-Year Federal Sector Barrier Survival
y = (df['Duration (Days)'] >= 1825).astype(int)

# --- Random Forest Multi-Tier Testing ---
# Testing to see if broadening the "Incumbency" definition to Mid-Tiers changes the predictive outcome
scenarios = {
    'Top 10 Mega-Vendors': ['Award Amount', 'Is_Top_10', 'Is_DoD'],
    'Top 25 Integrators': ['Award Amount', 'Is_Top_25', 'Is_DoD'],
    'Top 50 Major Vendors': ['Award Amount', 'Is_Top_50', 'Is_DoD']
}

results = []
print("\n--- INCUMBENCY SCALE EVALUATION: PREDICTING 5-YEAR SURVIVAL ---")

for label, features in scenarios.items():
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
    clf.fit(X_train, y_train)
    
    importances = clf.feature_importances_
    results.append({
        'Vendor Tier': label,
        'Financial Scope Importance': importances[0],
        'Corporate Scale Importance': importances[1],
        'DoD Importance': importances[2]
    })
    print(f"[{label:<22}] -> Financial Scope: {importances[0]*100:4.1f}% | Corp Scale: {importances[1]*100:4.1f}% | DoD: {importances[2]*100:4.1f}%")

results_df = pd.DataFrame(results)

# --- Visualization ---
artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"

plt.figure(figsize=(10, 5))
vendor_importance = results_df[['Vendor Tier', 'Corporate Scale Importance']]
sns.barplot(data=vendor_importance, x='Vendor Tier', y='Corporate Scale Importance', palette='rocket')
plt.title('The Limits of Federal Incumbency: Predictive Power of Corporate Size on 5-Year Contract Survival')
plt.ylabel('Random Forest Gini Importance (Fractional)')
plt.xlabel('Corporate Scale Tier Definition')

# Add percentage labels on top of the bars
for index, row in vendor_importance.iterrows():
    plt.text(index, row['Corporate Scale Importance'], f"{row['Corporate Scale Importance']*100:.2f}%", color='black', ha="center", va="bottom")

plt.savefig(os.path.join(artifact_dir, 'top_n_vendor_importance.png'), bbox_inches='tight')
plt.close()

print("\nSUCCESS: Top-N evaluation complete and visualizations exported.")
