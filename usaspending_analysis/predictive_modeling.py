import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

sns.set_theme(style="whitegrid")

print("Loading data for dynamic Multi-Threshold Machine Learning pipeline...")
df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

# --- Feature Engineering ---
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

df = df.dropna(subset=['Duration (Days)', 'Award Amount'])
df = df[df['Duration (Days)'] >= 0]

top_vendors = df.groupby('Recipient Name')['Award Amount'].sum().nlargest(10).index.tolist()
df['Is_Mega_Vendor'] = df['Recipient Name'].apply(lambda x: 1 if x in top_vendors else 0)

def is_dod(agency_str):
    upper_str = str(agency_str).upper()
    return 1 if any(a in upper_str for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE']) else 0

df['Is_DoD'] = df['Awarding Agency'].apply(is_dod)

features = ['Award Amount', 'Is_Mega_Vendor', 'Is_DoD']
X = df[features]

# --- Dynamic Threshold Logic ---
# Testing survival across transactional, standard, ceiling, and IDIQ boundaries
thresholds = {
    '1-Year Threshold': 365,
    '3-Year Threshold': 1095,
    '5-Year Federal Ceiling': 1825,
    '10-Year Mega-Contract': 3650
}

results = []
print("\n--- RANDOM FOREST: DYNAMIC THRESHOLD IMPORTANCE ---")
for label, days in thresholds.items():
    y = (df['Duration (Days)'] >= days).astype(int)
    
    # Filter out scenarios where data is too sparse
    if y.sum() < 10 or y.sum() == len(y):
        print(f"Skipping {label} due to zero variance or extreme class imbalance.")
        continue
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
    clf.fit(X_train, y_train)
    
    importances = clf.feature_importances_
    print(f"[{label}] -> Scope: {importances[0]:.3f} | Mega-Vendor: {importances[1]:.3f} | DoD: {importances[2]:.3f}")
    
    results.append({
        'Threshold': label,
        'Award Amount Impact': importances[0],
        'Mega-Vendor Impact': importances[1],
        'DoD Impact': importances[2]
    })

results_df = pd.DataFrame(results)

# --- Visualization ---
artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"
results_melted = results_df.melt(id_vars='Threshold', var_name='Feature', value_name='Relative Importance')

plt.figure(figsize=(12, 6))
sns.barplot(data=results_melted, x='Threshold', y='Relative Importance', hue='Feature', palette='magma')
plt.title('The Shift in Statistical Power: How ML Feature Importance Evolves Over Time')
plt.xlabel('Contract Survival Threshold (Time)')
plt.ylabel('Predictive Power (Gini Importance)')
plt.legend(title='Algorithmic Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(artifact_dir, 'multi_threshold_importance.png'), bbox_inches='tight')
plt.close()

print("\nSUCCESS: Multi-threshold model output exported.")
