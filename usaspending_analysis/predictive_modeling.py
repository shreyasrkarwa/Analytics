import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

sns.set_theme(style="whitegrid")

print("Loading data for Machine Learning pipeline...")
df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

# --- Feature Engineering ---
df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

# Clean base DataFrame
df = df.dropna(subset=['Duration (Days)', 'Award Amount'])
df = df[df['Duration (Days)'] >= 0]

# Classifications based on previous EDA
top_vendors = df.groupby('Recipient Name')['Award Amount'].sum().nlargest(10).index.tolist()
df['Is_Mega_Vendor'] = df['Recipient Name'].apply(lambda x: 1 if x in top_vendors else 0)

def is_dod(agency_str):
    upper_str = str(agency_str).upper()
    return 1 if any(a in upper_str for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE']) else 0

df['Is_DoD'] = df['Awarding Agency'].apply(is_dod)

# Define Target Variable: Long-Term Contract (>= 5 Years / 1825 Days)
# Does the contract beat the standard 5-year federal ceiling?
df['Long_Term_Contract'] = (df['Duration (Days)'] >= 1825).astype(int)

print(f"Target Distribution: {df['Long_Term_Contract'].sum()} Long-Term vs {len(df) - df['Long_Term_Contract'].sum()} Standard/Short-Term")

# --- Model Setup ---
# Using Award Amount, Vendor Status, and Agency Classification as features
features = ['Award Amount', 'Is_Mega_Vendor', 'Is_DoD']
X = df[features]
y = df['Long_Term_Contract']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate Random Forest with balanced class weights to handle any target skew
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n--- RANDOM FOREST RESULTS ---")
print(f"Accuracy predicting 5+ Year Contracts: {acc * 100:.1f}%")
print("Classification Report:\n", report)

# --- Feature Importance Visualization ---
importances = clf.feature_importances_
feature_names = ['Total Award Amount ($)', 'Mega-Vendor Status (Top 10)', 'DoD vs Civilian Agency']
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n--- FEATURE IMPORTANCES ---")
print(feat_df)

artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"
plt.figure(figsize=(10, 4))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='magma')
plt.title('Random Forest Decision Drivers: Predicting 5+ Year IT Contracts', pad=15)
plt.xlabel('Relative Importance (Gini Importance)')
plt.ylabel('Algorithmic Feature')
plt.savefig(os.path.join(artifact_dir, 'feature_importance.png'), bbox_inches='tight')
plt.close()

print("\nML pipeline complete and feature importance plotted.")
