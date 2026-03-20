import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

sns.set_theme(style="whitegrid")

print("Loading data for advanced splits...")
df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

# Clean rows
df = df.dropna(subset=['Duration (Days)'])
df = df[df['Duration (Days)'] >= 0]

# --- SPLIT 1: Mega-Vendors vs Rest ---
# Top 10 vendors by total massive Award Amount in this sample
top_vendors = df.groupby('Recipient Name')['Award Amount'].sum().nlargest(10).index.tolist()
df['Vendor Class'] = df['Recipient Name'].apply(lambda x: 'Top 10 Mega-Vendor' if x in top_vendors else 'Rest of Vendors')

print("\n--- SPLIT 1: MEGA-VENDORS VS OTHERS ---")
print(f"Total Mega-Vendor Contracts: {(df['Vendor Class'] == 'Top 10 Mega-Vendor').sum()}")
print(f"Total Other Vendor Contracts: {(df['Vendor Class'] == 'Rest of Vendors').sum()}")
print(df.groupby('Vendor Class')['Duration (Days)'].describe().round(2))

# --- SPLIT 3: DoD vs Civilian ---
def extract_agency(agency_str):
    try:
        # Handle string matching for DoD markers
        upper_str = str(agency_str).upper()
        if 'DEFENSE' in upper_str or 'ARMY' in upper_str or 'NAVY' in upper_str or 'AIR FORCE' in upper_str:
            return 'Department of Defense (DoD)'
        return 'Civilian Agencies'
    except:
        return 'Civilian Agencies'

df['Agency Sector'] = df['Awarding Agency'].apply(extract_agency)

print("\n--- SPLIT 3: DOD VS CIVILIAN ---")
print(f"Total DoD Contracts: {(df['Agency Sector'] == 'Department of Defense (DoD)').sum()}")
print(f"Total Civilian Contracts: {(df['Agency Sector'] == 'Civilian Agencies').sum()}")
print(df.groupby('Agency Sector')['Duration (Days)'].describe().round(2))

# --- Plots ---
artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"

plt.figure(figsize=(8, 5))
sns.boxplot(x='Vendor Class', y='Duration (Days)', data=df, hue='Vendor Class', palette='viridis', legend=False)
plt.title('Contract Longevity: Top 10 Mega-Vendors vs. The Market')
plt.xlabel('Vendor Classification')
plt.ylabel('Duration (Days)')
plt.savefig(os.path.join(artifact_dir, 'duration_split1.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Agency Sector', y='Duration (Days)', data=df, hue='Agency Sector', palette='magma', legend=False)
plt.title('Contract Longevity: DoD vs. Civilian Agencies')
plt.xlabel('Agency Sector')
plt.ylabel('Duration (Days)')
plt.savefig(os.path.join(artifact_dir, 'duration_split3.png'), bbox_inches='tight')
plt.close()

print("\nSUCCESS: Split Visualizations generated.")
