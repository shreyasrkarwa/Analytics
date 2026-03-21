import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

# Load data
print("Loading data...")
df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')

# Clean and Engineer Features
df['Start Date'] = pd.to_datetime(df['Start Date'])
df['End Date'] = pd.to_datetime(df['End Date'])
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

# Drop anomalies
df = df[df['Duration (Days)'] >= 0]

# --- Reseller vs Direct Logic ---
# Major federal IT Value-Added Resellers (VARs)
RESELLERS = ['CARAHSOFT', 'CDW', 'DLT', 'MYTHICS', 'FOUR INC', 'THUNDERCAT', 'SHI INTERNATIONAL', 'GOVCONNECTION', 'IMMIX']

def classify_vendor(name):
    # Ensure name is a string and uppercase for matching
    name_upper = str(name).upper()
    for r in RESELLERS:
        if r in name_upper:
            return 'Value-Added Reseller (VAR)'
    return 'Direct OEM / Integrator'

df['Vendor Type'] = df['Recipient Name'].apply(classify_vendor)

# --- Stats ---
print("\n--- SUMMARY STATISTICS ---")
print(f"Total IT Contracts Analyzed: {len(df)}")
print(f"Contracts Awarded to VARs: {(df['Vendor Type'] == 'Value-Added Reseller (VAR)').sum()}")
print(f"Contracts Awarded Direct: {(df['Vendor Type'] == 'Direct OEM / Integrator').sum()}")

print("\n--- DURATION STATS (Days) ---")
print(df.groupby('Vendor Type')['Duration (Days)'].describe().round(2))

print("\n--- FINANCIAL STATS (Award Amount in USD) ---")
print(df.groupby('Vendor Type')['Award Amount'].describe().round(2))

# --- Plots ---
artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Duration (Days)', hue='Vendor Type', bins=40, kde=True, element='step')
plt.title('Distribution of Federal IT Contract Durations: Direct vs. Resellers')
plt.xlabel('Duration (Days)')
plt.ylabel('Count of Contracts')
plt.savefig(os.path.join(artifact_dir, 'duration_hist_reseller.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Vendor Type', y='Duration (Days)', data=df, hue='Vendor Type', palette='Set2', legend=False)
plt.title('Contract Longevity: Direct OEMs vs. Value-Added Resellers')
plt.xlabel('Vendor Classification')
plt.ylabel('Duration (Days)')
plt.savefig(os.path.join(artifact_dir, 'duration_boxplot_reseller.png'), bbox_inches='tight')
plt.close()

print("\nSUCCESS: EDA Visualizations generated in artifact directory.")
