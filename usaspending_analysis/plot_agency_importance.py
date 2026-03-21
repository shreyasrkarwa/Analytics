import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style="whitegrid")

data = {
    'Agency Sector': ['Civilian Agencies', 'Civilian Agencies', 'Department of Defense (DoD)', 'Department of Defense (DoD)'],
    'Feature': ['Initial Financial Scope', 'Corporate Scale (Top 10)', 'Initial Financial Scope', 'Corporate Scale (Top 10)'],
    'Predictive Importance (%)': [96.8, 3.2, 91.7, 8.3]
}
df = pd.DataFrame(data)

artifact_dir = "/Users/shreyasrkarwa/.gemini/antigravity/brain/3818d78f-de3f-4c7f-94be-f43792dd0b66/"
plt.figure(figsize=(9, 5))

sns.barplot(data=df, x='Agency Sector', y='Predictive Importance (%)', hue='Feature', palette='mako')
plt.title('Feature Importance: Civilian vs. Defense Sectors (5-Year Contract Survival)')
plt.ylabel('Random Forest Gini Importance (%)')
plt.xlabel('Government Sector')

# Add percentage labels
for p in plt.gca().patches:
    plt.gca().annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.savefig(os.path.join(artifact_dir, 'dod_vs_civ_importance.png'), bbox_inches='tight')
plt.close()
print("SUCCESS: DoD vs Civilian Graph generated.")
