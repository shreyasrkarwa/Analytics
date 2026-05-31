"""
Enhanced Validation Pipeline for Federal IT Contract Longevity Study
====================================================================
This script extends the original Random Forest analysis with comprehensive
validation metrics required for peer-reviewed publication:

- Stratified K-Fold Cross-Validation (k=10)
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrices
- ROC-AUC Curves
- Permutation Importance (to address Gini bias per Strobl et al., 2007)
- Statistical significance testing (Mann-Whitney U, Chi-Square)
- Temporal threshold sweep with full metrics

Run from the project root:
    cd usaspending_analysis
    pip install scikit-learn pandas matplotlib seaborn numpy
    python enhanced_validation.py

All outputs are saved to: outputs/validation/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

# ─── Configuration ───────────────────────────────────────────────────────────
OUTPUT_DIR = 'outputs/validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ESTIMATORS = 100
RANDOM_STATE = 42
MAX_DEPTH = 5
TEST_SIZE = 0.25
CV_FOLDS = 10
TEMPORAL_THRESHOLDS = {
    '1-Year (365 days)': 365,
    '3-Year (1,095 days)': 1095,
    '5-Year (1,825 days)': 1825,
    '10-Year (3,650 days)': 3650,
}
VENDOR_TIERS = {
    'Top 10 Mega-Vendors': 10,
    'Top 25 Integrators': 25,
    'Top 50 Major Vendors': 50,
}

# ─── Data Loading & Preprocessing ────────────────────────────────────────────
print("=" * 80)
print("ENHANCED VALIDATION PIPELINE")
print("Federal IT Contract Longevity Study")
print("=" * 80)

df = pd.read_csv('data/longitudinal_it_contracts_fy18_fy24.csv')
print(f"\nRaw dataset: {len(df)} records")

df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
df['Duration (Days)'] = (df['End Date'] - df['Start Date']).dt.days
df['Award Amount'] = pd.to_numeric(df['Award Amount'], errors='coerce')

df = df.dropna(subset=['Duration (Days)', 'Award Amount'])
df = df[df['Duration (Days)'] >= 0]
print(f"After cleaning: {len(df)} records")

# Vendor tier assignment
vendor_sums = df.groupby('Recipient Name')['Award Amount'].sum().sort_values(ascending=False)
for tier_name, n in VENDOR_TIERS.items():
    col = f'Is_Top_{n}'
    top_vendors = vendor_sums.head(n).index.tolist()
    df[col] = df['Recipient Name'].apply(lambda x, tv=top_vendors: 1 if x in tv else 0)

# DoD classification
def is_dod(agency_str):
    upper = str(agency_str).upper()
    return 1 if any(a in upper for a in ['DEFENSE', 'ARMY', 'NAVY', 'AIR FORCE']) else 0

df['Is_DoD'] = df['Awarding Agency'].apply(is_dod)

# ─── Section 1: Descriptive Statistics ────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 1: DESCRIPTIVE STATISTICS")
print("=" * 80)

desc_stats = {
    'total_records': len(df),
    'unique_vendors': df['Recipient Name'].nunique(),
    'unique_agencies': df['Awarding Agency'].nunique(),
    'dod_count': int(df['Is_DoD'].sum()),
    'civilian_count': int((df['Is_DoD'] == 0).sum()),
    'award_amount': {
        'mean': float(df['Award Amount'].mean()),
        'median': float(df['Award Amount'].median()),
        'std': float(df['Award Amount'].std()),
        'min': float(df['Award Amount'].min()),
        'max': float(df['Award Amount'].max()),
    },
    'duration_days': {
        'mean': float(df['Duration (Days)'].mean()),
        'median': float(df['Duration (Days)'].median()),
        'std': float(df['Duration (Days)'].std()),
    },
}

for threshold_name, days in TEMPORAL_THRESHOLDS.items():
    count = int((df['Duration (Days)'] >= days).sum())
    desc_stats[f'contracts_above_{days}d'] = count
    desc_stats[f'pct_above_{days}d'] = round(count / len(df) * 100, 1)
    print(f"  Contracts >= {threshold_name}: {count} ({count/len(df)*100:.1f}%)")

print(f"\n  Total records: {len(df)}")
print(f"  Unique vendors: {df['Recipient Name'].nunique()}")
print(f"  DoD contracts: {df['Is_DoD'].sum()} | Civilian: {(df['Is_DoD']==0).sum()}")
print(f"  Award Amount - Mean: ${df['Award Amount'].mean():,.0f} | Median: ${df['Award Amount'].median():,.0f}")
print(f"  Duration - Mean: {df['Duration (Days)'].mean():,.0f} days | Median: {df['Duration (Days)'].median():,.0f} days")

# ─── Section 2: Statistical Tests ────────────────────────────────────────────
print("\n" + "=" * 80)
print("SECTION 2: STATISTICAL SIGNIFICANCE TESTS")
print("=" * 80)

stat_tests = {}

# Mann-Whitney U: Do top-10 vendors have significantly different contract durations?
top10_durations = df[df['Is_Top_10'] == 1]['Duration (Days)']
other_durations = df[df['Is_Top_10'] == 0]['Duration (Days)']
u_stat, u_pval = stats.mannwhitneyu(top10_durations, other_durations, alternative='two-sided')
stat_tests['mann_whitney_duration'] = {
    'U_statistic': float(u_stat),
    'p_value': float(u_pval),
    'top10_median': float(top10_durations.median()),
    'other_median': float(other_durations.median()),
}
print(f"\n  Mann-Whitney U (Duration: Top-10 vs Others):")
print(f"    U = {u_stat:.1f}, p = {u_pval:.6f}")
print(f"    Top-10 median: {top10_durations.median():.0f} days | Others median: {other_durations.median():.0f} days")

# Chi-Square: Is vendor tier associated with 5-year survival?
contingency = pd.crosstab(df['Is_Top_10'], (df['Duration (Days)'] >= 1825).astype(int))
chi2, chi_p, chi_dof, chi_expected = stats.chi2_contingency(contingency)
stat_tests['chi_square_5yr'] = {
    'chi2': float(chi2),
    'p_value': float(chi_p),
    'dof': int(chi_dof),
}
print(f"\n  Chi-Square (Top-10 vs 5-Year Survival):")
print(f"    chi2 = {chi2:.4f}, p = {chi_p:.6f}, dof = {chi_dof}")

# DoD vs Civilian duration comparison
dod_dur = df[df['Is_DoD'] == 1]['Duration (Days)']
civ_dur = df[df['Is_DoD'] == 0]['Duration (Days)']
u_dod, p_dod = stats.mannwhitneyu(dod_dur, civ_dur, alternative='two-sided')
stat_tests['mann_whitney_dod_civ'] = {
    'U_statistic': float(u_dod),
    'p_value': float(p_dod),
    'dod_median': float(dod_dur.median()),
    'civilian_median': float(civ_dur.median()),
}
print(f"\n  Mann-Whitney U (Duration: DoD vs Civilian):")
print(f"    U = {u_dod:.1f}, p = {p_dod:.6f}")
print(f"    DoD median: {dod_dur.median():.0f} days | Civilian median: {civ_dur.median():.0f} days")

# ─── Section 3: Core RF Model with Full Validation ──────────────────────────
print("\n" + "=" * 80)
print("SECTION 3: RANDOM FOREST WITH CROSS-VALIDATION (5-YEAR THRESHOLD)")
print("=" * 80)

all_results = {}

for tier_label, n in VENDOR_TIERS.items():
    tier_col = f'Is_Top_{n}'
    features = ['Award Amount', tier_col, 'Is_DoD']
    X = df[features].copy()
    y = (df['Duration (Days)'] >= 1825).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_depth=MAX_DEPTH
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # Stratified 10-fold CV
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_accuracy = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(clf, X, y, cv=skf, scoring='f1')
    cv_roc = cross_val_score(clf, X, y, cv=skf, scoring='roc_auc')

    # Gini Importance
    gini_imp = dict(zip(features, clf.feature_importances_))

    # Permutation Importance (addresses Strobl et al. 2007 bias concern)
    perm_result = permutation_importance(
        clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring='accuracy'
    )
    perm_imp = dict(zip(features, perm_result.importances_mean))
    perm_std = dict(zip(features, perm_result.importances_std))

    result = {
        'tier': tier_label,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'class_balance': {
            'positive (>=5yr)': int(y.sum()),
            'negative (<5yr)': int((y == 0).sum()),
            'positive_pct': round(y.mean() * 100, 1),
        },
        'holdout_metrics': {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(roc, 4),
        },
        'confusion_matrix': cm.tolist(),
        'cv_accuracy': {
            'mean': round(cv_accuracy.mean(), 4),
            'std': round(cv_accuracy.std(), 4),
            'folds': cv_accuracy.round(4).tolist(),
        },
        'cv_f1': {
            'mean': round(cv_f1.mean(), 4),
            'std': round(cv_f1.std(), 4),
        },
        'cv_roc_auc': {
            'mean': round(cv_roc.mean(), 4),
            'std': round(cv_roc.std(), 4),
        },
        'gini_importance': {k: round(v, 4) for k, v in gini_imp.items()},
        'permutation_importance': {k: round(v, 4) for k, v in perm_imp.items()},
        'permutation_std': {k: round(v, 4) for k, v in perm_std.items()},
    }
    all_results[tier_label] = result

    print(f"\n  ─── {tier_label} ───")
    print(f"  Train/Test: {len(X_train)}/{len(X_test)} | Class balance: {y.mean()*100:.1f}% positive")
    print(f"  Holdout => Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | AUC: {roc:.3f}")
    print(f"  10-Fold CV => Acc: {cv_accuracy.mean():.3f} (+/- {cv_accuracy.std():.3f})")
    print(f"  10-Fold CV => F1:  {cv_f1.mean():.3f} (+/- {cv_f1.std():.3f})")
    print(f"  10-Fold CV => AUC: {cv_roc.mean():.3f} (+/- {cv_roc.std():.3f})")
    print(f"  Confusion Matrix:\n    {cm}")
    print(f"  Gini Importance:       {gini_imp}")
    print(f"  Permutation Importance: {perm_imp}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['< 5 Years', '>= 5 Years']).plot(ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix: {tier_label}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{n}.png'), dpi=150)
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {tier_label}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'roc_curve_{n}.png'), dpi=150)
    plt.close()

# ─── Section 4: Gini vs Permutation Importance Comparison ────────────────────
print("\n" + "=" * 80)
print("SECTION 4: GINI vs PERMUTATION IMPORTANCE COMPARISON")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (tier_label, result) in enumerate(all_results.items()):
    features = list(result['gini_importance'].keys())
    gini_vals = [result['gini_importance'][f] for f in features]
    perm_vals = [result['permutation_importance'][f] for f in features]

    x = np.arange(len(features))
    width = 0.35
    axes[idx].bar(x - width/2, gini_vals, width, label='Gini', color='#2196F3')
    axes[idx].bar(x + width/2, perm_vals, width, label='Permutation', color='#FF9800')
    axes[idx].set_xticks(x)
    short_labels = ['Financial Scope', 'Corp. Scale', 'DoD Sector']
    axes[idx].set_xticklabels(short_labels, rotation=15)
    axes[idx].set_title(tier_label)
    axes[idx].set_ylabel('Importance')
    axes[idx].legend()
    axes[idx].set_ylim(0, 1.0)

    print(f"\n  {tier_label}:")
    for f, g, p in zip(short_labels, gini_vals, perm_vals):
        print(f"    {f:<18s} Gini: {g:.4f}  Permutation: {p:.4f}")

plt.suptitle('Feature Importance: Gini vs Permutation (5-Year Threshold)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'gini_vs_permutation.png'), dpi=150)
plt.close()

# ─── Section 5: Temporal Threshold Sweep with Full Metrics ───────────────────
print("\n" + "=" * 80)
print("SECTION 5: TEMPORAL THRESHOLD SWEEP")
print("=" * 80)

temporal_results = []

for threshold_name, days in TEMPORAL_THRESHOLDS.items():
    y = (df['Duration (Days)'] >= days).astype(int)
    if y.sum() < 10 or y.sum() == len(y):
        print(f"  Skipping {threshold_name} - insufficient class balance")
        continue

    for tier_label, n in VENDOR_TIERS.items():
        tier_col = f'Is_Top_{n}'
        features = ['Award Amount', tier_col, 'Is_DoD']
        X = df[features].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
            class_weight='balanced', max_depth=MAX_DEPTH
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Permutation importance
        perm = permutation_importance(
            clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE
        )

        temporal_results.append({
            'Threshold': threshold_name,
            'Days': days,
            'Tier': tier_label,
            'N': n,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else 0,
            'Gini_Scope': clf.feature_importances_[0],
            'Gini_Scale': clf.feature_importances_[1],
            'Gini_DoD': clf.feature_importances_[2],
            'Perm_Scope': perm.importances_mean[0],
            'Perm_Scale': perm.importances_mean[1],
            'Perm_DoD': perm.importances_mean[2],
            'Positive_Pct': y.mean() * 100,
        })

        print(f"  [{threshold_name}] {tier_label}: Acc={accuracy_score(y_test, y_pred):.3f} "
              f"F1={f1_score(y_test, y_pred, zero_division=0):.3f} "
              f"AUC={roc_auc_score(y_test, y_proba):.3f} "
              f"Gini(Scale)={clf.feature_importances_[1]*100:.1f}% "
              f"Perm(Scale)={perm.importances_mean[1]*100:.1f}%")

temporal_df = pd.DataFrame(temporal_results)

# Temporal sweep plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Gini Scale Importance over time
for tier_label in VENDOR_TIERS:
    subset = temporal_df[temporal_df['Tier'] == tier_label]
    axes[0, 0].plot(subset['Days'], subset['Gini_Scale'] * 100, 'o-', label=tier_label, lw=2)
axes[0, 0].set_title('Gini Importance of Corporate Scale Over Time')
axes[0, 0].set_ylabel('Importance (%)')
axes[0, 0].set_xlabel('Threshold (Days)')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 15)

# Plot 2: Permutation Scale Importance over time
for tier_label in VENDOR_TIERS:
    subset = temporal_df[temporal_df['Tier'] == tier_label]
    axes[0, 1].plot(subset['Days'], subset['Perm_Scale'] * 100, 'o-', label=tier_label, lw=2)
axes[0, 1].set_title('Permutation Importance of Corporate Scale Over Time')
axes[0, 1].set_ylabel('Importance (%)')
axes[0, 1].set_xlabel('Threshold (Days)')
axes[0, 1].legend()
axes[0, 1].set_ylim(-5, 15)

# Plot 3: Model Accuracy over time
for tier_label in VENDOR_TIERS:
    subset = temporal_df[temporal_df['Tier'] == tier_label]
    axes[1, 0].plot(subset['Days'], subset['Accuracy'], 'o-', label=tier_label, lw=2)
axes[1, 0].set_title('Model Accuracy Across Temporal Thresholds')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_xlabel('Threshold (Days)')
axes[1, 0].legend()

# Plot 4: ROC-AUC over time
for tier_label in VENDOR_TIERS:
    subset = temporal_df[temporal_df['Tier'] == tier_label]
    axes[1, 1].plot(subset['Days'], subset['ROC_AUC'], 'o-', label=tier_label, lw=2)
axes[1, 1].set_title('ROC-AUC Across Temporal Thresholds')
axes[1, 1].set_ylabel('ROC-AUC')
axes[1, 1].set_xlabel('Threshold (Days)')
axes[1, 1].legend()

plt.suptitle('Temporal Threshold Sweep: Model Performance & Feature Importance', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_sweep.png'), dpi=150, bbox_inches='tight')
plt.close()

# ─── Section 6: DoD vs Civilian Disaggregation ──────────────────────────────
print("\n" + "=" * 80)
print("SECTION 6: DoD vs CIVILIAN DISAGGREGATED ANALYSIS")
print("=" * 80)

sector_results = {}
for sector_name, sector_val in [('DoD', 1), ('Civilian', 0)]:
    sector_df = df[df['Is_DoD'] == sector_val].copy()
    y = (sector_df['Duration (Days)'] >= 1825).astype(int)

    if y.sum() < 10 or (y == 0).sum() < 10:
        print(f"  Skipping {sector_name} - insufficient class balance")
        continue

    sector_vendor_sums = sector_df.groupby('Recipient Name')['Award Amount'].sum().sort_values(ascending=False)
    top10_sector = sector_vendor_sums.head(10).index.tolist()
    sector_df['Is_Top_10_Sector'] = sector_df['Recipient Name'].apply(lambda x: 1 if x in top10_sector else 0)

    X = sector_df[['Award Amount', 'Is_Top_10_Sector']].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE,
        class_weight='balanced', max_depth=MAX_DEPTH
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE
    )

    skf = StratifiedKFold(n_splits=min(CV_FOLDS, min(y.sum(), (y==0).sum())),
                          shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    cv_f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1')

    sector_results[sector_name] = {
        'n_contracts': len(sector_df),
        'positive_pct': round(y.mean() * 100, 1),
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'f1': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        'gini_scope': round(float(clf.feature_importances_[0]), 4),
        'gini_scale': round(float(clf.feature_importances_[1]), 4),
        'perm_scope': round(float(perm.importances_mean[0]), 4),
        'perm_scale': round(float(perm.importances_mean[1]), 4),
        'cv_accuracy': f"{cv_acc.mean():.3f} +/- {cv_acc.std():.3f}",
        'cv_f1': f"{cv_f1_scores.mean():.3f} +/- {cv_f1_scores.std():.3f}",
        'top10_vendors': top10_sector,
    }

    print(f"\n  ─── {sector_name} ({len(sector_df)} contracts, {y.mean()*100:.1f}% >= 5yr) ───")
    print(f"  Holdout: Acc={accuracy_score(y_test, y_pred):.3f} F1={f1_score(y_test, y_pred):.3f} AUC={roc_auc_score(y_test, y_proba):.3f}")
    print(f"  Gini:  Scope={clf.feature_importances_[0]*100:.1f}% | Scale={clf.feature_importances_[1]*100:.1f}%")
    print(f"  Perm:  Scope={perm.importances_mean[0]*100:.1f}% | Scale={perm.importances_mean[1]*100:.1f}%")
    print(f"  CV Acc: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")

# ─── Section 7: Save All Results ─────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output = {
    'descriptive_statistics': desc_stats,
    'statistical_tests': stat_tests,
    'rf_validation_5yr': {k: {kk: vv for kk, vv in v.items() if kk != 'tier'}
                         for k, v in all_results.items()},
    'temporal_sweep': temporal_df.to_dict('records'),
    'sector_disaggregation': sector_results,
    'model_config': {
        'n_estimators': N_ESTIMATORS,
        'random_state': RANDOM_STATE,
        'max_depth': MAX_DEPTH,
        'test_size': TEST_SIZE,
        'cv_folds': CV_FOLDS,
        'class_weight': 'balanced',
    }
}

with open(os.path.join(OUTPUT_DIR, 'validation_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

# Save temporal results as CSV for easy reference
temporal_df.to_csv(os.path.join(OUTPUT_DIR, 'temporal_sweep_results.csv'), index=False)

print(f"\n  All results saved to: {OUTPUT_DIR}/")
print(f"  - validation_results.json (full metrics)")
print(f"  - temporal_sweep_results.csv")
print(f"  - confusion_matrix_*.png")
print(f"  - roc_curve_*.png")
print(f"  - gini_vs_permutation.png")
print(f"  - temporal_sweep.png")

# ─── Summary Table ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PUBLICATION-READY SUMMARY TABLE (5-Year Threshold)")
print("=" * 80)
print(f"\n{'Vendor Tier':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}  "
      f"{'CV-Acc':>12} {'Gini(Scale)':>12} {'Perm(Scale)':>12}")
print("-" * 110)
for tier_label, result in all_results.items():
    m = result['holdout_metrics']
    cv = result['cv_accuracy']
    g_scale = result['gini_importance'][f'Is_Top_{VENDOR_TIERS[tier_label]}']
    p_scale = result['permutation_importance'][f'Is_Top_{VENDOR_TIERS[tier_label]}']
    print(f"{tier_label:<25} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} "
          f"{m['f1_score']:>6.3f} {m['roc_auc']:>6.3f}  "
          f"{cv['mean']:.3f}+/-{cv['std']:.3f}  {g_scale*100:>10.1f}%  {p_scale*100:>10.1f}%")

print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
