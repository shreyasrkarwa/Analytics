"""
Publication-Quality Figures
============================
Generates all figures for the paper:
  "Dynamic Churn Prediction in B2B SaaS: A Time-Varying Covariate
   Survival Analysis Framework with Renewal Cliff Quantification"

Figures produced:
  Figure 1: Global KM survival curve with confidence bands
  Figure 2: KM survival curves by customer segment (SMB / Mid-Market / Enterprise)
  Figure 3: Hazard ratio forest plot (stratified Cox model coefficients)
  Figure 4: Time-dependent AUC comparison (all models)
  Figure 5: Renewal cliff hazard visualization (piecewise hazard by month)
  Figure 6: Landmarking C-Index by landmark time
  Figure 7: IBM Telco KM curves by contract type (validation)

All figures saved at 300 DPI for publication submission.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------------

COLORS = {
    'smb': '#e74c3c',
    'midmarket': '#f39c12',
    'enterprise': '#27ae60',
    'primary': '#2c3e50',
    'secondary': '#7f8c8d',
    'accent': '#3498db',
    'green': '#27ae60',
    'orange': '#e67e22',
}

rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# FIGURE 1: GLOBAL KAPLAN-MEIER CURVE
# ---------------------------------------------------------------------------

def plot_global_km(static_df: pd.DataFrame, save_path: str = 'fig1_global_km.png') -> None:
    """Figure 1: Global KM survival curve."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    kmf = KaplanMeierFitter()
    kmf.fit(
        static_df['time_to_event'],
        static_df['event_observed'],
        label='All B2B SaaS Accounts'
    )
    kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS['primary'],
                               ci_alpha=0.15, linewidth=2)

    # Annotate median survival
    try:
        median = kmf.median_survival_time_
        if not np.isinf(median) and not np.isnan(median):
            ax.axvline(median, color=COLORS['accent'], linestyle='--', alpha=0.7, linewidth=1.5)
            ax.text(median + 0.3, 0.55, f'Median = {median:.0f}m',
                    color=COLORS['accent'], fontsize=9)
    except Exception:
        pass

    ax.set_title('Figure 1: Global Kaplan-Meier Survival Curve\n'
                 'B2B SaaS Account Retention (N = {:,})'.format(len(static_df)), fontsize=11)
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Probability of Active Subscription')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, static_df['time_to_event'].max() + 1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 2: SEGMENT-STRATIFIED KM CURVES
# ---------------------------------------------------------------------------

def plot_segment_km(static_df: pd.DataFrame, save_path: str = 'fig2_segment_km.png') -> None:
    """Figure 2: KM survival curves by customer segment."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

    segment_configs = [
        ('SMB', COLORS['smb'], axes[0]),
        ('Mid-Market', COLORS['midmarket'], axes[1]),
        ('Enterprise', COLORS['enterprise'], axes[2]),
    ]

    for segment, color, ax in segment_configs:
        sub = static_df[static_df['account_segment'] == segment]
        kmf = KaplanMeierFitter()
        kmf.fit(sub['time_to_event'], sub['event_observed'], label=segment)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color,
                                   ci_alpha=0.2, linewidth=2, legend=False)

        ax.set_title(f'{segment}\n(n={len(sub):,})', fontsize=11)
        ax.set_xlabel('Contract Months')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, static_df['time_to_event'].max() + 1])

        # Mark renewal boundaries
        contract_mode = sub['contract_length_months'].mode()[0]
        for renewal_m in range(contract_mode, int(static_df['time_to_event'].max()), contract_mode):
            ax.axvline(renewal_m, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    axes[0].set_ylabel('Probability of Active Subscription')

    fig.suptitle('Figure 2: Segment-Stratified Survival Curves\n'
                 'Vertical dotted lines indicate contract renewal boundaries',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 3: HAZARD RATIO FOREST PLOT
# ---------------------------------------------------------------------------

def plot_hr_forest(hr_df: pd.DataFrame, save_path: str = 'fig3_forest_plot.png') -> None:
    """
    Figure 3: Forest plot of hazard ratios from the stratified Cox model.
    Each predictor shown with its HR point estimate and 95% CI.
    """
    df = hr_df.reset_index().copy()
    df.columns = [c.strip() for c in df.columns]

    # Clean up variable names for display
    def clean_name(n):
        replacements = {
            'monthly_active_users_pct': 'MAU %',
            'feature_adoption_score': 'Feature Adoption',
            'support_tickets_created': 'Support Tickets',
            'csat_score': 'CSAT Score',
            'executive_sponsor_turnover': 'Exec. Sponsor Turnover',
            'overdue_invoices': 'Overdue Invoices',
            'upsell_occurred': 'Upsell Event',
            'has_channel_partner': 'Channel Partner',
            'initial_arr': 'Initial ARR',
            'onboarding_duration_days': 'Onboarding Duration',
            'contract_length_months': 'Contract Length',
        }
        for k, v in replacements.items():
            if k in n:
                return v
        # Handle dummies
        n = n.replace('industry_', 'Industry: ').replace('account_region_', 'Region: ')
        return n

    df['label'] = df.iloc[:, 0].apply(clean_name)

    # Sort by HR
    df = df.sort_values('HR', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35 + 1)))

    for i, row in df.iterrows():
        color = COLORS['smb'] if row['HR'] > 1 else COLORS['green']
        sig = row.get('p_value', 1.0) < 0.05
        marker_size = 8 if sig else 5
        alpha = 1.0 if sig else 0.6

        ax.plot(row['HR'], i, 'o', color=color, markersize=marker_size, alpha=alpha)
        ax.plot([row['HR_lower'], row['HR_upper']], [i, i],
                color=color, linewidth=1.5, alpha=alpha)

    ax.axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'], fontsize=9)
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.set_title('Figure 3: Hazard Ratios — Stratified Cox Model\n'
                 'Filled markers: p < 0.05 | HR > 1 = increased churn risk', fontsize=11)

    # Legend
    patches = [
        mpatches.Patch(color=COLORS['smb'], label='Increases churn risk (HR > 1)'),
        mpatches.Patch(color=COLORS['green'], label='Reduces churn risk (HR < 1)'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 4: RENEWAL CLIFF HAZARD VISUALIZATION
# ---------------------------------------------------------------------------

def plot_renewal_cliff(static_df: pd.DataFrame, save_path: str = 'fig4_renewal_cliff.png') -> None:
    """
    Figure 4: Monthly churn rate showing the renewal cliff effect.
    Bar chart of observed churn rate per month, highlighting renewal months.
    """
    df = static_df[static_df['event_observed'] == 1].copy()
    churn_by_month = df.groupby('time_to_event').size().reset_index(name='churn_count')
    total_at_risk = static_df.groupby('time_to_event').size().reset_index(name='at_risk')

    # Merge to get churn RATE per month
    merged = churn_by_month.merge(total_at_risk, on='time_to_event', how='right')
    merged['churn_rate'] = (merged['churn_count'].fillna(0) / merged['at_risk'] * 100)
    merged = merged[merged['time_to_event'] <= 36]

    # Identify renewal months (any multiple of 12, 24, or 36)
    renewal_months = set()
    for cl in static_df['contract_length_months'].unique():
        for m in range(cl, 37, cl):
            renewal_months.add(m)

    colors_bar = [COLORS['smb'] if m in renewal_months else COLORS['primary']
                  for m in merged['time_to_event']]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(merged['time_to_event'], merged['churn_rate'],
                  color=colors_bar, alpha=0.85, width=0.8)

    # Annotate key renewal months
    for m in [12, 24, 36]:
        row = merged[merged['time_to_event'] == m]
        if not row.empty:
            rate = row['churn_rate'].values[0]
            ax.annotate(f'Month {m}\n({rate:.1f}%)',
                        xy=(m, rate),
                        xytext=(m + 0.3, rate + 0.5),
                        fontsize=8, color=COLORS['smb'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['smb'], lw=1.2))

    # Legend
    patches = [
        mpatches.Patch(color=COLORS['smb'], label='Renewal month (contract boundary)'),
        mpatches.Patch(color=COLORS['primary'], label='Non-renewal month'),
    ]
    ax.legend(handles=patches, fontsize=9)

    ax.set_xlabel('Contract Month')
    ax.set_ylabel('Observed Churn Rate (%)')
    ax.set_title('Figure 4: Renewal Cliff Effect — Monthly Churn Rate\n'
                 'Churn risk spikes dramatically at contract renewal boundaries', fontsize=11)
    ax.set_xlim([0.5, 36.5])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 5: LANDMARK C-INDEX BY TIMEPOINT
# ---------------------------------------------------------------------------

def plot_landmark_c_indices(
    landmark_results: dict,
    static_c_index: float,
    save_path: str = 'fig5_landmark_c_index.png'
) -> None:
    """
    Figure 5: C-Index at each landmark time vs. static Cox baseline.
    Shows how prediction improves with more recent covariate information.
    """
    if not landmark_results:
        print("  No landmark results to plot.")
        return

    times = sorted(landmark_results.keys())
    c_indices = [landmark_results[t]['c_index'] for t in times]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(times, c_indices, 'o-', color=COLORS['accent'],
            linewidth=2, markersize=8, label='Landmark C-Index', zorder=3)
    ax.axhline(static_c_index, color=COLORS['secondary'], linestyle='--',
               linewidth=1.5, label=f'Static Cox Baseline (C={static_c_index:.3f})', zorder=2)

    for t, c in zip(times, c_indices):
        ax.annotate(f'{c:.3f}', (t, c), textcoords='offset points',
                    xytext=(5, 8), fontsize=9, color=COLORS['accent'])

    ax.set_xlabel('Landmark Time (Months)')
    ax.set_ylabel("Concordance Index (C-Index)")
    ax.set_title('Figure 5: Landmark Model C-Index by Prediction Horizon\n'
                 'Later landmarks use richer real-time information', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim([0.5, 1.0])
    ax.set_xticks(times)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# FIGURE 6: MODEL COMPARISON BAR CHART
# ---------------------------------------------------------------------------

def plot_model_comparison(
    model_results: dict,
    save_path: str = 'fig6_model_comparison.png'
) -> None:
    """
    Figure 6: Bar chart comparing C-Index across all models.

    model_results: dict mapping model_name -> c_index
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    models = list(model_results.keys())
    scores = list(model_results.values())
    bar_colors = [COLORS['primary']] * len(models)

    # Highlight best model
    best_idx = scores.index(max(scores))
    bar_colors[best_idx] = COLORS['green']

    bars = ax.barh(models, scores, color=bar_colors, alpha=0.85, height=0.5)

    # Random baseline
    ax.axvline(0.5, color=COLORS['smb'], linestyle='--',
               linewidth=1.2, label='Random (C=0.5)', alpha=0.8)

    for bar, score in zip(bars, scores):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}', va='center', fontsize=9)

    ax.set_xlabel('Concordance Index (C-Index)')
    ax.set_title('Figure 6: Model Performance Comparison\n'
                 'All models evaluated on held-out test set (20%)', fontsize=11)
    ax.set_xlim([0.4, min(1.0, max(scores) + 0.08)])
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI: Generate all available figures from data alone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import generate_synthetic_b2b_data

    print("Generating data...")
    static, telemetry = generate_synthetic_b2b_data(n_accounts=5000)

    print("\nGenerating figures...")
    plot_global_km(static)
    plot_segment_km(static)
    plot_renewal_cliff(static)

    # Placeholder model results for standalone run
    plot_model_comparison({
        'Logistic Regression (Baseline)': 0.68,
        'Random Forest (Baseline)': 0.71,
        'Cox PH (Static)': 0.72,
        'Stratified Cox (Proposed)': 0.74,
        'Random Survival Forest': 0.75,
        'Time-Varying Cox (Proposed)': 0.77,
    })

    print("\nAll figures saved.")
