"""
run_all_models.py — Master Orchestration Script
================================================
Runs the full B2B Survival Analysis pipeline end-to-end:

  1. Generate synthetic B2B SaaS dataset (5,000 accounts)
  2. Fit time-varying Cox model + landmarking
  3. Fit stratified Cox model (Schoenfeld PH test + AIC comparison)
  4. Run renewal cliff analysis (piecewise exponential)
  5. Validate on IBM Telco public dataset
  6. Generate all publication figures
  7. Save all performance numbers to results.json

Usage:
  python3 run_all_models.py

Output:
  results.json  — all C-Index, AIC, hazard ratios, etc. for use in paper
  figures/      — all publication-quality figures (PNG, 300 DPI)
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Ensure we can import modules from this directory
sys.path.insert(0, os.path.dirname(__file__))

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

results = {}

# ---------------------------------------------------------------------------
# STEP 1: Generate Synthetic Data
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 1: Generating Synthetic B2B SaaS Dataset (5,000 accounts)")
print("=" * 70)
t0 = time.time()

from data_generator import generate_synthetic_b2b_data, generate_longitudinal_format

static_df, telemetry_df = generate_synthetic_b2b_data(n_accounts=5000, random_seed=42)
long_df = generate_longitudinal_format(static_df, telemetry_df)

results['dataset'] = {
    'n_accounts': len(static_df),
    'n_telemetry_rows': len(telemetry_df),
    'n_longitudinal_rows': len(long_df),
    'churn_rate_pct': round(static_df['event_observed'].mean() * 100, 1),
    'median_tenure_months': float(static_df['time_to_event'].median()),
    'segment_distribution': static_df['account_segment'].value_counts().to_dict(),
    'upsell_rate_pct': round(
        telemetry_df.groupby('account_id')['upsell_occurred'].max().mean() * 100, 1
    ),
}

print(f"  ✓ Accounts: {len(static_df):,}")
print(f"  ✓ Longitudinal rows: {len(long_df):,}")
print(f"  ✓ Overall churn rate: {results['dataset']['churn_rate_pct']}%")
print(f"  ✓ Median tenure: {results['dataset']['median_tenure_months']} months")
print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")

# Per-segment churn rates
for seg in ['SMB', 'Mid-Market', 'Enterprise']:
    seg_df = static_df[static_df['account_segment'] == seg]
    rate = round(seg_df['event_observed'].mean() * 100, 1)
    results['dataset'][f'churn_rate_{seg.lower().replace("-", "_")}'] = rate
    print(f"  ✓ {seg} churn rate: {rate}%")


# ---------------------------------------------------------------------------
# STEP 2: Time-Varying Cox Model + Landmarking
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 2: Time-Varying Cox Model + Landmarking")
print("=" * 70)
t0 = time.time()

from time_varying_cox import fit_time_varying_cox, fit_landmark_models

tv_results = fit_time_varying_cox(long_df)
lm_results = fit_landmark_models(static_df, telemetry_df)

results['time_varying_cox'] = {
    'c_index': round(tv_results['c_index'], 4),
    'n_train_accounts': int(len(tv_results['train_ids'])),
    'n_test_accounts': int(len(tv_results['test_ids'])),
    'landmark_results': {
        str(t): {'c_index': round(v['c_index'], 4), 'n_accounts': int(v['n_accounts'])}
        for t, v in lm_results.items()
    }
}

print(f"\n  ✓ TV Cox C-Index: {results['time_varying_cox']['c_index']}")
for t, v in results['time_varying_cox']['landmark_results'].items():
    print(f"  ✓ Landmark t={t}m C-Index: {v['c_index']}  (n={v['n_accounts']:,})")
print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 3: Stratified Cox Model (PH Test + Model Comparison)
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 3: Segment-Stratified Cox Model")
print("=" * 70)
t0 = time.time()

from stratified_cox import test_ph_assumption, fit_unstratified_cox, fit_stratified_cox, compare_models

ph_test_df = test_ph_assumption(static_df)
unstrat_results = fit_unstratified_cox(static_df)
strat_results = fit_stratified_cox(static_df)
comparison = compare_models(unstrat_results, strat_results)

# Count PH violations
ph_violations = int(ph_test_df['PH Violated?'].sum()) if not ph_test_df.empty else 0
ph_total = len(ph_test_df)

results['stratified_cox'] = {
    'ph_violations': ph_violations,
    'ph_covariates_tested': ph_total,
    'unstratified_c_index': round(unstrat_results['c_index'], 4),
    'unstratified_aic': round(unstrat_results['aic'], 2),
    'stratified_c_index': round(strat_results['c_index'], 4),
    'stratified_aic': round(strat_results['aic'], 2),
    'aic_improvement': round(unstrat_results['aic'] - strat_results['aic'], 2),
    'c_index_improvement': round(strat_results['c_index'] - unstrat_results['c_index'], 4),
}

# Top hazard ratios from stratified model
if 'hr_df' in strat_results:
    hr_df = strat_results['hr_df'].copy()
    top_hrs = hr_df.nlargest(5, 'HR')[['HR', 'HR_lower', 'HR_upper', 'p_value']].to_dict('index')
    results['stratified_cox']['top_hazard_ratios'] = {
        str(k): {kk: round(float(vv), 4) for kk, vv in v.items()}
        for k, v in top_hrs.items()
    }

print(f"\n  ✓ PH violations: {ph_violations}/{ph_total} covariates")
print(f"  ✓ Unstratified Cox C-Index: {results['stratified_cox']['unstratified_c_index']}")
print(f"  ✓ Stratified Cox C-Index:   {results['stratified_cox']['stratified_c_index']}")
print(f"  ✓ AIC improvement (stratified): {results['stratified_cox']['aic_improvement']:.2f}")
print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 4: Renewal Cliff Analysis
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 4: Renewal Cliff Analysis (Piecewise Exponential)")
print("=" * 70)
t0 = time.time()

from renewal_cliff_analysis import (
    run_renewal_logrank_tests,
    fit_piecewise_exponential,
    analyze_renewal_cliff_by_segment,
)

log_rank_results = run_renewal_logrank_tests(static_df)
pe_results = fit_piecewise_exponential(static_df)
segment_analysis = analyze_renewal_cliff_by_segment(static_df)

# Extract key renewal cliff hazard ratio from piecewise exponential model
renewal_hr = pe_results.get('renewal_hr', None)

results['renewal_cliff'] = {
    'log_rank_significant_count': int((log_rank_results['Significant (p<0.05)'].sum())
                                       if not log_rank_results.empty else 0),
    'renewal_hr': round(float(renewal_hr), 3) if renewal_hr else None,
    'model_aic': round(float(pe_results['model'].AIC_partial_), 2) if pe_results.get('model') else None,
    'segment_renewal_hrs': {},
}

# Segment-level renewal cliffs from DataFrame
if not segment_analysis.empty:
    for _, row in segment_analysis.iterrows():
        seg = row['Segment']
        results['renewal_cliff']['segment_renewal_hrs'][seg] = {
            'renewal_hr': round(float(row['Renewal HR']), 3),
            'hr_lower': round(float(row['HR Lower 95%']), 3),
            'hr_upper': round(float(row['HR Upper 95%']), 3),
            'p_value': round(float(row['p-value']), 4),
        }

print(f"  ✓ Renewal cliff HR (overall): {results['renewal_cliff']['renewal_hr']}")
print(f"  ✓ Log-rank significant renewal months: {results['renewal_cliff']['log_rank_significant_count']}")
for seg, v in results['renewal_cliff']['segment_renewal_hrs'].items():
    print(f"    {seg}: HR={v['renewal_hr']:.2f} (95% CI: {v['hr_lower']:.2f}–{v['hr_upper']:.2f}, p={v['p_value']})")
print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 5: IBM Telco Validation
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 5: IBM Telco Public Dataset Validation")
print("=" * 70)
t0 = time.time()

from validation_telco import load_telco_data, preprocess_telco, fit_cox_on_telco, fit_rsf_on_telco

raw_telco = load_telco_data()
processed_telco = preprocess_telco(raw_telco)
telco_cox = fit_cox_on_telco(processed_telco)
telco_rsf = fit_rsf_on_telco(processed_telco)

results['telco_validation'] = {
    'n_customers': len(raw_telco),
    'churn_rate_pct': round((raw_telco['Churn'] == 'Yes').mean() * 100, 1),
    'cox_c_index': round(telco_cox['c_index'], 4),
    'rsf_c_index': round(telco_rsf['c_index'], 4),
    'cox_aic': round(telco_cox['aic'], 2),
}

print(f"\n  ✓ Telco customers: {results['telco_validation']['n_customers']:,}")
print(f"  ✓ Telco churn rate: {results['telco_validation']['churn_rate_pct']}%")
print(f"  ✓ Cox C-Index (Telco): {results['telco_validation']['cox_c_index']}")
print(f"  ✓ RSF C-Index (Telco): {results['telco_validation']['rsf_c_index']}")
print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 6: Generate Figures
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  STEP 6: Generating Publication Figures")
print("=" * 70)
t0 = time.time()

try:
    from figures import (
        plot_global_km,
        plot_segment_km,
        plot_hr_forest,
        plot_renewal_cliff,
        plot_landmark_c_indices,
        plot_model_comparison,
    )

    # Figure 1: Global KM
    plot_global_km(static_df, save_path=os.path.join(FIGURES_DIR, "fig1_global_km.png"))

    # Figure 2: Segment-stratified KM curves
    plot_segment_km(static_df, save_path=os.path.join(FIGURES_DIR, "fig2_segment_km.png"))
    print("  ✓ KM curves saved")

    # Figure 3: Forest plot of hazard ratios
    if 'hr_df' in strat_results:
        plot_hr_forest(
            strat_results['hr_df'],
            save_path=os.path.join(FIGURES_DIR, "fig3_forest_plot.png")
        )
        print("  ✓ Hazard ratio forest plot saved")

    # Figure 4: Renewal cliff visualization
    plot_renewal_cliff(static_df, save_path=os.path.join(FIGURES_DIR, "fig4_renewal_cliff.png"))
    print("  ✓ Renewal cliff figure saved")

    # Figure 5: Landmark C-Index by time
    plot_landmark_c_indices(
        lm_results,
        static_c_index=results['stratified_cox']['unstratified_c_index'],
        save_path=os.path.join(FIGURES_DIR, "fig5_landmark_c_index.png")
    )
    print("  ✓ Landmark C-Index figure saved")

    # Figure 6: Model comparison
    model_comparison_data = {
        'Unstratified Cox (Baseline)': results['stratified_cox']['unstratified_c_index'],
        'Stratified Cox (Proposed)': results['stratified_cox']['stratified_c_index'],
        'Time-Varying Cox (Proposed)': results['time_varying_cox']['c_index'],
        'IBM Telco Cox (Validation)': results['telco_validation']['cox_c_index'],
        'IBM Telco RSF (Validation)': results['telco_validation']['rsf_c_index'],
    }
    plot_model_comparison(
        model_comparison_data,
        save_path=os.path.join(FIGURES_DIR, "fig6_model_comparison.png")
    )
    print("  ✓ Model comparison figure saved")

except Exception as e:
    import traceback
    print(f"  ⚠ Figure generation partial failure: {e}")
    traceback.print_exc()
    print("  (Non-fatal — results still saved)")

print(f"  ✓ Elapsed: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# STEP 7: Save Results
# ---------------------------------------------------------------------------

results_path = "results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 70)
print("  COMPLETE — All Results Saved")
print("=" * 70)
print(f"\n  📄 Results: {results_path}")
print(f"  📊 Figures: {FIGURES_DIR}/")
print("\n  KEY METRICS SUMMARY:")
print(f"  {'Model':<35} {'C-Index'}")
print(f"  {'-'*45}")
print(f"  {'Unstratified Cox (static baseline)':<35} {results['stratified_cox']['unstratified_c_index']}")
print(f"  {'Stratified Cox (Contribution #2)':<35} {results['stratified_cox']['stratified_c_index']}")
print(f"  {'Time-Varying Cox (Contribution #1)':<35} {results['time_varying_cox']['c_index']}")
print(f"  {'IBM Telco Cox (External Validation)':<35} {results['telco_validation']['cox_c_index']}")
print(f"  {'IBM Telco RSF (External Validation)':<35} {results['telco_validation']['rsf_c_index']}")
print(f"\n  Churn rate (synthetic): {results['dataset']['churn_rate_pct']}%")
print(f"  Churn rate (IBM Telco): {results['telco_validation']['churn_rate_pct']}%")
print(f"  AIC improvement (stratification): {results['stratified_cox']['aic_improvement']:.1f}")
print(f"  PH violations: {results['stratified_cox']['ph_violations']}/{results['stratified_cox']['ph_covariates_tested']}")
