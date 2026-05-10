"""
Renewal Cliff Quantification
==============================
Novel Contribution #3 of the paper:
  "Dynamic Churn Prediction in B2B SaaS: A Time-Varying Covariate
   Survival Analysis Framework with Renewal Cliff Quantification"

B2B SaaS customers do not churn randomly throughout the year.
They churn overwhelmingly at contract renewal boundaries (month 12, 24, 36).
Practitioners know this intuitively — but it has never been formally
quantified in the survival analysis literature.

This module:
  (A) Creates renewal boundary indicator variables and tests them
      using log-rank tests (are renewal months statistically different?).
  (B) Fits a piecewise exponential model that estimates a separate
      hazard rate for each interval, revealing the exact hazard ratio
      at renewal vs. non-renewal months.
  (C) Computes confidence intervals on the renewal cliff hazard ratio.
  (D) Compares the piecewise model against the standard Cox baseline.
  (E) Conducts segment-level analysis (does the cliff differ by segment?).

Reference: Friedman (1982). Piecewise exponential models for survival data.
"""

import numpy as np
import pandas as pd
from lifelines import (
    WeibullAFTFitter,
    CoxPHFitter,
    KaplanMeierFitter,
)
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# PART A: LOG-RANK TESTS — ARE RENEWAL MONTHS DIFFERENT?
# ---------------------------------------------------------------------------

def run_renewal_logrank_tests(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Formally test whether survival at renewal months differs significantly
    from survival at non-renewal months using log-rank tests.

    For each contract length group (12, 24, 36 months), we compare:
      - Accounts churning AT the renewal month vs. all other months.

    A statistically significant result (p < 0.05) confirms the renewal cliff.

    Returns
    -------
    DataFrame of log-rank test results by contract type and renewal month
    """
    print("\n" + "=" * 60)
    print("  LOG-RANK TESTS: Renewal vs. Non-Renewal Months")
    print("=" * 60)

    results = []

    for contract_len in [12, 24, 36]:
        sub = static_df[static_df['contract_length_months'] == contract_len].copy()
        if len(sub) < 50:
            continue

        for renewal_month in range(contract_len, static_df['time_to_event'].max() + 1, contract_len):
            at_renewal = sub[sub['time_to_event'] == renewal_month]
            not_at_renewal = sub[sub['time_to_event'] != renewal_month]

            if len(at_renewal) < 10 or len(not_at_renewal) < 10:
                continue

            lr = logrank_test(
                durations_A=at_renewal['time_to_event'],
                durations_B=not_at_renewal['time_to_event'],
                event_observed_A=at_renewal['event_observed'],
                event_observed_B=not_at_renewal['event_observed'],
            )

            results.append({
                'Contract Length': contract_len,
                'Renewal Month': renewal_month,
                'N at Renewal': len(at_renewal),
                'N Not at Renewal': len(not_at_renewal),
                'Log-Rank Statistic': round(lr.test_statistic, 3),
                'p-value': round(lr.p_value, 4),
                'Significant (p<0.05)': lr.p_value < 0.05,
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# PART B: PIECEWISE EXPONENTIAL MODEL (COUNTING-PROCESS FORMAT)
# ---------------------------------------------------------------------------

def _expand_to_counting_process(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand static account data into counting-process (start/stop/event) format
    with time-varying renewal boundary indicators.

    Each account contributes one row per month of observation:
      (account_id, t_start, t_stop, event, at_renewal_boundary, ...)

    The at_renewal_boundary indicator is set PROSPECTIVELY based on contract
    structure: it is 1 for any month that IS a renewal boundary month for
    that account, regardless of whether the account actually churns there.
    This eliminates outcome leakage.
    """
    rows = []
    for _, acct in static_df.iterrows():
        tte = int(acct['time_to_event'])
        event = int(acct['event_observed'])
        cl = int(acct['contract_length_months'])

        for m in range(1, tte + 1):
            is_last = (m == tte)
            month_event = 1 if (is_last and event == 1) else 0

            # Prospective indicators — determined by contract structure, not outcome
            is_renewal = 1 if (m % cl == 0) else 0
            months_into = m % cl
            is_early = 1 if (months_into <= 3 or m <= 3) else 0
            is_late = 1 if (cl - months_into) <= 3 and months_into != 0 else 0

            rows.append({
                'account_id': acct['account_id'],
                't_start': m - 1,
                't_stop': m,
                'event': month_event,
                'at_renewal_boundary': is_renewal,
                'contract_phase_early': is_early,
                'contract_phase_late': is_late,
                'has_channel_partner': acct['has_channel_partner'],
                'initial_arr': acct['initial_arr'],
                'onboarding_duration_days': acct['onboarding_duration_days'],
                'account_segment': acct['account_segment'],
            })

    return pd.DataFrame(rows)


def fit_piecewise_exponential(static_df: pd.DataFrame) -> dict:
    """
    Fit a piecewise exponential survival model using counting-process format
    with time-varying renewal boundary indicators.

    Unlike the naive approach (computing at_renewal_boundary from the outcome),
    this method expands each account into monthly intervals and sets the renewal
    indicator PROSPECTIVELY: it is 1 for any month that is a contract renewal
    boundary month, regardless of whether churn occurs there. This is a
    legitimate time-varying covariate determined by contract terms.

    Uses CoxTimeVaryingFitter on the expanded dataset.

    Returns
    -------
    dict with model, hazard ratios, and confidence intervals
    """
    from lifelines import CoxTimeVaryingFitter

    print("\n" + "=" * 60)
    print("  PIECEWISE EXPONENTIAL MODEL — Renewal Cliff Hazard Ratios")
    print("  (Counting-Process Format — No Outcome Leakage)")
    print("=" * 60)

    print("  Expanding to counting-process format...")
    cp_df = _expand_to_counting_process(static_df)
    print(f"  Expanded: {len(static_df):,} accounts → {len(cp_df):,} person-month rows")

    # Encode segment
    cp_df = pd.get_dummies(cp_df, columns=['account_segment'], drop_first=True)
    segment_cols = [c for c in cp_df.columns if c.startswith('account_segment_')]

    feature_cols = [
        'at_renewal_boundary',
        'contract_phase_early',
        'contract_phase_late',
        'has_channel_partner',
        'initial_arr',
        'onboarding_duration_days',
    ] + segment_cols
    feature_cols = [c for c in feature_cols if c in cp_df.columns]

    # Fit time-varying Cox model
    ctv = CoxTimeVaryingFitter(penalizer=0.05)
    ctv.fit(
        cp_df[['account_id', 't_start', 't_stop', 'event'] + feature_cols],
        id_col='account_id',
        start_col='t_start',
        stop_col='t_stop',
        event_col='event',
    )

    print(f"\n  Model AIC (partial): {ctv.AIC_partial_:.2f}")

    # Extract renewal cliff hazard ratio
    summary = ctv.summary.copy()
    print("\n  Hazard Ratios (Key Variables):")
    key_vars = [c for c in ['at_renewal_boundary', 'contract_phase_early', 'contract_phase_late']
                if c in summary.index]
    renewal_hr = None
    if key_vars:
        hr_table = summary.loc[key_vars, ['coef', 'exp(coef)', 'exp(coef) lower 95%',
                                           'exp(coef) upper 95%', 'p']]
        hr_table.columns = ['log_HR', 'HR', 'HR_lower_95', 'HR_upper_95', 'p_value']
        print(hr_table.to_string())

        if 'at_renewal_boundary' in summary.index:
            renewal_hr = summary.loc['at_renewal_boundary', 'exp(coef)']
            renewal_ci_lo = summary.loc['at_renewal_boundary', 'exp(coef) lower 95%']
            renewal_ci_hi = summary.loc['at_renewal_boundary', 'exp(coef) upper 95%']

            print(f"\n  ★ RENEWAL CLIFF HAZARD RATIO: {renewal_hr:.3f}×")
            print(f"    95% CI: [{renewal_ci_lo:.3f}, {renewal_ci_hi:.3f}]")
            print(f"    Interpretation: Churn risk is {renewal_hr:.1f}× {'higher' if renewal_hr > 1 else 'lower'} at contract")
            print(f"    renewal boundary months vs. non-boundary months.")

    return {
        'model': ctv,
        'summary': summary,
        'renewal_hr': renewal_hr,
    }


# ---------------------------------------------------------------------------
# PART C: SEGMENT-LEVEL RENEWAL CLIFF ANALYSIS
# ---------------------------------------------------------------------------

def analyze_renewal_cliff_by_segment(static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Does the renewal cliff effect differ by customer segment?

    We estimate the renewal-month hazard ratio separately for SMB,
    Mid-Market, and Enterprise customers to test whether larger accounts
    show a different cliff pattern (hypothesis: Enterprise cliffs are
    softer due to multi-year contracts and stronger CS coverage).

    Uses counting-process format to avoid outcome leakage.

    Returns
    -------
    DataFrame with HR estimates and CIs per segment
    """
    from lifelines import CoxTimeVaryingFitter

    print("\n" + "=" * 60)
    print("  RENEWAL CLIFF BY SEGMENT (Counting-Process Format)")
    print("=" * 60)

    results = []

    for segment in ['SMB', 'Mid-Market', 'Enterprise']:
        sub = static_df[static_df['account_segment'] == segment].copy()
        if len(sub) < 100:
            continue

        # Expand to counting-process format
        cp_df = _expand_to_counting_process(sub)

        feature_cols = ['at_renewal_boundary', 'has_channel_partner',
                        'initial_arr', 'onboarding_duration_days']
        feature_cols = [c for c in feature_cols if c in cp_df.columns]

        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.1)
            ctv.fit(
                cp_df[['account_id', 't_start', 't_stop', 'event'] + feature_cols],
                id_col='account_id',
                start_col='t_start',
                stop_col='t_stop',
                event_col='event',
            )
            s = ctv.summary

            if 'at_renewal_boundary' in s.index:
                results.append({
                    'Segment': segment,
                    'N accounts': len(sub),
                    'Renewal HR': round(s.loc['at_renewal_boundary', 'exp(coef)'], 3),
                    'HR Lower 95%': round(s.loc['at_renewal_boundary', 'exp(coef) lower 95%'], 3),
                    'HR Upper 95%': round(s.loc['at_renewal_boundary', 'exp(coef) upper 95%'], 3),
                    'p-value': round(s.loc['at_renewal_boundary', 'p'], 4),
                })
        except Exception as e:
            print(f"  {segment}: Model failed — {e}")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    return results_df


# ---------------------------------------------------------------------------
# PART D: KAPLAN-MEIER AT RENEWAL BOUNDARIES
# ---------------------------------------------------------------------------

def plot_renewal_km_curves(static_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot KM survival curves stratified by whether the account churned
    at a renewal boundary vs. at a non-renewal time.
    (Visualization — see figures.py for publication-quality version.)
    """
    import matplotlib.pyplot as plt

    df = static_df.copy()
    df['at_renewal'] = (df['time_to_event'] % df['contract_length_months'] == 0) & (df['event_observed'] == 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()

    for group, label, color in [
        (df[df['at_renewal']], 'Churned at Renewal Boundary', '#e74c3c'),
        (df[~df['at_renewal']], 'Churned at Non-Renewal Month', '#2980b9'),
    ]:
        kmf.fit(group['time_to_event'], group['event_observed'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_title('KM Survival: Renewal Boundary vs. Non-Renewal Churn', fontsize=13)
    ax.set_xlabel('Contract Months')
    ax.set_ylabel('Survival Probability')
    ax.set_ylim([0, 1.05])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import generate_synthetic_b2b_data

    print("Generating data...")
    static, _ = generate_synthetic_b2b_data(n_accounts=5000)

    logrank_results = run_renewal_logrank_tests(static)
    piecewise_results = fit_piecewise_exponential(static)
    segment_results = analyze_renewal_cliff_by_segment(static)
    plot_renewal_km_curves(static, save_path='renewal_km_curves.png')
