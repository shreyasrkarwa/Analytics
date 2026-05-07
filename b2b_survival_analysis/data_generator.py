"""
B2B SaaS Survival Analysis — Synthetic Data Generator
=======================================================
Generates two complementary datasets:
  1. Static dataset (firmographics + outcome) for Cox PH and RSF models.
  2. Longitudinal counting-process dataset (start/stop/event per month per account)
     for time-varying covariate Cox models and landmarking.

Novel features vs. baseline:
  - Seasonal churn patterns (Q4 renewal spike, Q1 slowdown effect)
  - Upsell / expansion events that reset churn risk (product stickiness)
  - Proper start-stop interval format for lifelines.CoxTimeVaryingFitter
  - Cohort size increased to 5,000 accounts for statistical power
"""

import uuid
import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SEGMENTS = ['SMB', 'Mid-Market', 'Enterprise']
REGIONS = ['AMER', 'APAC', 'EMEA']
INDUSTRIES = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']

SEGMENT_CONFIG = {
    'SMB': dict(
        arr_mean=15_000, arr_std=3_000,
        contract_choices=[12], contract_probs=[1.0],
        onboarding_mean=15, onboarding_std=5,
        base_churn_prob=0.04
    ),
    'Mid-Market': dict(
        arr_mean=75_000, arr_std=15_000,
        contract_choices=[12, 24], contract_probs=[0.7, 0.3],
        onboarding_mean=40, onboarding_std=10,
        base_churn_prob=0.02
    ),
    'Enterprise': dict(
        arr_mean=250_000, arr_std=50_000,
        contract_choices=[12, 24, 36], contract_probs=[0.2, 0.5, 0.3],
        onboarding_mean=90, onboarding_std=20,
        base_churn_prob=0.01
    ),
}

# Months that correspond to Q4 (fiscal year end pressure)
Q4_MONTHS = {10, 11, 12}
# Months that correspond to Q1 budget realignment
Q1_MONTHS = {1, 2, 3}


# ---------------------------------------------------------------------------
# MAIN GENERATOR
# ---------------------------------------------------------------------------

def generate_synthetic_b2b_data(
    n_accounts: int = 5000,
    max_months: int = 36,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic B2B SaaS account data.

    Returns
    -------
    static_df : DataFrame
        One row per account. Firmographics + event/time outcome.
    telemetry_df : DataFrame
        One row per (account, month). Time-varying engagement metrics.
        Does NOT include start/stop format (use generate_longitudinal_format for that).
    """
    np.random.seed(random_seed)

    static_records = []
    telemetry_records = []

    for account_idx in range(n_accounts):
        account_id = str(uuid.uuid4())

        # ── Static Firmographic Generation ──────────────────────────────────
        segment = np.random.choice(SEGMENTS, p=[0.5, 0.3, 0.2])
        region = np.random.choice(REGIONS, p=[0.6, 0.15, 0.25])
        industry = np.random.choice(INDUSTRIES)
        has_channel_partner = np.random.binomial(1, 0.25)

        cfg = SEGMENT_CONFIG[segment]
        arr = np.random.normal(cfg['arr_mean'], cfg['arr_std'])
        contract_length = int(np.random.choice(cfg['contract_choices'], p=cfg['contract_probs']))
        onboarding_days = max(1, np.random.normal(cfg['onboarding_mean'], cfg['onboarding_std']))
        initial_arr = max(5_000, arr)
        base_churn_prob = cfg['base_churn_prob']

        # Channel partner lowers risk
        if has_channel_partner:
            base_churn_prob *= 0.8

        # Assign a random cohort start month (1–12) for seasonal effects
        cohort_start_month = np.random.randint(1, 13)

        static_record = {
            'account_id': account_id,
            'industry': industry,
            'account_segment': segment,
            'account_region': region,
            'has_channel_partner': has_channel_partner,
            'initial_arr': round(initial_arr, 2),
            'contract_length_months': contract_length,
            'onboarding_duration_days': round(onboarding_days),
            'cohort_start_month': cohort_start_month,
        }

        # ── Telemetry Time-Varying Simulation ───────────────────────────────
        current_mau = np.random.normal(0.85, 0.1)   # start healthy
        has_churned = False
        months_survived = 0
        had_upsell = False

        for month in range(1, max_months + 1):
            if has_churned:
                break

            months_survived = month
            calendar_month = ((cohort_start_month + month - 2) % 12) + 1  # 1-12

            # Drift engagement metrics
            current_mau = max(0.1, min(1.0, current_mau + np.random.normal(0, 0.05)))
            feature_adoption = max(0, min(100, current_mau * 100 + np.random.normal(0, 10)))
            support_tickets = max(0, int(np.random.normal((1.0 - current_mau) * 5, 2)))
            csat_score = min(5, max(1, np.random.normal(current_mau * 5, 0.5)))
            exec_sponsor_turnover = np.random.binomial(1, 0.02)   # 2% per month
            overdue_invoices = np.random.binomial(1, 0.01 + (1 - current_mau) * 0.05)

            # Upsell event: small probability each month, MAU > 0.75 makes it more likely
            # Upsell resets base churn risk (strong retention signal)
            upsell_occurred = False
            if not had_upsell and np.random.random() < (0.015 if current_mau > 0.75 else 0.005):
                upsell_occurred = True
                had_upsell = True
                current_mau = min(1.0, current_mau + 0.1)  # usage boost post-expansion

            telemetry_records.append({
                'account_id': account_id,
                'time_period': month,
                'calendar_month': calendar_month,
                'monthly_active_users_pct': round(current_mau, 3),
                'feature_adoption_score': round(feature_adoption, 1),
                'support_tickets_created': support_tickets,
                'csat_score': round(csat_score, 2),
                'executive_sponsor_turnover': exec_sponsor_turnover,
                'overdue_invoices': overdue_invoices,
                'upsell_occurred': int(upsell_occurred),
            })

            # ── Dynamic Churn Risk Calculation ──────────────────────────────
            monthly_risk = base_churn_prob

            # Telemetry-driven risk adjustments
            if exec_sponsor_turnover:
                monthly_risk += 0.05
            if overdue_invoices:
                monthly_risk += 0.03
            if current_mau < 0.4:
                monthly_risk += 0.04

            # Seasonal effects
            if calendar_month in Q4_MONTHS:
                monthly_risk *= 1.2   # Budget scrutiny at fiscal year end
            elif calendar_month in Q1_MONTHS:
                monthly_risk *= 0.9   # Budget just refreshed, lower churn

            # Upsell protective effect (sticky after expansion)
            if had_upsell:
                monthly_risk *= 0.6

            # ── Renewal Cliff: 3× spike at contract boundary ────────────────
            if month % contract_length == 0:
                monthly_risk *= 3.0

            if np.random.random() < monthly_risk:
                has_churned = True

        # Record survival outcome
        static_record['time_to_event'] = months_survived
        static_record['event_observed'] = 1 if has_churned else 0
        static_records.append(static_record)

    static_df = pd.DataFrame(static_records)
    telemetry_df = pd.DataFrame(telemetry_records)

    return static_df, telemetry_df


def generate_longitudinal_format(
    static_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts the static + telemetry data into counting-process (start-stop) format
    required by lifelines.CoxTimeVaryingFitter.

    Each row represents one time interval [t_start, t_stop) for one account.
    The 'event' column is 1 only in the interval where the account churns.

    Parameters
    ----------
    static_df : DataFrame from generate_synthetic_b2b_data()
    telemetry_df : DataFrame from generate_synthetic_b2b_data()

    Returns
    -------
    long_df : DataFrame with columns:
        account_id, t_start, t_stop, event,
        <all time-varying covariates>,
        <all static covariates (carried forward)>
    """
    outcome_map = (
        static_df[['account_id', 'time_to_event', 'event_observed']]
        .set_index('account_id')
    )

    static_features = static_df.set_index('account_id')[
        ['industry', 'account_segment', 'account_region',
         'has_channel_partner', 'initial_arr', 'contract_length_months',
         'onboarding_duration_days']
    ]

    records = []
    for account_id, grp in telemetry_df.groupby('account_id'):
        grp = grp.sort_values('time_period').reset_index(drop=True)
        t_event = outcome_map.loc[account_id, 'time_to_event']
        churned = outcome_map.loc[account_id, 'event_observed']
        static_row = static_features.loc[account_id]

        for _, row in grp.iterrows():
            t = int(row['time_period'])
            t_start = t - 1
            t_stop = t
            # event=1 only in the interval the account actually churned
            event = int(churned == 1 and t == t_event)

            rec = {
                'account_id': account_id,
                't_start': t_start,
                't_stop': t_stop,
                'event': event,
                # Time-varying
                'monthly_active_users_pct': row['monthly_active_users_pct'],
                'feature_adoption_score': row['feature_adoption_score'],
                'support_tickets_created': row['support_tickets_created'],
                'csat_score': row['csat_score'],
                'executive_sponsor_turnover': row['executive_sponsor_turnover'],
                'overdue_invoices': row['overdue_invoices'],
                'upsell_occurred': row['upsell_occurred'],
                # Static (carried forward)
                'industry': static_row['industry'],
                'account_segment': static_row['account_segment'],
                'account_region': static_row['account_region'],
                'has_channel_partner': static_row['has_channel_partner'],
                'initial_arr': static_row['initial_arr'],
                'contract_length_months': static_row['contract_length_months'],
                'onboarding_duration_days': static_row['onboarding_duration_days'],
            }
            records.append(rec)

            # Stop after churn event — no more intervals for churned accounts
            if event == 1:
                break

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic B2B dataset (5,000 accounts)...")
    static, telemetry = generate_synthetic_b2b_data(n_accounts=5000)
    long_df = generate_longitudinal_format(static, telemetry)

    print(f"  Accounts:          {len(static):,}")
    print(f"  Telemetry rows:    {len(telemetry):,}")
    print(f"  Longitudinal rows: {len(long_df):,}")
    print(f"  Churn rate:        {static['event_observed'].mean() * 100:.1f}%")
    print(f"  Upsell rate:       {telemetry.groupby('account_id')['upsell_occurred'].max().mean() * 100:.1f}%")

    static.to_csv("static_data.csv", index=False)
    telemetry.to_csv("telemetry_data.csv", index=False)
    long_df.to_csv("longitudinal_data.csv", index=False)
    print("\nSaved: static_data.csv, telemetry_data.csv, longitudinal_data.csv")
