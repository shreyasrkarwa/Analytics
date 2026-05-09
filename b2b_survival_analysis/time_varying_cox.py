"""
Time-Varying Covariate Cox Model with Landmarking
===================================================
Novel Contribution #1 of the paper:
  "Dynamic Churn Prediction in B2B SaaS: A Time-Varying Covariate
   Survival Analysis Framework with Renewal Cliff Quantification"

Standard churn models use only static features recorded at signup.
This module implements:

  (A) Time-Varying Cox Model — updates risk scores each month using
      live engagement signals (MAU, CSAT, support tickets, exec turnover).
      Uses lifelines.CoxTimeVaryingFitter on counting-process data.

  (B) Landmarking — at each landmark time (3, 6, 12, 18 months), fit
      a separate Cox PH model using only accounts that survived to that
      point, conditioned on their current covariate values. This generates
      actionable current-state risk scores for CS teams.

Reference: van Houwelingen (2007). Dynamic prediction by landmarking.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lifelines import CoxTimeVaryingFitter, CoxPHFitter
from lifelines.utils import concordance_index
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

STATIC_CATEGORICALS = ['industry', 'account_segment', 'account_region']
STATIC_NUMERICS = ['has_channel_partner', 'initial_arr', 'contract_length_months',
                   'onboarding_duration_days']
TV_FEATURES = ['monthly_active_users_pct', 'feature_adoption_score',
               'support_tickets_created', 'csat_score',
               'executive_sponsor_turnover', 'overdue_invoices', 'upsell_occurred']


def _encode_categoricals(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One-hot encode categorical columns, dropping first to avoid collinearity."""
    return pd.get_dummies(df, columns=cols, drop_first=True)


def load_longitudinal_data(path: str = 'longitudinal_data.csv') -> pd.DataFrame:
    return pd.read_csv(path)


def load_static_data(path: str = 'static_data.csv') -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# PART A: TIME-VARYING COX MODEL
# ---------------------------------------------------------------------------

def fit_time_varying_cox(long_df: pd.DataFrame) -> dict:
    """
    Fit a time-varying covariate Cox model using counting-process format.

    Each row represents [t_start, t_stop) with covariates recorded at t_start.
    The 'event' column flags the interval where churn occurred.

    Parameters
    ----------
    long_df : counting-process DataFrame from data_generator.generate_longitudinal_format()

    Returns
    -------
    dict with fitted model, training metrics, and feature summary
    """
    print("\n" + "=" * 60)
    print("  TIME-VARYING COVARIATE COX MODEL")
    print("=" * 60)

    df = long_df.copy()
    df = _encode_categoricals(df, STATIC_CATEGORICALS)

    # Identify all feature columns
    feature_cols = (
        [c for c in df.columns if c.startswith('industry_') or
         c.startswith('account_segment_') or c.startswith('account_region_')]
        + STATIC_NUMERICS
        + TV_FEATURES
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Split accounts (not rows) to avoid data leakage
    account_ids = df['account_id'].unique()
    train_ids, test_ids = train_test_split(account_ids, test_size=0.2, random_state=42)

    train_df = df[df['account_id'].isin(train_ids)].copy()
    test_df = df[df['account_id'].isin(test_ids)].copy()

    # Fit model
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    ctv.fit(
        train_df[['account_id', 't_start', 't_stop', 'event'] + feature_cols],
        id_col='account_id',
        start_col='t_start',
        stop_col='t_stop',
        event_col='event',
    )

    # Compute concordance index on test set using partial hazards
    # (use last observation per test account as the feature vector)
    last_obs = (
        test_df.sort_values('t_stop')
        .groupby('account_id')
        .last()
        .reset_index()
    )

    # Outcome for concordance
    outcome = (
        test_df[test_df['event'] == 1][['account_id', 't_stop', 'event']]
        .drop_duplicates('account_id')
        .set_index('account_id')
    )
    # For censored accounts, use last observed time
    censored = last_obs[~last_obs['account_id'].isin(outcome.index)]
    for _, row in censored.iterrows():
        outcome.loc[row['account_id']] = [row['t_stop'], 0]

    # predict_partial_hazard returns a positionally-indexed Series;
    # use feature_df positionally aligned with last_obs
    feature_df = last_obs[feature_cols].copy()
    risk_scores = ctv.predict_partial_hazard(feature_df)  # Series with same positional index as last_obs

    aligned_ids = last_obs['account_id'].values
    times = np.array([outcome.loc[aid, 't_stop'] for aid in aligned_ids])
    events = np.array([outcome.loc[aid, 'event'] for aid in aligned_ids]).astype(bool)
    scores = risk_scores.values  # positionally aligned with last_obs / aligned_ids

    c_idx = concordance_index(times, -scores, events)

    print(f"  Test Concordance Index (Harrell's C): {c_idx:.4f}")
    print(f"  Training accounts: {len(train_ids):,} | Test accounts: {len(test_ids):,}")

    print("\n  Top 10 Predictors (by |coefficient|):")
    coef_df = ctv.summary[['coef', 'exp(coef)', 'p']].sort_values('coef', key=abs, ascending=False).head(10)
    print(coef_df.to_string())

    return {
        'model': ctv,
        'c_index': c_idx,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'feature_cols': feature_cols,
        'summary': ctv.summary,
    }


# ---------------------------------------------------------------------------
# PART B: LANDMARKING
# ---------------------------------------------------------------------------

LANDMARK_TIMES = [3, 6, 12, 18]  # months


def fit_landmark_models(static_df: pd.DataFrame, telemetry_df: pd.DataFrame) -> dict:
    """
    Fit separate Cox PH models at each landmark time.

    At landmark t_L:
      1. Include only accounts that survived to t_L (not yet churned).
      2. Use covariate values measured at t_L (current state, not signup).
      3. Fit a Cox model predicting time-to-churn *from* t_L onward.
      4. This gives CS teams a current-state risk score, not a stale prediction.

    Parameters
    ----------
    static_df : from generate_synthetic_b2b_data()
    telemetry_df : from generate_synthetic_b2b_data()

    Returns
    -------
    dict mapping landmark_time -> (fitted CoxPHFitter, c_index)
    """
    print("\n" + "=" * 60)
    print("  LANDMARKING — DYNAMIC RISK PREDICTION")
    print("=" * 60)

    results = {}

    for t_L in LANDMARK_TIMES:
        # Accounts still active at t_L
        active_accounts = static_df[static_df['time_to_event'] >= t_L]['account_id'].values

        if len(active_accounts) < 100:
            print(f"  Landmark t={t_L}: Insufficient accounts ({len(active_accounts)}), skipping.")
            continue

        # Get covariates at t_L (or last available observation before t_L)
        tele_at_lm = (
            telemetry_df[
                (telemetry_df['account_id'].isin(active_accounts)) &
                (telemetry_df['time_period'] <= t_L)
            ]
            .sort_values('time_period')
            .groupby('account_id')
            .last()
            .reset_index()
        )

        # Residual time: time remaining from t_L to event (or censoring)
        static_sub = static_df[static_df['account_id'].isin(active_accounts)].copy()
        static_sub['residual_time'] = static_sub['time_to_event'] - t_L
        static_sub['event_observed'] = static_sub['event_observed']

        # Merge static + telemetry at landmark
        lm_df = tele_at_lm.merge(
            static_sub[['account_id', 'residual_time', 'event_observed',
                         'initial_arr', 'contract_length_months',
                         'onboarding_duration_days', 'has_channel_partner',
                         'account_segment', 'account_region', 'industry']],
            on='account_id'
        )

        lm_df = _encode_categoricals(lm_df, STATIC_CATEGORICALS)

        feature_cols = (
            [c for c in lm_df.columns if c.startswith('industry_') or
             c.startswith('account_segment_') or c.startswith('account_region_')]
            + STATIC_NUMERICS
            + TV_FEATURES
        )
        feature_cols = [c for c in feature_cols if c in lm_df.columns]

        model_df = lm_df[feature_cols + ['residual_time', 'event_observed']].copy()
        model_df = model_df.dropna()

        # Train/test split at account level
        account_ids = lm_df['account_id'].values
        train_idx, test_idx = train_test_split(
            range(len(account_ids)), test_size=0.2, random_state=42
        )

        train_df = model_df.iloc[train_idx]
        test_df = model_df.iloc[test_idx]

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(train_df, duration_col='residual_time', event_col='event_observed')

            risk = cph.predict_partial_hazard(test_df[feature_cols])
            c_idx = concordance_index(
                test_df['residual_time'].values,
                -risk.values,
                test_df['event_observed'].values.astype(bool),
            )
        except Exception as e:
            print(f"  Landmark t={t_L}: Model failed — {e}")
            continue

        results[t_L] = {'model': cph, 'c_index': c_idx, 'n_accounts': len(active_accounts)}
        print(f"  Landmark t={t_L:2d}m | Active accounts: {len(active_accounts):,} "
              f"| C-Index: {c_idx:.4f}")

    return results


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_generator import generate_synthetic_b2b_data, generate_longitudinal_format

    print("Generating data...")
    static, telemetry = generate_synthetic_b2b_data(n_accounts=5000)
    long_df = generate_longitudinal_format(static, telemetry)

    print("\n--- Part A: Time-Varying Cox Model ---")
    tv_results = fit_time_varying_cox(long_df)

    print("\n--- Part B: Landmark Models ---")
    lm_results = fit_landmark_models(static, telemetry)

    print("\n\nSUMMARY")
    print(f"  TV Cox C-Index:  {tv_results['c_index']:.4f}")
    for t_L, res in lm_results.items():
        print(f"  Landmark t={t_L:2d}m C-Index: {res['c_index']:.4f}")
