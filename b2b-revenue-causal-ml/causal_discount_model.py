"""
B2B Revenue Causal Inference Demo
---------------------------------
This script demonstrates how selection bias (defensive discounting) in B2B SaaS
corrupts traditional A/B testing and naive data analysis, and how causal ML
techniques (Multivariate Regression and Propensity Score Matching) recover the
true causal impact of contract discounts on Net Revenue Retention (NRR).

This version is optimized for robustness using only pandas, numpy, and scikit-learn.
It avoids statsmodels and gracefully handles missing visualization packages.

Author: Antigravity AI
Date: May 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_crm_data(n_deals=1000):
    """
    Generates a realistic B2B CRM dataset with defensive discounting selection bias.
    
    Confounders:
      - health_score: 1.0 to 10.0 (measure of customer product adoption/engagement)
      - competitor_presence: 0 or 1 (whether a competitor is bidding on the renewal)
      
    Treatment:
      - discount: 0 or 1 (whether a 15% discount was offered)
      - Defensive Discounting: reps discount deals with poor health or competitor threats.
      
    Outcome:
      - nrr: Net Revenue Retention percentage after Year 1.
      - True Causal Effect of discount is exactly +5.0% NRR.
    """
    # 1. Generate Confounders
    health_score = np.random.normal(loc=6.0, scale=1.8, size=n_deals)
    health_score = np.clip(health_score, 1.0, 10.0) # clip to valid range
    
    competitor_presence = np.random.binomial(n=1, p=0.3, size=n_deals)
    
    # 2. Propensity Score: Probability of getting a discount (reps discount defensive deals)
    # Logit link: higher risk (low health) and competitor presence -> higher discount probability
    logit_p = 1.2 - 0.4 * health_score + 1.5 * competitor_presence
    propensity_score = 1 / (1 + np.exp(-logit_p))
    
    # Assign Treatment (Discount) based on Propensity Score
    discount = np.random.binomial(n=1, p=propensity_score)
    
    # 3. Generate Outcome (NRR)
    # True Treatment Effect = +5.0% NRR
    # Confounders also affect NRR directly: high health = high NRR, competitor = low NRR
    true_treatment_effect = 5.0
    noise = np.random.normal(loc=0.0, scale=2.5, size=n_deals)
    
    nrr = (
        82.0 + 
        true_treatment_effect * discount + 
        3.0 * health_score - 
        4.5 * competitor_presence + 
        noise
    )
    
    df = pd.DataFrame({
        'deal_id': [f"DEAL_{i:04d}" for i in range(n_deals)],
        'health_score': health_score,
        'competitor_presence': competitor_presence,
        'propensity_score_true': propensity_score,
        'discount': discount,
        'nrr': nrr
    })
    
    return df

def perform_naive_analysis(df):
    """Simple comparison of means and naive regression."""
    print("=== NAIVE ANALYSIS ===")
    
    # Difference in Means
    mean_treated = df[df['discount'] == 1]['nrr'].mean()
    mean_control = df[df['discount'] == 0]['nrr'].mean()
    naive_diff = mean_treated - mean_control
    
    print(f"Mean NRR (Discounted / Treated):  {mean_treated:.2f}%")
    print(f"Mean NRR (No Discount / Control): {mean_control:.2f}%")
    print(f"Naive Difference in Means:        {naive_diff:.2f}% (True Effect: +5.00%)")
    
    # Naive Regression
    lr = LinearRegression()
    lr.fit(df[['discount']], df['nrr'])
    estimated_effect = lr.coef_[0]
    
    print(f"\nNaive Regression (NRR ~ Discount):")
    print(f"Estimated Effect: {estimated_effect:.3f}% (True Effect: +5.000%)")
    print("-" * 50)
    return naive_diff

def perform_regression_adjustment(df):
    """Multivariate regression controlling for confounders."""
    print("\n=== MULTIVARIATE REGRESSION ADJUSTMENT ===")
    
    X = df[['discount', 'health_score', 'competitor_presence']]
    y = df['nrr']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    estimated_effect = lr.coef_[0]
    
    print("Multivariate Regression (NRR ~ Discount + Health + Competitor):")
    print(f"Estimated Causal Effect: {estimated_effect:.3f}% (True Effect: +5.000%)")
    print(f"Control Coefficients: Health Score = {lr.coef_[1]:.3f}, Competitor Presence = {lr.coef_[2]:.3f}")
    print("-" * 50)
    return estimated_effect

def perform_propensity_score_matching(df):
    """
    Estimates propensity scores and performs 1:1 nearest neighbor matching
    to estimate the Average Treatment Effect on the Treated (ATT).
    """
    print("\n=== PROPENSITY SCORE MATCHING (PSM) ===")
    
    # 1. Estimate Propensity Scores using Logistic Regression
    X_prop = df[['health_score', 'competitor_presence']]
    y_prop = df['discount']
    
    lr = LogisticRegression(penalty=None)
    lr.fit(X_prop, y_prop)
    df['estimated_propensity'] = lr.predict_proba(X_prop)[:, 1]
    
    # 2. Separate Treated (Discounted) and Control (Undiscounted)
    treated = df[df['discount'] == 1].copy()
    control = df[df['discount'] == 0].copy()
    
    # 3. Match 1:1 using Nearest Neighbors on the Logit of Propensity Score
    # logit(p) = log(p / (1-p))
    def logit(p):
        # Clip to prevent division by zero or log of zero
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    
    treated['logit_propensity'] = logit(treated['estimated_propensity'])
    control['logit_propensity'] = logit(control['estimated_propensity'])
    
    # Fit Nearest Neighbors on Control pool
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control['logit_propensity'].values.reshape(-1, 1))
    
    # Find closest match in Control for each Treated unit
    distances, indices = nn.kneighbors(treated['logit_propensity'].values.reshape(-1, 1))
    
    # Construct matched control dataframe
    matched_control = control.iloc[indices.flatten()].copy()
    
    # 4. Compare matched groups
    mean_treated_nrr = treated['nrr'].mean()
    mean_matched_control_nrr = matched_control['nrr'].mean()
    psm_att = mean_treated_nrr - mean_matched_control_nrr
    
    print(f"Matched Cohort Size: {len(treated)} pairs")
    print(f"Treated Mean NRR:         {mean_treated_nrr:.2f}%")
    print(f"Matched Control Mean NRR: {mean_matched_control_nrr:.2f}%")
    print(f"Estimated PSM ATT Effect: {psm_att:.3f}% (True Effect: +5.00%)")
    
    # 5. Plot Propensity Score Distribution before and after matching (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        plot_propensity_distribution(df, treated, matched_control)
    except ImportError:
        print("\n[Note] matplotlib is not installed. Skipping plot generation.")
    
    print("-" * 50)
    return psm_att

def plot_propensity_distribution(df_orig, treated, matched_control):
    """Helper to plot propensity scores to verify covariate balance."""
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Before Matching
        axes[0].hist(df_orig[df_orig['discount'] == 1]['estimated_propensity'], bins=20, alpha=0.6, label='Treated (Discounted)', color='#e74c3c')
        axes[0].hist(df_orig[df_orig['discount'] == 0]['estimated_propensity'], bins=20, alpha=0.6, label='Control (No Discount)', color='#3498db')
        axes[0].set_title('Propensity Score Distribution (Before Matching)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Estimated Propensity Score')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # After Matching
        axes[1].hist(treated['estimated_propensity'], bins=20, alpha=0.6, label='Treated (Discounted)', color='#e74c3c')
        axes[1].hist(matched_control['estimated_propensity'], bins=20, alpha=0.6, label='Matched Control', color='#2ecc71')
        axes[1].set_title('Propensity Score Distribution (After Matching)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Estimated Propensity Score')
        axes[1].set_ylabel('Count')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot to current directory
        output_path = os.path.join(os.path.dirname(__file__), 'propensity_score_balance.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"\n[Visualized Balance] Propensity score balance plot saved to:\n  {output_path}")
    except Exception as e:
        print(f"\n[Warning] Could not save propensity score plot: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("B2B REVENUE CAUSAL INFERENCE SIMULATION")
    print("=" * 60)
    
    # 1. Generate realistic data
    data = generate_synthetic_crm_data(n_deals=1000)
    print(f"Generated B2B CRM dataset with {len(data)} deals.")
    print("Variables: 'deal_id', 'health_score', 'competitor_presence', 'discount', 'nrr'\n")
    print("Sample Data (First 5 Rows):")
    print(data.head().to_string(index=False))
    print("=" * 60)
    
    # 2. Naive Analysis
    naive_diff = perform_naive_analysis(data)
    
    # 3. Regression Adjustment
    reg_effect = perform_regression_adjustment(data)
    
    # 4. Propensity Score Matching
    psm_effect = perform_propensity_score_matching(data)
    
    print("\nSUMMARY OF RESULTS:")
    print(f"  True Underlying Causal Effect:   +5.000%")
    print(f"  Naive Difference in Means:       {naive_diff:+.3f}%  ❌ (Severe selection bias!)")
    print(f"  Multivariate Regression Estimate: {reg_effect:+.3f}%  ✅ (Adjusted for confounders)")
    print(f"  Propensity Score Matching (ATT):  {psm_effect:+.3f}%  ✅ (Matched cohort comparison)")
    print("=" * 60)
