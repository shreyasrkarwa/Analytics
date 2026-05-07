"""
Unified Revenue Platform — End-to-End Demo Pipeline
====================================================

Demonstrates the complete URP workflow using synthetic data:
    1. Generate organizational hierarchy (DAG)
    2. Generate synthetic opportunity data
    3. Tag fiscal quarters and compute coverage
    4. Cascade quota targets through the hierarchy
    5. Detect and reconcile manager commit bias
    6. Train and evaluate the ML revenue forecast model
    7. Submit structured weekly forecasts

Run:
    python demo_pipeline.py
"""

import sys
import numpy as np
import pandas as pd

from unified_revenue_platform.data_generator import (
    generate_org_hierarchy,
    generate_opportunities,
    generate_historical_commits,
    generate_forecast_features,
)
from unified_revenue_platform.dimension_engineering import (
    tag_fiscal_quarters,
    get_fiscal_quarter,
    compute_pipeline_coverage,
)
from unified_revenue_platform.quota_cascader import QuotaCascader
from unified_revenue_platform.bias_reconciler import (
    BiasReconciler,
    reconcile_forecast,
)
from unified_revenue_platform.revenue_forecaster import RevenueForecaster
from unified_revenue_platform.weekly_forecast import WeeklyForecastManager


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    """Run the full demo pipeline."""

    # ──────────────────────────────────────────────────────
    # Step 1: Generate Organizational Hierarchy
    # ──────────────────────────────────────────────────────
    print_section("Step 1: Organizational Hierarchy")

    dag, node_metadata = generate_org_hierarchy(n_ics=50)
    leaf_nodes = [n for n in dag.nodes if dag.out_degree(n) == 0]

    print(f"Total nodes:    {dag.number_of_nodes()}")
    print(f"Total edges:    {dag.number_of_edges()}")
    print(f"ICs (leaf):     {len(leaf_nodes)}")
    print(f"Managers:       {len(node_metadata[node_metadata['level'] == 'Manager'])}")
    print(f"Directors:      {len(node_metadata[node_metadata['level'] == 'Director'])}")
    print(f"VPs:            {len(node_metadata[node_metadata['level'] == 'VP'])}")
    print(f"Regions:        {node_metadata['region'].unique().tolist()}")

    # ──────────────────────────────────────────────────────
    # Step 2: Generate Opportunity Data
    # ──────────────────────────────────────────────────────
    print_section("Step 2: Synthetic Opportunity Data")

    opps_df = generate_opportunities(dag, node_metadata, n_quarters=8)
    print(f"Total opportunities:  {len(opps_df):,}")
    print(f"Closed Won:           {opps_df['IsWon'].sum():,}")
    print(f"Closed Lost:          {(opps_df['IsClosed'] & ~opps_df['IsWon']).sum():,}")
    print(f"Open:                 {(~opps_df['IsClosed']).sum():,}")
    print(f"Total ACV:            ${opps_df['ACV__c'].sum():,.0f}")
    print(f"Won ACV:              ${opps_df[opps_df['IsWon']]['ACV__c'].sum():,.0f}")

    # ──────────────────────────────────────────────────────
    # Step 3: Dimension Engineering
    # ──────────────────────────────────────────────────────
    print_section("Step 3: Dimension Engineering")

    opps_df = tag_fiscal_quarters(opps_df, date_column="CloseDate")
    print(f"Fiscal quarters found: {sorted(opps_df['fiscal_quarter'].dropna().unique())}")
    print(f"Current quarter deals: {opps_df['is_current_quarter'].sum()}")

    # ──────────────────────────────────────────────────────
    # Step 4: Quota Cascading
    # ──────────────────────────────────────────────────────
    print_section("Step 4: DAG-Based Quota Cascading")

    # Build historical attainment for capacity weights
    won_opps = opps_df[opps_df["IsWon"] == True].copy()
    historical = (
        won_opps.groupby("rep_name")["ACV__c"]
        .sum()
        .reset_index()
        .rename(columns={"rep_name": "node_id", "ACV__c": "acv_closed"})
    )

    cascader = QuotaCascader(dag=dag, historical_attainment=historical)

    MACRO_TARGET = 500_000_000  # $500M
    HEDGE = 1.05  # 5% per level

    quotas = cascader.cascade(
        root="CRO",
        macro_target=MACRO_TARGET,
        hedge_multiplier=HEDGE,
        locked_nodes={"VP_APAC": 45_000_000},
    )

    leaf_quotas = cascader.get_leaf_quotas()
    leaf_sum = sum(leaf_quotas.values())

    print(f"Macro target:         ${MACRO_TARGET / 1e6:.0f}M")
    print(f"Hedge multiplier:     {HEDGE}x per level")
    print(f"IC quota sum:         ${leaf_sum / 1e6:.2f}M")
    print(f"Compound hedge:       ${(leaf_sum - MACRO_TARGET) / 1e6:.2f}M buffer")
    print(f"Locked VP_APAC at:    $45.00M")

    # Validation
    coherence = cascader.validate_coherence()
    print(f"Coherence check:      {'PASS' if coherence['is_coherent'] else 'FAIL'}")

    # Summary table
    summary = cascader.get_cascade_summary()
    print(f"\nCascade by level:")
    for level in ["CRO", "VP", "Director", "Manager", "IC"]:
        level_data = summary[summary["node_id"].str.startswith(level) | (summary["node_id"] == level)]
        if not level_data.empty:
            print(f"  {level:10s}: {len(level_data):3d} nodes, "
                  f"avg quota ${level_data['quota'].mean():,.0f}")

    # ──────────────────────────────────────────────────────
    # Step 5: Bias Detection & Reconciliation
    # ──────────────────────────────────────────────────────
    print_section("Step 5: Manager Bias Reconciliation")

    commits_df = generate_historical_commits(dag, node_metadata)
    reconciler = BiasReconciler(commits_df, trailing_quarters=4)

    org_summary = reconciler.get_org_summary()
    print(f"Total managers:       {org_summary['total_managers']}")
    print(f"Sandbaggers:          {org_summary['sandbaggers']}")
    print(f"Happy Ears:           {org_summary['happy_ears']}")
    print(f"Neutral:              {org_summary['neutral']}")
    print(f"Mean Bias Index:      {org_summary['mean_bias_index']:.3f}")

    # Demonstrate reconciliation for a specific manager
    managers = node_metadata[node_metadata["level"] == "Manager"]["node_id"].tolist()
    if managers:
        test_mgr = managers[0]
        result = reconcile_forecast(
            manager_id=test_mgr,
            manual_commit=400_000,
            ml_baseline=490_000,
            bias_reconciler=reconciler,
        )
        print(f"\nReconciliation example ({test_mgr}):")
        for k, v in result.to_dict().items():
            print(f"  {k:25s}: {v}")

    # ──────────────────────────────────────────────────────
    # Step 6: ML Revenue Forecast
    # ──────────────────────────────────────────────────────
    print_section("Step 6: ML Revenue Forecast")

    features_df = generate_forecast_features(opps_df)

    if len(features_df) >= 6:
        forecaster = RevenueForecaster(n_estimators=100)
        result = forecaster.train(features_df, n_cv_splits=3)
        print(result.summary())

        importance = forecaster.get_feature_importance()
        print(f"\nFeature Importance Ranking:")
        for _, row in importance.iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['rank']}. {row['feature']:25s} {row['importance']:.3f} {bar}")
    else:
        print(f"Insufficient data for CV ({len(features_df)} samples). "
              f"Need >= 6 for 3-fold split.")

    # ──────────────────────────────────────────────────────
    # Step 7: Weekly Forecast Submission
    # ──────────────────────────────────────────────────────
    print_section("Step 7: Weekly Forecast Workflow")

    forecast_mgr = WeeklyForecastManager()

    # Simulate 3 weeks of submissions from 3 managers
    test_managers = managers[:3] if len(managers) >= 3 else managers
    for week in range(3):
        for mgr in test_managers:
            base = 300_000 + week * 15_000
            forecast_mgr.submit_forecast(
                manager_id=mgr,
                forecast_period="FY25Q2",
                commit_amount=base,
                best_case_amount=base * 1.15,
                submitted_by=mgr,
                notes=f"Week {week + 1} forecast",
            )

    summary_df = forecast_mgr.get_forecast_summary("FY25Q2")
    print("Latest forecast summary:")
    print(summary_df[["manager_id", "commit_amount", "best_case_amount"]].to_string(index=False))

    # Detect forecast drift
    drifters = forecast_mgr.detect_forecast_drift("FY25Q2")
    if drifters:
        print(f"\nForecast drift detected ({len(drifters)} managers):")
        for d in drifters:
            print(f"  {d['manager_id']}: {d['drift_direction']} "
                  f"{abs(d['total_drift']):.1%} over {d['n_revisions']} revisions")

    # ──────────────────────────────────────────────────────
    print_section("Demo Complete")
    print("All 7 pipeline stages executed successfully.")
    print("See the companion articles for full documentation:")
    print("  - TowardsAI article:  towards_ai_article.md")
    print("  - Research paper:     eswa_research_paper.md")


if __name__ == "__main__":
    main()
