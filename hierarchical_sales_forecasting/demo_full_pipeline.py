#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   b2b_revenue_forecasting — Full Demo                      ║
║                                                                            ║
║  End-to-end demonstration of the B2B Revenue Forecasting library:          ║
║    1. Building an Org Hierarchy (SalesHierarchy)                           ║
║    2. Top-Down Quota Cascading (QuotaCascader)                             ║
║    3. Commit Reconciliation / Bias Detection (CommitReconciler)            ║
║    4. Pipeline Health Diagnosis & Redistribution (PipelineAdjuster)        ║
║                                                                            ║
║  Run:  python demo_full_pipeline.py                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np

from b2b_revenue_forecasting.hierarchy import SalesHierarchy
from b2b_revenue_forecasting.quota_cascader import QuotaCascader
from b2b_revenue_forecasting.commit_reconciler import CommitReconciler
from b2b_revenue_forecasting.pipeline_adjuster import PipelineAdjuster

SEP = "=" * 90
SUB = "-" * 90


def banner(title):
    print(f"\n\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Building the Sales Hierarchy
# ─────────────────────────────────────────────────────────────────────────────
def demo_hierarchy():
    banner("PART 1: Building the Sales Hierarchy (DAG)")

    # Load the synthetic CRM dataset
    # NOTE: keep_default_na=False prevents pandas from reading 'NA' (North America) as NaN
    df = pd.read_csv('tests/data/synthetic_hierarchy.csv', keep_default_na=False)

    print(f"\n  📄 Loaded CRM data: {len(df)} rows (one per IC)")
    print(f"  📊 Columns: {list(df.columns)}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))

    # Define the org structure (works with any depth: 3 levels or 10 levels)
    taxonomy = ['Global', 'Region', 'RVP', 'Director', 'Manager', 'IC']
    metrics = ['Q1_Attainment', 'Q2_Attainment', 'Q3_Attainment', 'Q4_Attainment', 'Current_Pipeline']

    # Build the hierarchy
    hierarchy = SalesHierarchy()
    hierarchy.from_dataframe(df, path_cols=taxonomy, metrics_cols=metrics)

    total_nodes = len(hierarchy.graph.nodes)
    total_ics = len(hierarchy.get_leaves('Global_Corp'))
    total_edges = len(hierarchy.graph.edges)

    print(f"\n  🏢 Hierarchy Built:")
    print(f"     Total Nodes:        {total_nodes}")
    print(f"     Total ICs (Leaves): {total_ics}")
    print(f"     Reporting Edges:    {total_edges}")

    # Show the children of the root
    regions = hierarchy.get_children('Global_Corp')
    print(f"\n  🌍 Regions under Global_Corp: {regions}")

    for region in regions:
        rvps = hierarchy.get_children(region)
        ic_count = len(hierarchy.get_leaves(region))
        print(f"     └── {region}: {len(rvps)} RVP(s), {ic_count} total ICs")

    return hierarchy, df


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Quota Cascading
# ─────────────────────────────────────────────────────────────────────────────
def demo_quota_cascading(hierarchy):
    banner("PART 2: Quota Cascading (QuotaCascader)")

    cascader = QuotaCascader(hierarchy)
    GLOBAL_TARGET = 100_000_000.0

    # ── 2a. Basic Cascade (No Hedge) ──
    print(f"\n  ── 2a. Basic Cascade: ${GLOBAL_TARGET:,.0f}, no hedge ──")
    quotas_flat = cascader.cascade_quota('Global_Corp', GLOBAL_TARGET)
    ic_total_flat = sum(quotas_flat[n] for n in hierarchy.get_leaves('Global_Corp'))
    print(f"     Sum of IC quotas: ${ic_total_flat:,.2f}")
    print(f"     Matches target?   {'✅ Yes' if abs(ic_total_flat - GLOBAL_TARGET) < 1.0 else '❌ No'}")

    # ── 2b. Cascade with 5% Uniform Hedge ──
    print(f"\n  ── 2b. Cascade with 5% Managerial Hedge ──")
    quotas_hedged = cascader.cascade_quota('Global_Corp', GLOBAL_TARGET, hedge_multiplier=1.05)
    ic_total_hedged = sum(quotas_hedged[n] for n in hierarchy.get_leaves('Global_Corp'))
    overassignment = ((ic_total_hedged / GLOBAL_TARGET) - 1) * 100
    print(f"     Sum of IC quotas: ${ic_total_hedged:,.2f}")
    print(f"     Compounded overassignment: {overassignment:.2f}%")
    print(f"     (5% hedge × 5 management layers = ~{1.05**5:.2%} compound)")

    # ── 2c. Cascade with Per-Node Hedge ──
    print(f"\n  ── 2c. Per-Node Hedge (VP NA gets 10%, all others 5%) ──")
    custom_hedge = {'Global_Corp': 1.05, 'NA': 1.10, 'EMEA': 1.05, 'APAC': 1.05}
    quotas_custom = cascader.cascade_quota('Global_Corp', GLOBAL_TARGET, hedge_multiplier=custom_hedge)

    na_quota = quotas_custom.get('NA', 0)
    emea_quota = quotas_custom.get('EMEA', 0)
    apac_quota = quotas_custom.get('APAC', 0)
    print(f"     NA:   ${na_quota:>14,.2f}  (10% hedge at region level)")
    print(f"     EMEA: ${emea_quota:>14,.2f}  (5% hedge)")
    print(f"     APAC: ${apac_quota:>14,.2f}  (5% hedge)")

    # ── 2d. Show Sample IC Distributions ──
    print(f"\n  ── 2d. Sample IC Quotas (weighted by 4-quarter capacity) ──")
    sample_ics = hierarchy.get_leaves('Global_Corp')[:5]
    print(f"     {'IC':<42s} {'4Q Capacity':>14s} {'Assigned Quota':>14s}")
    print(f"     {SUB}")
    for ic in sample_ics:
        cap = cascader._calculate_node_historical_capacity(ic)
        quota = quotas_hedged[ic]
        print(f"     {ic:<42s} ${cap:>12,.2f} ${quota:>12,.2f}")

    return cascader, quotas_hedged


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Commit Reconciliation (Bias Detection)
# ─────────────────────────────────────────────────────────────────────────────
def demo_commit_reconciler():
    banner("PART 3: Commit Reconciliation (CommitReconciler)")

    # Simulate historical commit data from CRM
    historical = pd.DataFrame({
        'Manager_ID':             ['Mgr_Alpha', 'Mgr_Alpha', 'Mgr_Beta', 'Mgr_Beta', 'Mgr_Gamma', 'Mgr_Gamma'],
        'Historical_Commit':      [  200_000,     250_000,    300_000,    350_000,    180_000,      220_000],
        'Historical_Actual_Closed': [300_000,     375_000,    270_000,    280_000,    180_000,      220_000],
    })

    print(f"\n  📋 Historical Commit vs. Actual Data:")
    print(historical.to_string(index=False))

    reconciler = CommitReconciler(historical)

    print(f"\n  📐 Calculated Bias Quotients:")
    for mgr, bias in reconciler.bias_weights.items():
        label = "🟢 Sandbagger" if bias > 1.05 else ("🔴 Happy Ears" if bias < 0.95 else "⚪ Neutral")
        print(f"     {mgr}: {bias:.3f}x  ({label})")

    # Reconcile a real-time commit
    print(f"\n  ── Live Reconciliation Examples ──")

    scenarios = [
        ("Mgr_Alpha", 100_000, 120_000, "Sandbagger — commit inflated"),
        ("Mgr_Beta", 200_000, 190_000, "Happy Ears — commit deflated"),
        ("Mgr_Gamma", 150_000, 155_000, "Neutral — blended with ML"),
    ]

    for mgr_id, commit, ml_forecast, note in scenarios:
        bias = reconciler.bias_weights.get(mgr_id, 1.0)

        # Without ML blend
        adjusted_human = reconciler.reconcile_forecast(mgr_id, commit)
        # With ML blend
        blended = reconciler.reconcile_forecast(mgr_id, commit, machine_forecast=ml_forecast)

        print(f"\n     {mgr_id} ({note}):")
        print(f"       Raw Commit:        ${commit:>12,.2f}")
        print(f"       Bias Quotient:            {bias:.3f}x")
        print(f"       Adjusted (human):  ${adjusted_human:>12,.2f}")
        print(f"       ML Baseline:       ${ml_forecast:>12,.2f}")
        print(f"       Blended Forecast:  ${blended:>12,.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Pipeline Health Diagnosis & Redistribution
# ─────────────────────────────────────────────────────────────────────────────
def demo_pipeline_adjuster(hierarchy, quotas):
    banner("PART 4: Pipeline Health & Redistribution (PipelineAdjuster)")

    adjuster = PipelineAdjuster(hierarchy, quotas, pipeline_attr='Current_Pipeline')

    # ── 4a. Diagnose with per-region thresholds ──
    print(f"\n  ── 4a. Pipeline Diagnosis (per-region thresholds) ──")

    thresholds = {
        'NA':       {'healthy': 1.5, 'at_risk': 0.8},   # NA: mature market, lower bar
        'EMEA':     {'healthy': 2.5, 'at_risk': 1.2},   # EMEA: higher bar
        'APAC':     {'healthy': 3.0, 'at_risk': 1.5},   # APAC: highest bar
        '_default': {'healthy': 2.0, 'at_risk': 1.0}    # Fallback
    }

    print(f"\n  Configured Thresholds:")
    for region, t in thresholds.items():
        if region != '_default':
            print(f"     {region}: healthy ≥ {t['healthy']}x, at_risk ≥ {t['at_risk']}x")
    print(f"     Default: healthy ≥ {thresholds['_default']['healthy']}x, at_risk ≥ {thresholds['_default']['at_risk']}x")

    diagnosis = adjuster.diagnose(thresholds)

    # Summary by risk status
    risk_summary = diagnosis.groupby('Risk_Status').agg(
        Count=('Node', 'count'),
        Total_Quota=('Cascaded_Quota', 'sum'),
        Total_Pipeline=('Pipeline', 'sum'),
        Avg_Coverage=('Coverage_Ratio', 'mean')
    ).reset_index()

    print(f"\n  📊 Risk Status Summary:")
    print(f"     {'Status':<10s} {'Count':>6s} {'Total Quota':>16s} {'Total Pipeline':>16s} {'Avg Coverage':>14s}")
    print(f"     {SUB}")
    for _, row in risk_summary.iterrows():
        status_icon = {'Healthy': '🟢', 'Moderate': '🟡', 'At Risk': '🟠', 'Critical': '🔴'}.get(row['Risk_Status'], '⚪')
        print(f"     {status_icon} {row['Risk_Status']:<8s} {row['Count']:>6d} ${row['Total_Quota']:>14,.2f} ${row['Total_Pipeline']:>14,.2f} {row['Avg_Coverage']:>12.2f}x")

    # Show region-level diagnosis
    region_diagnosis = diagnosis[diagnosis['Level'] == 1]
    print(f"\n  🌍 Region-Level Diagnosis:")
    for _, row in region_diagnosis.iterrows():
        print(f"     {row['Node']:<8s} | Quota: ${row['Cascaded_Quota']:>14,.2f} | Pipeline: ${row['Pipeline']:>14,.2f} | Coverage: {row['Coverage_Ratio']:.2f}x | {row['Risk_Status']}")

    # ── 4b. Flag-Only Mode ──
    print(f"\n  ── 4b. Flag-Only Mode (read-only, no changes) ──")
    flagged = adjuster.adjust(mode='flag_only', coverage_thresholds=thresholds)
    unchanged = all(abs(flagged[n] - quotas[n]) < 0.01 for n in quotas)
    print(f"     Quotas unchanged? {'✅ Yes' if unchanged else '❌ No'}")
    print(f"     (Use this mode to diagnose before committing to redistribution)")

    # ── 4c. Redistribute Mode ──
    print(f"\n  ── 4c. Redistribute Mode (zero-sum IC adjustment) ──")
    redistributed = adjuster.adjust(
        mode='redistribute',
        coverage_thresholds=thresholds,
        max_adjustment_pct=0.20  # No IC can change by more than 20%
    )

    # Verify zero-sum per manager
    manager_nodes = [n for n in hierarchy.graph.nodes
                     if list(hierarchy.graph.successors(n))
                     and all(hierarchy.graph.out_degree(c) == 0 for c in hierarchy.graph.successors(n))]

    all_zero_sum = True
    for mgr in manager_nodes:
        ics = list(hierarchy.graph.successors(mgr))
        orig_total = sum(quotas.get(ic, 0) for ic in ics)
        adj_total = sum(redistributed.get(ic, 0) for ic in ics)
        if abs(orig_total - adj_total) > 1.0:
            all_zero_sum = False

    leaves = [n for n in hierarchy.graph.nodes if hierarchy.graph.out_degree(n) == 0]
    adjustments_made = sum(1 for ic in leaves if abs(redistributed.get(ic, 0) - quotas.get(ic, 0)) > 0.01)

    print(f"     ✅ Manager-level totals preserved (zero-sum): {'Yes' if all_zero_sum else 'No'}")
    print(f"     📊 ICs adjusted: {adjustments_made} / {len(leaves)}")
    print(f"     🛡️  Max adjustment cap: 20%")

    # Show the biggest movers
    deltas = []
    for ic in leaves:
        orig = quotas.get(ic, 0)
        adj = redistributed.get(ic, 0)
        if orig > 0:
            pct = ((adj - orig) / orig) * 100
            deltas.append((ic, orig, adj, pct))

    deltas.sort(key=lambda x: x[3])

    print(f"\n  📉 Top 3 Quota Reductions (Donors — strong pipeline, share the load):")
    print(f"     {'IC':<42s} {'Original':>14s} {'Adjusted':>14s} {'Change':>10s}")
    print(f"     {SUB}")
    for ic, orig, adj, pct in deltas[:3]:
        print(f"     {ic:<42s} ${orig:>12,.2f} ${adj:>12,.2f} {pct:>+8.1f}%")

    print(f"\n  📈 Top 3 Quota Increases (Receivers — weak pipeline, get help):")
    print(f"     {'IC':<42s} {'Original':>14s} {'Adjusted':>14s} {'Change':>10s}")
    print(f"     {SUB}")
    for ic, orig, adj, pct in deltas[-3:]:
        print(f"     {ic:<42s} ${orig:>12,.2f} ${adj:>12,.2f} {pct:>+8.1f}%")

    # ── 4d. Locked Nodes ──
    print(f"\n  ── 4d. Locked Nodes (CRO-protected ICs) ──")
    # Pick an IC that exists in the adjusted quotas
    locked_ic = [ic for ic in leaves if ic in quotas][0]
    locked_quota = 999_999.99
    adj_locked = adjuster.adjust(
        mode='redistribute',
        coverage_thresholds=thresholds,
        locked_nodes={locked_ic: locked_quota}
    )
    print(f"     Locked: {locked_ic[:40]}...")
    print(f"     Fixed quota: ${locked_quota:,.2f}")
    print(f"     Actual in output: ${adj_locked[locked_ic]:,.2f}")
    print(f"     Protected? {'✅ Yes' if abs(adj_locked[locked_ic] - locked_quota) < 0.01 else '❌ No'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'═' * 90}")
    print(f"  b2b_revenue_forecasting v0.2.0 — Full Pipeline Demo")
    print(f"  pip install b2b-revenue-forecasting==0.2.0")
    print(f"{'═' * 90}")

    # Part 1: Build the hierarchy
    hierarchy, df = demo_hierarchy()

    # Part 2: Cascade quotas
    cascader, quotas = demo_quota_cascading(hierarchy)

    # Part 3: Commit reconciliation
    demo_commit_reconciler()

    # Part 4: Pipeline health & redistribution
    demo_pipeline_adjuster(hierarchy, quotas)

    print(f"\n\n{'═' * 90}")
    print(f"  ✅ DEMO COMPLETE")
    print(f"{'═' * 90}\n")
