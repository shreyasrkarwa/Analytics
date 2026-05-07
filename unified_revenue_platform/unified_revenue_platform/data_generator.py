"""
Synthetic Data Generator for Testing and Demonstration
======================================================

Generates realistic B2B enterprise sales data for testing all URP
modules without requiring access to production CRM systems.

Data characteristics mirror real enterprise SaaS patterns:
    - 5-level organizational hierarchy (CRO -> VP -> Dir -> Mgr -> IC)
    - 3 regions (AMER, EMEA, APAC) with realistic revenue splits
    - 3 segments (Enterprise, Mid-Market, SMB) with different ASPs
    - Seasonal patterns and fiscal calendar alignment
    - Manager bias distributions matching observed archetypes
"""

import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Optional


def generate_org_hierarchy(
    n_ics: int = 50,
    regions: Optional[list] = None,
    seed: int = 42,
) -> tuple:
    """
    Generate a realistic organizational hierarchy DAG.

    Creates a 5-level hierarchy:
        CRO -> Regional VPs -> Directors -> Managers -> ICs

    Args:
        n_ics: Total number of individual contributors.
        regions: List of regions. Default: ['AMER', 'EMEA', 'APAC'].
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (DAG: nx.DiGraph, node_metadata: pd.DataFrame).
    """
    rng = np.random.default_rng(seed)
    regions = regions or ["AMER", "EMEA", "APAC"]

    dag = nx.DiGraph()
    metadata = []

    # Root
    dag.add_node("CRO")
    metadata.append({"node_id": "CRO", "level": "CRO", "region": "Global"})

    # VPs (one per region)
    vps = [f"VP_{r}" for r in regions]
    for vp, region in zip(vps, regions):
        dag.add_edge("CRO", vp)
        metadata.append({"node_id": vp, "level": "VP", "region": region})

    # Distribute ICs across regions (weighted: AMER 50%, EMEA 30%, APAC 20%)
    region_weights = {"AMER": 0.5, "EMEA": 0.3, "APAC": 0.2}
    ic_counts = {
        r: max(4, int(n_ics * region_weights.get(r, 1 / len(regions))))
        for r in regions
    }

    ic_id = 0
    for region, vp in zip(regions, vps):
        n_region_ics = ic_counts[region]
        n_directors = max(1, n_region_ics // 8)
        n_managers = max(n_directors, n_region_ics // 4)

        # Directors
        directors = [f"Dir_{region}_{d}" for d in range(n_directors)]
        for d in directors:
            dag.add_edge(vp, d)
            metadata.append({"node_id": d, "level": "Director", "region": region})

        # Managers (distributed across directors)
        managers = [f"Mgr_{region}_{m}" for m in range(n_managers)]
        for i, mgr in enumerate(managers):
            director = directors[i % len(directors)]
            dag.add_edge(director, mgr)
            metadata.append({"node_id": mgr, "level": "Manager", "region": region})

        # ICs (distributed across managers)
        for i in range(n_region_ics):
            ic_name = f"IC_{region}_{ic_id}"
            manager = managers[i % len(managers)]
            dag.add_edge(manager, ic_name)
            metadata.append({"node_id": ic_name, "level": "IC", "region": region})
            ic_id += 1

    return dag, pd.DataFrame(metadata)


def generate_opportunities(
    dag: nx.DiGraph,
    node_metadata: pd.DataFrame,
    n_quarters: int = 8,
    fy_start_month: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic opportunity data for all ICs in the hierarchy.

    Creates realistic B2B SaaS opportunities with:
        - Segment-dependent ASP (Enterprise: $80-250K, Mid: $20-80K, SMB: $5-20K)
        - Stage progression with realistic win rates
        - Seasonal patterns (Q4 spike, Q1 dip)
        - Manager hierarchy stamped on each opportunity

    Args:
        dag: Organizational DAG from generate_org_hierarchy.
        node_metadata: Node metadata DataFrame.
        n_quarters: Number of fiscal quarters to generate.
        fy_start_month: Fiscal year start month.
        seed: Random seed.

    Returns:
        DataFrame mimicking Salesforce opportunity extract.
    """
    rng = np.random.default_rng(seed)

    # Get leaf nodes (ICs)
    ics = [n for n in dag.nodes if dag.out_degree(n) == 0]
    ic_meta = node_metadata[node_metadata["level"] == "IC"]

    segments = ["Enterprise", "Mid-Market", "SMB"]
    segment_config = {
        "Enterprise": {"asp_range": (80_000, 250_000), "win_rate": 0.20, "deals_per_q": 3},
        "Mid-Market": {"asp_range": (20_000, 80_000), "win_rate": 0.30, "deals_per_q": 6},
        "SMB": {"asp_range": (5_000, 20_000), "win_rate": 0.45, "deals_per_q": 10},
    }

    stages = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
    stage_probs = {"Prospecting": 0.6, "Qualification": 0.5, "Proposal": 0.7, "Negotiation": 0.8}

    records = []
    opp_id = 1000

    # Generate quarters
    base_date = datetime(2023, fy_start_month, 1)
    quarters = []
    for q in range(n_quarters):
        q_start = base_date + timedelta(days=q * 91)
        quarters.append(q_start)

    for ic in ics:
        region = ic_meta[ic_meta["node_id"] == ic]["region"].values[0]
        segment = rng.choice(segments, p=[0.3, 0.4, 0.3])
        config = segment_config[segment]

        # Find manager chain
        predecessors = list(nx.ancestors(dag, ic))
        managers = [n for n in predecessors if n.startswith("Mgr_")]
        directors = [n for n in predecessors if n.startswith("Dir_")]
        mgr1 = managers[0] if managers else None
        mgr2 = directors[0] if directors else None

        for q_start in quarters:
            # Seasonal adjustment (Q4 spike)
            q_num = ((q_start.month - fy_start_month) % 12) // 3 + 1
            seasonal = {1: 0.85, 2: 0.95, 3: 1.0, 4: 1.20}.get(q_num, 1.0)

            n_deals = max(1, int(config["deals_per_q"] * seasonal * rng.normal(1, 0.2)))

            for _ in range(n_deals):
                opp_id += 1
                acv = rng.uniform(*config["asp_range"])

                # Simulate stage progression
                final_stage = stages[0]
                is_won = False
                is_closed = False

                for stage in stages[:-2]:  # Up to Negotiation
                    if rng.random() < stage_probs.get(stage, 0.5):
                        final_stage = stages[stages.index(stage) + 1]
                    else:
                        break

                # Win/loss at final stage
                if final_stage == "Negotiation":
                    if rng.random() < config["win_rate"] * seasonal:
                        final_stage = "Closed Won"
                        is_won = True
                        is_closed = True
                    else:
                        final_stage = "Closed Lost"
                        is_closed = True

                close_date = q_start + timedelta(days=rng.integers(0, 90))

                records.append({
                    "Id": f"006{opp_id:010d}",
                    "Name": f"Opp-{opp_id}",
                    "AccountId": f"001{rng.integers(1000, 9999):04d}",
                    "OwnerId": ic,
                    "rep_name": ic,
                    "mgr1_name": mgr1,
                    "mgr2_name": mgr2,
                    "Amount": round(acv, 2),
                    "ACV__c": round(acv, 2),
                    "StageName": final_stage,
                    "CloseDate": close_date,
                    "ForecastCategoryName": (
                        "Closed" if is_closed
                        else "Commit" if final_stage in ("Negotiation", "Proposal")
                        else "Pipeline"
                    ),
                    "Probability": (
                        1.0 if is_won
                        else 0.0 if (is_closed and not is_won)
                        else stage_probs.get(final_stage, 0.3)
                    ),
                    "Type": "New Business",
                    "LeadSource": rng.choice(["Outbound", "Inbound", "Partner", "AE-Sourced"]),
                    "CreatedDate": q_start - timedelta(days=rng.integers(10, 60)),
                    "LastModifiedDate": close_date - timedelta(days=rng.integers(0, 5)),
                    "IsClosed": is_closed,
                    "IsWon": is_won,
                    "Segment__c": segment,
                    "Region__c": region,
                    "Sub_Region__c": f"{region}_sub",
                })

    return pd.DataFrame(records)


def generate_historical_commits(
    dag: nx.DiGraph,
    node_metadata: pd.DataFrame,
    n_quarters: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic historical commit vs. actual data for managers.

    Creates realistic bias patterns:
        - ~30% of managers are Sandbaggers (beta > 1.10)
        - ~20% are Happy Ears (beta < 0.90)
        - ~50% are Neutral (0.90 <= beta <= 1.10)

    Args:
        dag: Organizational DAG.
        node_metadata: Node metadata DataFrame.
        n_quarters: Number of quarters of history.
        seed: Random seed.

    Returns:
        DataFrame with columns: manager_id, fiscal_quarter,
        manual_commit, actual_closed.
    """
    rng = np.random.default_rng(seed)

    managers = node_metadata[node_metadata["level"] == "Manager"]["node_id"].tolist()

    records = []
    for mgr in managers:
        # Assign an archetype bias
        archetype_roll = rng.random()
        if archetype_roll < 0.30:
            # Sandbagger: actual > commit
            base_bias = rng.uniform(1.12, 1.35)
        elif archetype_roll < 0.50:
            # Happy Ears: actual < commit
            base_bias = rng.uniform(0.70, 0.88)
        else:
            # Neutral
            base_bias = rng.uniform(0.92, 1.08)

        for q in range(n_quarters):
            quarter_label = f"FY{23 + q // 4}Q{q % 4 + 1}"
            base_revenue = rng.uniform(200_000, 800_000)
            commit = base_revenue
            actual = commit * base_bias * rng.normal(1.0, 0.05)

            records.append({
                "manager_id": mgr,
                "fiscal_quarter": quarter_label,
                "manual_commit": round(commit, 2),
                "actual_closed": round(max(0, actual), 2),
            })

    return pd.DataFrame(records)


def generate_forecast_features(
    opps_df: pd.DataFrame,
    n_quarters: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the 7-feature matrix for the revenue forecasting model.

    Args:
        opps_df: Opportunities DataFrame.
        n_quarters: Number of quarters.
        seed: Random seed.

    Returns:
        Feature matrix aligned with RevenueForecaster.DEFAULT_FEATURES.
    """
    rng = np.random.default_rng(seed)

    quarters = opps_df["CloseDate"].dt.to_period("Q").unique()
    records = []

    for q in quarters:
        q_opps = opps_df[opps_df["CloseDate"].dt.to_period("Q") == q]
        open_pipeline = q_opps[q_opps["IsClosed"] == False]["ACV__c"].sum()
        closed_acv = q_opps[q_opps["IsWon"] == True]["ACV__c"].sum()

        records.append({
            "fiscal_quarter": str(q),
            "open_pipeline_acv": open_pipeline + rng.uniform(100_000, 500_000),
            "pipeline_coverage_ratio": rng.uniform(1.5, 4.5),
            "avg_deal_age_days": rng.uniform(30, 90),
            "pct_stage4_plus": rng.uniform(0.15, 0.55),
            "avg_win_rate_trailing": rng.uniform(0.20, 0.40),
            "manager_bias_index": rng.uniform(0.85, 1.25),
            "headcount_quota_ratio": rng.uniform(0.7, 1.1),
            "actual_closed_acv": closed_acv if closed_acv > 0 else rng.uniform(500_000, 2_000_000),
        })

    return pd.DataFrame(records)
