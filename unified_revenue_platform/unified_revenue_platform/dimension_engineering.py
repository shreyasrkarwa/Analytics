"""
Layer 2 — Dimension Engineering: Bronze to Silver Transformations
================================================================

Transforms raw CRM data into analytically useful dimensions that encode
enterprise-specific business logic: fiscal calendars, pipeline coverage
ratios, segmented win rates, and deal velocity metrics.

Design Principle:
    Raw Salesforce dimensions are generic — they know nothing about your
    company's fiscal calendar, GTM motion, or organizational structure.
    The Silver pipeline engineers these custom dimensions so that
    downstream consumers (quota cascading, forecasting, presentation)
    operate on semantically meaningful fields.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional, Union


def get_fiscal_quarter(
    close_date: Union[datetime, date, None],
    fy_start_month: int = 8,
) -> Optional[str]:
    """
    Map a calendar date to a fiscal quarter string.

    Enterprise sales runs on fiscal calendars that rarely align with
    the calendar year. This function handles the mapping for any
    fiscal year start month.

    Args:
        close_date: The date to map. Can be datetime or date.
        fy_start_month: Month the fiscal year starts (default 8 = August,
            common for SaaS companies like Salesforce, Atlassian).

    Returns:
        Fiscal quarter string, e.g., 'FY25Q1', or None if input is None.

    Examples:
        >>> get_fiscal_quarter(datetime(2024, 9, 15), fy_start_month=8)
        'FY25Q1'
        >>> get_fiscal_quarter(datetime(2025, 1, 10), fy_start_month=8)
        'FY25Q2'
        >>> get_fiscal_quarter(datetime(2025, 7, 30), fy_start_month=8)
        'FY25Q4'
    """
    if close_date is None:
        return None

    month = close_date.month
    year = close_date.year

    # Shift months so fiscal year starts at month 1
    fiscal_month = (month - fy_start_month) % 12 + 1
    fiscal_quarter = (fiscal_month - 1) // 3 + 1
    fiscal_year = year if month >= fy_start_month else year - 1

    return f"FY{str(fiscal_year + 1)[-2:]}Q{fiscal_quarter}"


def tag_fiscal_quarters(
    df: pd.DataFrame,
    date_column: str = "CloseDate",
    fy_start_month: int = 8,
) -> pd.DataFrame:
    """
    Add fiscal_quarter and is_current_quarter columns to a DataFrame.

    Args:
        df: DataFrame with a date column.
        date_column: Name of the date column to map.
        fy_start_month: Month the fiscal year starts.

    Returns:
        DataFrame with added columns: fiscal_quarter, is_current_quarter.
    """
    df = df.copy()
    df["fiscal_quarter"] = df[date_column].apply(
        lambda d: get_fiscal_quarter(d, fy_start_month)
    )

    current_fq = get_fiscal_quarter(datetime.utcnow(), fy_start_month)
    df["is_current_quarter"] = df["fiscal_quarter"] == current_fq

    return df


def compute_pipeline_coverage(
    opps_df: pd.DataFrame,
    quotas_df: pd.DataFrame,
    owner_col: str = "OwnerId",
    acv_col: str = "ACV__c",
    quota_col: str = "quota_acv",
    healthy_threshold: float = 3.0,
    moderate_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Compute pipeline coverage ratio per individual contributor.

    Pipeline coverage is one of the most operationally critical metrics
    in enterprise sales. It answers: "For every dollar of quota, how many
    dollars of open pipeline does this rep carry?"

    Formula:
        coverage_ratio = open_pipeline_acv / quota_acv

    Industry benchmarks:
        >= 3.0x : Healthy (3x pipeline needed for ~33% win rate)
        1.5-3.0x: Moderate (some risk of missing quota)
        < 1.5x : At Risk (insufficient pipeline to absorb deal slippage)

    Args:
        opps_df: Opportunities DataFrame. Must contain columns for owner,
            ACV, IsClosed, and is_current_quarter.
        quotas_df: Quotas DataFrame with owner and quota columns.
        owner_col: Column identifying the opportunity owner.
        acv_col: Column with deal value (ACV or Amount).
        quota_col: Column in quotas_df with the quota value.
        healthy_threshold: Coverage ratio threshold for 'Healthy' status.
        moderate_threshold: Coverage ratio threshold for 'Moderate' status.

    Returns:
        DataFrame with columns: owner, open_pipeline_acv, quota_acv,
        coverage_ratio, coverage_status.
    """
    # Filter to open deals in the current quarter
    open_pipeline = (
        opps_df[
            (opps_df["IsClosed"] == False)
            & (opps_df.get("is_current_quarter", pd.Series(True)))
        ]
        .groupby(owner_col)
        .agg(open_pipeline_acv=(acv_col, "sum"))
        .reset_index()
    )

    # Join against quotas
    coverage = open_pipeline.merge(quotas_df, on=owner_col, how="left")

    # Compute coverage ratio
    coverage["coverage_ratio"] = np.where(
        coverage[quota_col] > 0,
        (coverage["open_pipeline_acv"] / coverage[quota_col]).round(2),
        np.nan,
    )

    # Classify coverage status
    conditions = [
        coverage["coverage_ratio"] >= healthy_threshold,
        coverage["coverage_ratio"] >= moderate_threshold,
    ]
    choices = ["Healthy", "Moderate"]
    coverage["coverage_status"] = np.select(
        conditions, choices, default="At Risk"
    )

    return coverage


def compute_win_rate_by_segment(
    opps_df: pd.DataFrame,
    trailing_quarters: int = 4,
    segment_col: str = "Segment__c",
    region_col: str = "Region__c",
    acv_col: str = "ACV__c",
) -> pd.DataFrame:
    """
    Compute rolling win rate and average selling price (ASP) by
    Segment x Region over the trailing N fiscal quarters.

    Segmenting win rates is critical for setting realistic pipeline-to-quota
    ratios. Enterprise, Mid-Market, and SMB deals close at wildly different
    rates (typically 15-25%, 25-35%, and 35-50% respectively).

    Args:
        opps_df: Opportunities DataFrame with fiscal_quarter tagged.
        trailing_quarters: Number of recent quarters to include.
        segment_col: Column for market segment.
        region_col: Column for geographic region.
        acv_col: Column for deal value.

    Returns:
        DataFrame with columns: segment, region, total_deals, won_deals,
        win_rate, avg_deal_size_acv.
    """
    closed_opps = opps_df[opps_df["IsClosed"] == True].copy()

    if closed_opps.empty:
        return pd.DataFrame()

    # Get the N most recent fiscal quarters
    recent_quarters = (
        closed_opps["fiscal_quarter"]
        .drop_duplicates()
        .sort_values(ascending=False)
        .head(trailing_quarters)
        .tolist()
    )

    trailing = closed_opps[
        closed_opps["fiscal_quarter"].isin(recent_quarters)
    ]

    win_rates = (
        trailing.groupby([segment_col, region_col])
        .agg(
            total_deals=("Id", "count"),
            won_deals=("IsWon", "sum"),
            avg_deal_size_acv=(acv_col, lambda x: x[trailing["IsWon"] == True].mean()),
        )
        .reset_index()
    )

    win_rates["win_rate"] = (
        win_rates["won_deals"] / win_rates["total_deals"]
    ).round(3)

    return win_rates


def compute_deal_velocity(
    opps_df: pd.DataFrame,
    stage_order: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute average days spent in each sales stage.

    Deal velocity metrics help identify bottlenecks in the sales process
    and forecast how quickly deals will move through the pipeline.

    Args:
        opps_df: Opportunities DataFrame with StageName and stage dates.
        stage_order: Ordered list of stage names. If None, uses a common
            B2B SaaS default: ['Prospecting', 'Qualification', 'Proposal',
            'Negotiation', 'Closed Won', 'Closed Lost'].

    Returns:
        DataFrame with average days per stage by segment.
    """
    if stage_order is None:
        stage_order = [
            "Prospecting",
            "Qualification",
            "Proposal",
            "Negotiation",
            "Closed Won",
            "Closed Lost",
        ]

    df = opps_df.copy()

    # Compute deal age
    df["deal_age_days"] = (
        pd.to_datetime("today") - pd.to_datetime(df["CreatedDate"])
    ).dt.days

    velocity = (
        df.groupby("StageName")
        .agg(
            avg_deal_age=("deal_age_days", "mean"),
            deal_count=("Id", "count"),
        )
        .reset_index()
    )

    # Order by stage sequence
    stage_map = {s: i for i, s in enumerate(stage_order)}
    velocity["stage_order"] = velocity["StageName"].map(stage_map)
    velocity = velocity.sort_values("stage_order").drop(
        columns=["stage_order"]
    )

    return velocity
