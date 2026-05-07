"""
Layer 1 — Data Ingestion: Salesforce CRM to Delta Lake Bronze
=============================================================

Handles authenticated extraction of CRM objects (Opportunity, Account,
User, OpportunityHistory) via the Salesforce SOQL API, with merge-upsert
into a Bronze Delta Lake table.

Design Principle:
    The Bronze layer is deliberately raw — no transformations, no business
    logic. Its job is fidelity to the source system. All business logic
    lives in the Silver/Gold layers downstream.

Key Engineering Decision:
    Manager hierarchy (Owner -> Manager -> Manager's Manager) is pulled
    directly inside the SOQL query via relationship traversal, avoiding
    expensive post-hoc joins and ensuring every opportunity is stamped
    with its full organizational lineage at extraction time.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def get_sf_connection(
    username: str,
    password: str,
    security_token: str,
    domain: str = "login",
):
    """
    Establish an authenticated Salesforce connection.

    Args:
        username: Salesforce service account username.
        password: Service account password (from secrets manager).
        security_token: Salesforce security token.
        domain: 'login' for production, 'test' for sandbox.

    Returns:
        Authenticated simple_salesforce.Salesforce instance.

    Raises:
        ImportError: If simple-salesforce is not installed.
        SalesforceAuthenticationFailed: If credentials are invalid.
    """
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        raise ImportError(
            "simple-salesforce is required for CRM ingestion. "
            "Install with: pip install simple-salesforce"
        )

    return Salesforce(
        username=username,
        password=password,
        security_token=security_token,
        domain=domain,
    )


def extract_opportunities(
    sf,
    lookback_days: int = 90,
    custom_fields: Optional[list] = None,
) -> pd.DataFrame:
    """
    Pull all open and recently-closed opportunities from Salesforce.

    Extracts core opportunity fields plus the full manager hierarchy
    (Owner -> Manager -> Manager's Manager) via SOQL relationship
    traversal. The lookback window ensures we capture deals closed
    in the current quarter for accurate attainment tracking.

    Args:
        sf: Authenticated Salesforce connection.
        lookback_days: Days to look back for closed opportunities.
        custom_fields: Additional custom fields to extract (e.g.,
            ['Custom_Field__c', 'Another__c']).

    Returns:
        DataFrame with flattened opportunity data including:
        - rep_name, mgr1_name, mgr2_name (organizational lineage)
        - CloseDate (datetime), Amount/ACV__c (numeric)
        - All standard Salesforce opportunity fields
    """
    cutoff = (
        datetime.utcnow() - timedelta(days=lookback_days)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    extra = ""
    if custom_fields:
        extra = ", " + ", ".join(custom_fields)

    soql = f"""
        SELECT
            Id,
            Name,
            AccountId,
            Account.Name,
            OwnerId,
            Owner.Name,
            Owner.Manager.Name,
            Owner.Manager.Manager.Name,
            Amount,
            ACV__c,
            StageName,
            CloseDate,
            ForecastCategoryName,
            Probability,
            Type,
            LeadSource,
            CreatedDate,
            LastModifiedDate,
            IsClosed,
            IsWon,
            Segment__c,
            Region__c,
            Sub_Region__c{extra}
        FROM Opportunity
        WHERE (IsClosed = false OR CloseDate >= {cutoff})
        AND IsDeleted = false
        ORDER BY LastModifiedDate DESC
    """

    records = sf.query_all(soql)["records"]
    df = pd.DataFrame(records).drop(columns=["attributes"], errors="ignore")

    # Flatten nested Owner.Manager hierarchy
    df["rep_name"] = df["Owner"].apply(
        lambda x: x.get("Name") if isinstance(x, dict) else None
    )
    df["mgr1_name"] = df["Owner"].apply(
        lambda x: (
            x.get("Manager", {}).get("Name")
            if isinstance(x, dict) and isinstance(x.get("Manager"), dict)
            else None
        )
    )
    df["mgr2_name"] = df["Owner"].apply(
        lambda x: (
            x.get("Manager", {}).get("Manager", {}).get("Name")
            if isinstance(x, dict)
            and isinstance(x.get("Manager"), dict)
            and isinstance(x.get("Manager", {}).get("Manager"), dict)
            else None
        )
    )
    df = df.drop(columns=["Owner"], errors="ignore")

    # Type coercions
    df["CloseDate"] = pd.to_datetime(df["CloseDate"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
    df["ACV__c"] = pd.to_numeric(df["ACV__c"], errors="coerce").fillna(0)

    return df


def upsert_to_delta(
    df: pd.DataFrame,
    delta_path: str,
    merge_key: str = "Id",
):
    """
    Merge-upsert a pandas DataFrame into a Bronze Delta Lake table.

    Uses Salesforce record Id as the merge key. New records are inserted;
    existing records are updated with the latest field values.

    Args:
        df: DataFrame to upsert.
        delta_path: DBFS path for the Delta table
            (e.g., 'dbfs:/mnt/urp/bronze/opportunities').
        merge_key: Column to use for merge deduplication.

    Note:
        Requires a running Spark session with Delta Lake support.
        Typically executed within a Databricks notebook or job.
    """
    try:
        from pyspark.sql import SparkSession
        from delta.tables import DeltaTable
    except ImportError:
        raise ImportError(
            "PySpark and delta-spark are required for Delta Lake operations. "
            "This function is designed to run in a Databricks environment."
        )

    spark = SparkSession.builder.appName("URP-Ingestion").getOrCreate()
    spark_df = spark.createDataFrame(df)

    if DeltaTable.isDeltaTable(spark, delta_path):
        bronze = DeltaTable.forPath(spark, delta_path)
        (
            bronze.alias("target")
            .merge(
                spark_df.alias("source"),
                f"target.{merge_key} = source.{merge_key}",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        spark_df.write.format("delta").save(delta_path)

    print(f"Upserted {len(df):,} records to {delta_path}")


def extract_opportunity_history(sf, opportunity_ids: list) -> pd.DataFrame:
    """
    Extract stage-change history for a set of opportunities.

    Useful for computing deal velocity metrics (average days per stage),
    stage-skip patterns, and regression detection.

    Args:
        sf: Authenticated Salesforce connection.
        opportunity_ids: List of Salesforce Opportunity Ids.

    Returns:
        DataFrame with columns: OpportunityId, StageName, CreatedDate,
        OldValue, NewValue.
    """
    # Chunk to avoid SOQL IN clause limits (max ~200)
    chunk_size = 200
    all_records = []

    for i in range(0, len(opportunity_ids), chunk_size):
        chunk = opportunity_ids[i : i + chunk_size]
        ids_str = "','".join(chunk)
        soql = f"""
            SELECT
                OpportunityId,
                StageName,
                CreatedDate,
                OldValue,
                NewValue
            FROM OpportunityFieldHistory
            WHERE OpportunityId IN ('{ids_str}')
            AND Field = 'StageName'
            ORDER BY CreatedDate ASC
        """
        records = sf.query_all(soql)["records"]
        all_records.extend(records)

    df = pd.DataFrame(all_records).drop(columns=["attributes"], errors="ignore")
    if not df.empty:
        df["CreatedDate"] = pd.to_datetime(df["CreatedDate"])
    return df
