"""
Layer 3 — Commit Reconciliation: The Bias Index
================================================

Solves the human problem in revenue forecasting. Every quarter, managers
submit manual "commits" — personal guarantees of how much revenue will
close. These commits systematically diverge from algorithmic baselines
due to two universal behavioral archetypes:

    Sandbaggers: Consistently under-commit to protect their bonus.
        Actual/Commit ratio > 1.0
    Happy Ears: Consistently over-commit due to irrational optimism.
        Actual/Commit ratio < 1.0

The Bias Index (beta) quantifies this systematically and enables
bias-corrected blending of human judgment with ML baselines.

Mathematical Framework:
    beta_i = (1/K) * sum_{k=1}^{K} (actual_{i,k} / commit_{i,k})

    where K is the number of trailing quarters and i indexes the manager.

    Adjusted forecast:
        F_adj = (1 - w) * F_ml + w * (commit * beta)

    where w is the bias weight (how much to trust the human signal
    after correction) and F_ml is the pure ML baseline.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReconciliationResult:
    """Result of a forecast reconciliation for a single manager."""

    manager_id: str
    manual_commit: float
    bias_index: float
    archetype: str
    bias_corrected_commit: float
    ml_baseline: float
    adjusted_forecast: float
    hidden_upside: float
    confidence_interval: tuple

    def to_dict(self) -> dict:
        return {
            "manager_id": self.manager_id,
            "manual_commit": self.manual_commit,
            "bias_index": round(self.bias_index, 3),
            "archetype": self.archetype,
            "bias_corrected_commit": round(self.bias_corrected_commit, 2),
            "ml_baseline": self.ml_baseline,
            "adjusted_forecast": round(self.adjusted_forecast, 2),
            "hidden_upside": round(self.hidden_upside, 2),
            "confidence_interval": (
                round(self.confidence_interval[0], 2),
                round(self.confidence_interval[1], 2),
            ),
        }


class BiasReconciler:
    """
    Compute per-manager Bias Index from historical commit vs. actual data.

    The Bias Index captures systematic directional forecast error at the
    individual manager level. Unlike aggregate error metrics (MAPE, RMSE),
    it preserves the sign of the bias, enabling corrective adjustment
    rather than just accuracy measurement.

    Attributes:
        bias_df: DataFrame with computed bias indices per manager.
        archetype_thresholds: Tuple of (sandbagger_threshold,
            happy_ears_threshold) for classifying managers.
    """

    def __init__(
        self,
        historical_commits: pd.DataFrame,
        manager_col: str = "manager_id",
        quarter_col: str = "fiscal_quarter",
        commit_col: str = "manual_commit",
        actual_col: str = "actual_closed",
        trailing_quarters: int = 4,
        sandbagger_threshold: float = 1.10,
        happy_ears_threshold: float = 0.90,
    ):
        """
        Initialize the reconciler with historical commit data.

        Args:
            historical_commits: DataFrame with historical commit vs.
                actual performance per manager per quarter.
            manager_col: Column identifying the manager.
            quarter_col: Column identifying the fiscal quarter.
            commit_col: Column with the manager's manual commit amount.
            actual_col: Column with the actual closed amount.
            trailing_quarters: Number of recent quarters to compute
                the index over. More quarters = more stable index but
                slower to react to behavioral changes.
            sandbagger_threshold: Bias index above this value classifies
                as 'Sandbagger' (default 1.10 = consistently delivers
                10%+ above commit).
            happy_ears_threshold: Bias index below this value classifies
                as 'Happy Ears' (default 0.90 = consistently delivers
                10%+ below commit).
        """
        self.archetype_thresholds = (
            sandbagger_threshold,
            happy_ears_threshold,
        )

        self.bias_df = self._compute_bias_index(
            historical_commits,
            manager_col,
            quarter_col,
            commit_col,
            actual_col,
            trailing_quarters,
        )

    def _compute_bias_index(
        self,
        df: pd.DataFrame,
        manager_col: str,
        quarter_col: str,
        commit_col: str,
        actual_col: str,
        trailing_quarters: int,
    ) -> pd.DataFrame:
        """Compute bias index and classify archetypes."""
        # Sort by quarter descending and take trailing N per manager
        df = df.sort_values(quarter_col, ascending=False)
        recent = df.groupby(manager_col).head(trailing_quarters)

        # Avoid division by zero
        recent = recent[recent[commit_col] > 0].copy()

        # Compute per-quarter ratios
        recent["ratio"] = recent[actual_col] / recent[commit_col]

        # Aggregate: mean and std of ratios per manager
        bias = (
            recent.groupby(manager_col)["ratio"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        bias.columns = [manager_col, "bias_index", "bias_std", "n_quarters"]
        bias["bias_std"] = bias["bias_std"].fillna(0)

        # Classify archetypes
        sb_thresh, he_thresh = self.archetype_thresholds
        bias["archetype"] = bias["bias_index"].apply(
            lambda b: (
                "Sandbagger"
                if b > sb_thresh
                else ("Happy Ears" if b < he_thresh else "Neutral")
            )
        )

        return bias

    def get_bias(self, manager_id: str) -> tuple:
        """
        Get the bias index and archetype for a specific manager.

        Returns:
            Tuple of (bias_index, archetype, bias_std).
            Returns (1.0, 'Unknown', 0.0) if manager not found.
        """
        row = self.bias_df[
            self.bias_df.iloc[:, 0] == manager_id
        ]
        if row.empty:
            return (1.0, "Unknown", 0.0)
        return (
            row["bias_index"].values[0],
            row["archetype"].values[0],
            row["bias_std"].values[0],
        )

    def get_org_summary(self) -> dict:
        """
        Return summary statistics about bias across the organization.

        Useful for executive reporting on forecast accuracy culture.

        Returns:
            Dict with archetype counts, mean bias, and bias distribution.
        """
        df = self.bias_df
        return {
            "total_managers": len(df),
            "sandbaggers": int((df["archetype"] == "Sandbagger").sum()),
            "happy_ears": int((df["archetype"] == "Happy Ears").sum()),
            "neutral": int((df["archetype"] == "Neutral").sum()),
            "mean_bias_index": round(df["bias_index"].mean(), 3),
            "median_bias_index": round(df["bias_index"].median(), 3),
            "std_bias_index": round(df["bias_index"].std(), 3),
        }


def reconcile_forecast(
    manager_id: str,
    manual_commit: float,
    ml_baseline: float,
    bias_reconciler: BiasReconciler,
    bias_weight: float = 0.6,
    confidence_level: float = 0.90,
) -> ReconciliationResult:
    """
    Blend a manager's manual commit with the ML baseline, adjusted
    for their computed Bias Index.

    Formula:
        adjusted = (1 - w) * F_ml + w * (commit * beta)

    where:
        w = bias_weight (how much to trust the bias-corrected human signal)
        F_ml = ML baseline forecast
        beta = manager's bias index
        commit = manager's manual commit

    The bias_weight parameter controls the human-vs-machine blend:
        - 0.6 (default): 60% weight on bias-corrected human judgment,
          40% on pure ML. Appropriate when managers have domain context
          the model lacks (deal-level knowledge, relationship signals).
        - 0.3: ML-dominated blend. Use when managers have poor forecasting
          track records even after bias correction.
        - 0.8: Human-dominated blend. Use when the ML model has limited
          training data or when regulatory/contractual factors make
          human judgment essential.

    Args:
        manager_id: Identifier for the manager.
        manual_commit: Manager's stated revenue commit for the period.
        ml_baseline: ML model's forecast for this manager's team.
        bias_reconciler: Fitted BiasReconciler instance.
        bias_weight: Weight on the bias-corrected commit (0-1).
        confidence_level: Confidence level for the interval estimate.

    Returns:
        ReconciliationResult with all computed values.
    """
    bias_index, archetype, bias_std = bias_reconciler.get_bias(manager_id)

    bias_corrected_commit = manual_commit * bias_index
    adjusted_forecast = (ml_baseline * (1 - bias_weight)) + (
        bias_corrected_commit * bias_weight
    )
    hidden_upside = bias_corrected_commit - manual_commit

    # Compute confidence interval using bias standard deviation
    z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(
        confidence_level, 1.96
    )
    margin = manual_commit * bias_std * z_score * bias_weight
    ci_lower = adjusted_forecast - margin
    ci_upper = adjusted_forecast + margin

    return ReconciliationResult(
        manager_id=manager_id,
        manual_commit=manual_commit,
        bias_index=bias_index,
        archetype=archetype,
        bias_corrected_commit=bias_corrected_commit,
        ml_baseline=ml_baseline,
        adjusted_forecast=adjusted_forecast,
        hidden_upside=hidden_upside,
        confidence_interval=(ci_lower, ci_upper),
    )
