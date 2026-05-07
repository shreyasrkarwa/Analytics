"""
Evaluation Metrics for B2B Survival Analysis
=============================================
Provides a comprehensive suite of survival model evaluation metrics:

  1. Concordance Index (Harrell's C) — discrimination measure
  2. Uno's C-Index — censoring-weighted concordance (more robust)
  3. Brier Score — calibration at a fixed time horizon
  4. Integrated Brier Score (IBS) — calibration across all time points
  5. Time-Dependent AUC — discrimination at each time horizon

References
----------
Harrell et al. (1982); Uno et al. (2011); Graf et al. (1999).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    brier_score,
    cumulative_dynamic_auc,
)
from sksurv.util import Surv


# ---------------------------------------------------------------------------
# C-INDEX (HARRELL'S)
# ---------------------------------------------------------------------------

def compute_c_index(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    risk_score: np.ndarray,
) -> Dict:
    """
    Harrell's Concordance Index for evaluating survival model discrimination.

    A value of 0.5 is random; 1.0 is perfect discrimination.
    """
    event_indicator = np.asarray(event_indicator).astype(bool)
    event_time = np.asarray(event_time)
    c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
        event_indicator, event_time, risk_score
    )
    return {
        'C-Index (Harrell)': round(c_index, 4),
        'Concordant Pairs': concordant,
        'Discordant Pairs': discordant,
    }


# ---------------------------------------------------------------------------
# UNO'S C-INDEX (CENSORING-WEIGHTED)
# ---------------------------------------------------------------------------

def compute_uno_c(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_score: np.ndarray,
    tau: float = None,
) -> Dict:
    """
    Uno's censoring-weighted C-Index.

    Unlike Harrell's C, Uno's C re-weights pairs by the inverse probability
    of censoring, making it more robust when censoring is informative.

    Parameters
    ----------
    y_train : structured array (event, time) from training set
    y_test  : structured array (event, time) from test set
    risk_score : predicted risk scores (higher = higher risk)
    tau : upper time limit for pairs (defaults to 80th percentile of event times)
    """
    if tau is None:
        event_times = y_test['time_to_event'][y_test['event_observed']]
        tau = np.percentile(event_times, 80) if len(event_times) > 0 else None

    result = concordance_index_ipcw(y_train, y_test, risk_score, tau=tau)
    return {
        'C-Index (Uno)': round(result.concordance, 4),
        'Tau (time horizon)': round(tau, 1) if tau else None,
    }


# ---------------------------------------------------------------------------
# BRIER SCORE
# ---------------------------------------------------------------------------

def compute_brier_score(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_functions: List,
    times: np.ndarray,
) -> Dict:
    """
    Brier Score at specified time points.

    The Brier Score measures calibration: how well predicted survival
    probabilities match observed outcomes. Lower is better (0 = perfect).
    An uninformative model scores ~0.25.

    Parameters
    ----------
    y_train : structured array from training set
    y_test  : structured array from test set
    survival_functions : list of StepFunction objects (one per test sample)
    times : array of time points to evaluate
    """
    # Evaluate survival probability at each time point for each subject
    preds = np.row_stack([fn(times) for fn in survival_functions])

    times_eval, scores = brier_score(y_train, y_test, preds, times)

    # Integrated Brier Score (trapezoidal rule)
    ibs = float(np.trapz(scores, times_eval) / (times_eval[-1] - times_eval[0]))

    return {
        'Brier Score (t=12)': round(scores[np.searchsorted(times_eval, 12)], 4) if 12 in times_eval else None,
        'Brier Score (t=24)': round(scores[np.searchsorted(times_eval, 24)], 4) if 24 in times_eval else None,
        'Integrated Brier Score': round(ibs, 4),
    }


# ---------------------------------------------------------------------------
# TIME-DEPENDENT AUC
# ---------------------------------------------------------------------------

def compute_time_dependent_auc(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores_at_times: np.ndarray,
    times: np.ndarray,
) -> Dict:
    """
    Cumulative/Dynamic Time-Dependent AUC.

    Measures discrimination at each time horizon: among all pairs (i, j)
    where i experienced the event before time t and j did not, what fraction
    correctly has a higher predicted risk?

    Parameters
    ----------
    y_train : structured array from training set
    y_test  : structured array from test set
    risk_scores_at_times : (n_samples, n_times) array of risk scores at each time
    times : evaluation time points
    """
    auc_values, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores_at_times, times)

    result = {'Mean AUC': round(float(mean_auc), 4)}
    for t, auc in zip(times, auc_values):
        result[f'AUC (t={int(t)})'] = round(float(auc), 4)

    return result


# ---------------------------------------------------------------------------
# STRUCTURED ARRAY HELPER
# ---------------------------------------------------------------------------

def make_structured_array(event_col: pd.Series, time_col: pd.Series) -> np.ndarray:
    """
    Converts two pandas Series to a sksurv structured array.
    """
    return Surv.from_arrays(
        event=event_col.astype(bool).values,
        time=time_col.values,
    )


# ---------------------------------------------------------------------------
# SUMMARY PRINTER
# ---------------------------------------------------------------------------

def print_metrics(model_name: str, metrics: Dict) -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k:<35} {v}")
