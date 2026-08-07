"""
telco_boundary_analysis.py — IBM Telco secondary dataset, done honestly
========================================================================
Supersedes validation_telco.py, which suffered target leakage
(TotalCharges ≈ tenure × MonthlyCharges encodes survival time; the
resulting C≈0.90 was inflated). This script:

  1. Fits a discrete-time logistic hazard on person-month expansion
     WITHOUT any tenure-derived billing covariates (honest baseline).
  2. Tests the renewal-cliff generalization: do churn hazards spike at
     contract-anniversary tenures (12/24/36/48/60 for one-year,
     24/48 for two-year contracts), with month-to-month customers as
     the structural comparison (their boundary recurs monthly)?
     Boundary-spike test: observed churn events at anniversary tenures
     vs expectation from neighbor-tenure hazards (exact Poisson tail).
  3. Adds boundary indicators to the hazard model; LR test (df=2,
     χ²(2) sf = exp(-LR/2)) and holdout AUC.

Cross-sectional caveat (reported with results): Telco is a snapshot —
tenure at churn/censoring, assumed-stationary hazards.

Outputs: telco_results.json, figures/fig11_telco_boundaries.png
Run: python3 telco_boundary_analysis.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys_path_note = None
import sys  # noqa: E402
sys.path.insert(0, BASE_DIR)
from kkbox_hazard_models import fit_logistic, auc_rank, logloss  # noqa: E402

CSV = os.path.join(BASE_DIR, "telco_churn.csv")


def poisson_tail(k, lam):
    """P(X >= k) for X ~ Poisson(lam), exact summation."""
    if k <= 0:
        return 1.0
    # sum pmf from 0..k-1
    logp = -lam
    cdf = math.exp(logp)
    for i in range(1, int(k)):
        logp += math.log(lam) - math.log(i)
        cdf += math.exp(logp)
    return max(0.0, 1.0 - cdf)


def load():
    df = pd.read_csv(CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df[df["tenure"] > 0].copy()
    df["event"] = (df["Churn"] == "Yes").astype(int)
    return df


def hazard_by_tenure(df, group_mask):
    """events_t, atrisk_t for t = 1..72 within a customer subset."""
    t = df.loc[group_mask, "tenure"].to_numpy()
    e = df.loc[group_mask, "event"].to_numpy()
    ev = np.zeros(73)
    n_end = np.zeros(73)
    for ti, ei in zip(t, e):
        ev[ti] += ei
        n_end[ti] += 1
    atrisk = np.cumsum(n_end[::-1])[::-1]      # customers with tenure >= t
    return ev[1:], atrisk[1:]                  # index 0 ↔ tenure 1


def spike_test(ev, atrisk, boundaries, halo=2):
    """Observed vs neighbor-expected churn events at boundary tenures."""
    obs = exp = 0.0
    details = []
    for b in boundaries:
        if b > 72 or atrisk[b - 1] < 30:
            continue
        nb = [b + d for d in range(-halo, halo + 1)
              if d != 0 and 1 <= b + d <= 72]
        nb_ev = sum(ev[t - 1] for t in nb)
        nb_ar = sum(atrisk[t - 1] for t in nb)
        rate = nb_ev / nb_ar if nb_ar else 0.0
        lam = rate * atrisk[b - 1]
        obs += ev[b - 1]
        exp += lam
        details.append({"tenure": b, "observed": int(ev[b - 1]),
                        "expected": round(lam, 1),
                        "at_risk": int(atrisk[b - 1])})
    p = poisson_tail(obs, exp) if exp > 0 else 1.0
    return {"observed_total": int(obs), "expected_total": round(exp, 1),
            "ratio": round(obs / exp, 2) if exp else None,
            "poisson_p_one_sided": round(p, 6), "by_boundary": details}


def person_months(df):
    """Expand to person-month rows with covariates + boundary flags."""
    n = df["tenure"].to_numpy()
    total = int(n.sum())
    cust = np.repeat(np.arange(len(df)), n)
    t = np.concatenate([np.arange(1, k + 1) for k in n]).astype(np.float64)
    last = np.concatenate([np.r_[np.zeros(k - 1), 1] for k in n])
    y = last * np.repeat(df["event"].to_numpy(), n)

    c1 = np.repeat((df["Contract"] == "One year").to_numpy(), n)
    c2 = np.repeat((df["Contract"] == "Two year").to_numpy(), n)
    at_b = (c1 & (t % 12 == 0)) | (c2 & (t % 24 == 0))
    near_b = (c1 & ((t % 12 == 0) | (t % 12 >= 11))) | \
             (c2 & ((t % 24 == 0) | (t % 24 >= 23)))

    def rep(col, val=None):
        v = (df[col] == val).to_numpy() if val else df[col].to_numpy()
        return np.repeat(v, n).astype(np.float64)

    cols = {
        "intercept": np.ones(total),
        "ten": t / 72, "ten2": (t / 72) ** 2, "log_ten": np.log(t),
        "contract_1yr": c1.astype(float), "contract_2yr": c2.astype(float),
        "monthly": rep("MonthlyCharges") / 100.0,
        "fiber": rep("InternetService", "Fiber optic"),
        "no_internet": rep("InternetService", "No"),
        "echeck": rep("PaymentMethod", "Electronic check"),
        "paperless": rep("PaperlessBilling", "Yes"),
        "senior": np.repeat(df["SeniorCitizen"].to_numpy(), n).astype(float),
        "partner": rep("Partner", "Yes"),
        "dependents": rep("Dependents", "Yes"),
        "techsupport": rep("TechSupport", "Yes"),
        "onlinesec": rep("OnlineSecurity", "Yes"),
    }
    names = list(cols)
    X0 = np.column_stack([cols[c] for c in names]).astype(np.float64)
    Xb = np.column_stack([at_b.astype(float), near_b.astype(float)])
    return X0, Xb, y, cust, names


def main():
    df = load()
    R = {"n_customers": int(len(df)),
         "churn_rate": round(float(df["event"].mean()), 4),
         "leakage_note": ("TotalCharges excluded (deterministic in tenure "
                          "x MonthlyCharges -> target leakage; prior "
                          "C~0.90 in results.json is deprecated)")}

    groups = {"one_year": (df["Contract"] == "One year",
                           [12, 24, 36, 48, 60]),
              "two_year": (df["Contract"] == "Two year", [24, 48]),
              "month_to_month_placebo": (df["Contract"] == "Month-to-month",
                                         [12, 24, 36, 48, 60])}
    haz = {}
    for gname, (mask, bnds) in groups.items():
        ev, ar = hazard_by_tenure(df, mask)
        haz[gname] = (ev, ar)
        R[f"spike_test_{gname}"] = spike_test(ev, ar, bnds)
        print(gname, json.dumps(R[f"spike_test_{gname}"], indent=1)[:400],
              flush=True)

    # discrete-time hazard models
    X0, Xb, y, cust, names = person_months(df)
    rng = np.random.default_rng(11)
    test_cust = rng.random(int(cust.max()) + 1) < 0.5
    te = test_cust[cust]
    tr = ~te
    X1 = np.column_stack([X0, Xb])
    res_models = {}
    lls = {}
    for label, X in [("M0_no_boundary", X0), ("M1_boundary", X1)]:
        beta, _ = fit_logistic(X[tr], y[tr])
        p_te = 1 / (1 + np.exp(-(X[te] @ beta)))
        # in-sample ll on train for LR test
        p_tr = np.clip(1 / (1 + np.exp(-(X[tr] @ beta))), 1e-12, 1 - 1e-12)
        lls[label] = float(np.sum(y[tr] * np.log(p_tr)
                                  + (1 - y[tr]) * np.log(1 - p_tr)))
        res_models[label] = {"auc": round(auc_rank(y[te], p_te), 4),
                             "logloss": round(logloss(y[te], p_te), 5)}
        if label == "M1_boundary":
            res_models[label]["coef_at_boundary"] = round(float(beta[-2]), 3)
            res_models[label]["coef_near_boundary"] = round(float(beta[-1]), 3)
            res_models[label]["OR_at_boundary"] = round(
                float(np.exp(beta[-2])), 2)
    lr = 2 * (lls["M1_boundary"] - lls["M0_no_boundary"])
    res_models["LR_test_boundary_terms"] = {
        "LR": round(lr, 1), "df": 2, "p": f"{math.exp(-lr / 2):.2e}"}
    R["hazard_models"] = res_models
    print(json.dumps(res_models, indent=1), flush=True)

    json.dump(R, open(os.path.join(BASE_DIR, "telco_results.json"), "w"),
              indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    fig.suptitle("Contract-Anniversary Churn Spikes in a Second Dataset "
                 "(IBM Telco, honest specification)", fontsize=12,
                 fontweight="bold")
    tt = np.arange(1, 73)
    for ax, (gname, bnds, c) in zip(axes, [
            ("month_to_month_placebo", [], "#7f8c8d"),
            ("one_year", [12, 24, 36, 48, 60], "#2c7fb8"),
            ("two_year", [24, 48], "#de2d26")]):
        ev, ar = haz[gname]
        h = np.divide(ev, ar, out=np.zeros(72), where=ar > 20)
        ax.plot(tt, h, "o-", ms=2.5, lw=0.8, color=c)
        for b in bnds:
            ax.axvline(b, color="crimson", ls=":", lw=1)
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Monthly churn hazard")
        st = R[f"spike_test_{gname}"]
        ax.set_title(f"{gname.replace('_', ' ')}\nobs/exp at boundaries: "
                     f"{st['ratio']}, p={st['poisson_p_one_sided']:.1e}"
                     if st["ratio"] else gname, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(os.path.join(BASE_DIR, "figures",
                             "fig11_telco_boundaries.png"), dpi=300)
    print("figure saved", flush=True)


if __name__ == "__main__":
    main()
