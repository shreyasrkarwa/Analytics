"""
Expanded Analysis Pipeline — Federal IT Contract Longevity Study
================================================================
Adds three peer-review-grade extensions to the original Random Forest study
and fills in the validation table for the IEEE Access submission:

  1. Survival analysis      — Kaplan-Meier curves + log-rank tests, and a
                              Cox proportional-hazards model (hazard ratios).
  2. Market concentration   — Herfindahl-Hirschman Index (HHI) by agency,
                              overall, and over fiscal years.
  3. Out-of-time validation — train on FY2018-2022, test on FY2023-2024.
  PLUS the full Random Forest validation metrics (Table II) at the 5-year
  threshold across the Top-10 / Top-25 / Top-50 vendor-tier definitions.

Run on your machine (where scikit-learn / lifelines are installed):

    cd usaspending_analysis
    pip install scikit-learn pandas numpy matplotlib lifelines
    python expanded_analysis.py

All numeric results are written to  outputs/expanded/expanded_results.json
Figures are written to               outputs/expanded/*.png
Paste expanded_results.json back to Claude to auto-fill the manuscript.
"""

import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.inspection import permutation_importance
from scipy import stats

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False
    print("WARNING: lifelines not installed — Cox/KM will be skipped. "
          "Run: pip install lifelines")

OUT = "outputs/expanded"
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42
N_EST, MAX_DEPTH, CV = 100, 5, 10
THRESHOLDS = {"1yr": 365, "3yr": 1095, "5yr": 1825, "10yr": 3650}
TIERS = (10, 25, 50)

# ── Load & prepare ────────────────────────────────────────────────────────────
df = pd.read_csv("data/longitudinal_it_contracts_fy18_fy24.csv")
df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
df["End Date"]   = pd.to_datetime(df["End Date"], errors="coerce")
df["dur"] = (df["End Date"] - df["Start Date"]).dt.days
df["Award Amount"] = pd.to_numeric(df["Award Amount"], errors="coerce")
df = df.dropna(subset=["dur", "Award Amount"])
df = df[df["dur"] >= 0].copy()
df["fy"] = df["Start Date"].dt.year
df["log_award"] = np.log1p(df["Award Amount"])

vs = df.groupby("Recipient Name")["Award Amount"].sum().sort_values(ascending=False)
for n in TIERS:
    df[f"Top{n}"] = df["Recipient Name"].isin(set(vs.head(n).index)).astype(int)

def is_dod(a):
    u = str(a).upper()
    return int(any(k in u for k in ["DEFENSE", "ARMY", "NAVY", "AIR FORCE"]))
df["Is_DoD"] = df["Awarding Agency"].apply(is_dod)

results = {"n_contracts": int(len(df)),
           "n_vendors": int(df["Recipient Name"].nunique()),
           "n_agencies": int(df["Awarding Agency"].nunique())}

# ── 1. Random Forest validation (Table II) ──────────────────────────────────────
def rf_metrics(X, y):
    if len(np.unique(y)) < 2:
        return None
    rf = RandomForestClassifier(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                class_weight="balanced", random_state=RANDOM_STATE)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=RANDOM_STATE)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte); proba = rf.predict_proba(Xte)[:, 1]
    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
    cv_pred = cross_val_predict(rf, X, y, cv=skf)
    cv_proba = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    gini = dict(zip(["log_award", "corporate_scale", "Is_DoD"],
                    [round(float(v), 4) for v in rf.feature_importances_]))
    return {
        "holdout_accuracy": round(accuracy_score(yte, pred), 4),
        "holdout_precision": round(precision_score(yte, pred, zero_division=0), 4),
        "holdout_recall": round(recall_score(yte, pred, zero_division=0), 4),
        "holdout_f1": round(f1_score(yte, pred, zero_division=0), 4),
        "holdout_roc_auc": round(roc_auc_score(yte, proba), 4),
        "cv_accuracy": round(accuracy_score(y, cv_pred), 4),
        "cv_f1": round(f1_score(y, cv_pred, zero_division=0), 4),
        "cv_roc_auc": round(roc_auc_score(y, cv_proba), 4),
        "gini_importance": gini,
        "positive_rate": round(float(np.mean(y)), 4),
    }

table2 = {}
for n in TIERS:
    feats = ["log_award", f"Top{n}", "Is_DoD"]
    y = (df["dur"] > THRESHOLDS["5yr"]).astype(int).values
    table2[f"Top{n}"] = rf_metrics(df[feats].values, y)
results["random_forest_5yr"] = table2

# temporal sweep (Top-10)
sweep = {}
for name, thr in THRESHOLDS.items():
    y = (df["dur"] > thr).astype(int).values
    sweep[name] = rf_metrics(df[["log_award", "Top10", "Is_DoD"]].values, y)
results["temporal_sweep_top10"] = sweep

# permutation importance (Top-10, 5yr)
y = (df["dur"] > THRESHOLDS["5yr"]).astype(int).values
X = df[["log_award", "Top10", "Is_DoD"]].values
rf = RandomForestClassifier(n_estimators=N_EST, max_depth=MAX_DEPTH,
                            class_weight="balanced", random_state=RANDOM_STATE).fit(X, y)
pi = permutation_importance(rf, X, y, n_repeats=30, random_state=RANDOM_STATE)
results["permutation_importance_top10_5yr"] = dict(zip(
    ["log_award", "corporate_scale", "Is_DoD"],
    [{"mean": round(float(m), 4), "std": round(float(s), 4)}
     for m, s in zip(pi.importances_mean, pi.importances_std)]))

# ── 2. Survival analysis ────────────────────────────────────────────────────────
if HAVE_LIFELINES:
    surv = {}
    for n in TIERS:
        inc = df[df[f"Top{n}"] == 1]["dur"]; oth = df[df[f"Top{n}"] == 0]["dur"]
        lr = logrank_test(inc, oth)
        surv[f"Top{n}"] = {"median_incumbent": float(inc.median()),
                           "median_other": float(oth.median()),
                           "logrank_p": round(float(lr.p_value), 4)}
    results["kaplan_meier"] = surv

    cox_df = df[["dur", "log_award", "Top10", "Is_DoD"]].copy()
    cox_df["event"] = 1  # period-of-performance length (all observed)
    cph = CoxPHFitter().fit(cox_df, duration_col="dur", event_col="event")
    results["cox_model"] = {
        "concordance": round(float(cph.concordance_index_), 4),
        "hazard_ratios": {k: round(float(v), 4) for k, v in cph.hazard_ratios_.items()},
        "p_values": {k: round(float(v), 4) for k, v in cph.summary["p"].items()},
    }

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 4.2))
    for n, c in zip(TIERS, ["#3b0f4f", "#b5316b", "#e08a5b"]):
        kmf.fit(df[df[f"Top{n}"] == 1]["dur"] / 365, label=f"Top-{n} incumbents")
        kmf.plot_survival_function(ci_show=False, color=c)
    kmf.fit(df["dur"] / 365, label="All contracts")
    kmf.plot_survival_function(ci_show=False, color="gray", linestyle="--")
    plt.xlabel("Contract period of performance (years)"); plt.ylabel("S(t)")
    plt.title("Kaplan-Meier: Contract Length by Vendor Tier")
    plt.tight_layout(); plt.savefig(f"{OUT}/km_survival.png", dpi=150); plt.close()

# Mann-Whitney U (Top-10 vs rest, duration)
U, p = stats.mannwhitneyu(df[df.Top10 == 1]["dur"], df[df.Top10 == 0]["dur"],
                          alternative="two-sided")
results["mann_whitney_top10_duration"] = {"U": float(U), "p": round(float(p), 4)}

# ── 3. HHI market concentration ─────────────────────────────────────────────────
def hhi(amounts):
    tot = amounts.sum()
    return float(((amounts / tot * 100) ** 2).sum()) if tot > 0 else float("nan")
by_agency = {ag: round(hhi(sub.groupby("Recipient Name")["Award Amount"].sum()), 1)
             for ag, sub in df.groupby("Awarding Agency") if len(sub) >= 20}
by_year = {int(y): round(hhi(s.groupby("Recipient Name")["Award Amount"].sum()), 1)
           for y, s in df[df.fy.between(2018, 2024)].groupby("fy")}
results["hhi"] = {"overall": round(hhi(df.groupby("Recipient Name")["Award Amount"].sum()), 1),
                  "by_agency": dict(sorted(by_agency.items(), key=lambda x: -x[1])),
                  "by_fiscal_year": by_year}

# ── 4. Out-of-time validation (train FY18-22, test FY23-24) ─────────────────────
tr = df[df.fy.between(2018, 2022)]; te = df[df.fy.between(2023, 2024)]
oot = {}
if len(tr) > 50 and len(te) > 20:
    for n in TIERS:
        feats = ["log_award", f"Top{n}", "Is_DoD"]
        ytr = (tr["dur"] > THRESHOLDS["5yr"]).astype(int).values
        yte = (te["dur"] > THRESHOLDS["5yr"]).astype(int).values
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=N_EST, max_depth=MAX_DEPTH,
                                    class_weight="balanced", random_state=RANDOM_STATE)
        rf.fit(tr[feats].values, ytr)
        pred = rf.predict(te[feats].values); proba = rf.predict_proba(te[feats].values)[:, 1]
        oot[f"Top{n}"] = {"test_accuracy": round(accuracy_score(yte, pred), 4),
                          "test_f1": round(f1_score(yte, pred, zero_division=0), 4),
                          "test_roc_auc": round(roc_auc_score(yte, proba), 4),
                          "n_train": int(len(tr)), "n_test": int(len(te))}
results["out_of_time_validation"] = oot

json.dump(results, open(f"{OUT}/expanded_results.json", "w"), indent=2)
print("=" * 70)
print("DONE. Results -> outputs/expanded/expanded_results.json")
print("Figures -> outputs/expanded/*.png")
print("=" * 70)
print(json.dumps(results, indent=2)[:2000])
