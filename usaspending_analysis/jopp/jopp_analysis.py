"""
JOPP Analysis Pipeline — Richer Feature Set
===========================================
Extends the study for Journal of Public Procurement submission with a
substantially richer feature set built from the existing dataset:

  - Award size (log)                     [original]
  - Vendor tier (Top-10/25/50)           [original]
  - DoD binary                           [original]
  - Awarding-agency category dummies     [NEW: top-6 agencies one-hot]
  - Fiscal-year cohort                   [NEW]
  - Contract-description keyword flags   [NEW: cloud, cyber, modernization,
                                          maintenance/support, software dev,
                                          data/hosting, professional services]

Outputs:
  outputs/jopp/jopp_results.json   — all numeric results
  outputs/jopp/*.png               — updated figures

Run:
    cd usaspending_analysis
    source venv/bin/activate   # if you use the venv
    pip install scikit-learn pandas numpy matplotlib lifelines
    python jopp_analysis.py
"""

import os, json, re
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
    from lifelines import CoxPHFitter
    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False

OUT = "outputs/jopp"
os.makedirs(OUT, exist_ok=True)
RS = 42
THRESH_5YR = 1825

# ── Load & engineer ───────────────────────────────────────────────────────────
df = pd.read_csv("data/longitudinal_it_contracts_fy18_fy24.csv")
df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
df["dur"] = (df["End Date"] - df["Start Date"]).dt.days
df["Award Amount"] = pd.to_numeric(df["Award Amount"], errors="coerce")
df = df.dropna(subset=["dur", "Award Amount"])
df = df[df["dur"] >= 0].copy()
df["log_award"] = np.log1p(df["Award Amount"])
df["fy"] = df["Start Date"].dt.year.clip(2000, 2026)

vs = df.groupby("Recipient Name")["Award Amount"].sum().sort_values(ascending=False)
for n in (10, 25, 50):
    df[f"Top{n}"] = df["Recipient Name"].isin(set(vs.head(n).index)).astype(int)

def is_dod(a):
    u = str(a).upper()
    return int(any(k in u for k in ["DEFENSE", "ARMY", "NAVY", "AIR FORCE"]))
df["Is_DoD"] = df["Awarding Agency"].apply(is_dod)

# NEW: agency dummies (top 6 by contract count, excluding DoD which has its own flag)
top_agencies = [a for a in df["Awarding Agency"].value_counts().index
                if "Defense" not in a][:6]
agency_cols = []
for a in top_agencies:
    col = "Ag_" + re.sub(r"\W+", "_", a)[:28]
    df[col] = (df["Awarding Agency"] == a).astype(int)
    agency_cols.append(col)

# NEW: description keyword flags
desc = df["Description"].fillna("").str.upper()
kw = {
    "KW_cloud": r"CLOUD|IAAS|PAAS|SAAS|HOSTING",
    "KW_cyber": r"CYBER|SECURITY|INFOSEC|ZERO TRUST",
    "KW_modernization": r"MODERNIZ|TRANSFORM|MIGRAT",
    "KW_maintenance": r"MAINTEN|SUPPORT|SUSTAIN|OPERAT",
    "KW_softwaredev": r"SOFTWARE|DEVELOP|AGILE|DEVSECOPS|APPLICATION",
    "KW_data": r"DATA|ANALYTIC|WAREHOUS|INTELLIGENCE",
    "KW_profservices": r"PROFESSIONAL|ADVISORY|CONSULT|STAFF",
}
kw_cols = []
for col, pat in kw.items():
    df[col] = desc.str.contains(pat, regex=True).astype(int)
    kw_cols.append(col)

results = {"n": int(len(df)),
           "keyword_prevalence": {c: round(float(df[c].mean()), 3) for c in kw_cols}}

# ── Model comparison: 3-feature vs rich-feature ──────────────────────────────
y = (df["dur"] > THRESH_5YR).astype(int).values
feature_sets = {
    "baseline_3feat": ["log_award", "Top10", "Is_DoD"],
    "rich": ["log_award", "Top10", "Is_DoD", "fy"] + agency_cols + kw_cols,
}
model_out = {}
for name, feats in feature_sets.items():
    X = df[feats].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=None if name == "rich" else 5,
                                min_samples_leaf=5, class_weight="balanced",
                                random_state=RS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RS)
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]; pred = rf.predict(Xte)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RS)
    cvp = cross_val_predict(rf, X, y, cv=skf, method="predict_proba")[:, 1]
    gini = sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])
    pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=RS)
    perm = sorted(zip(feats, pi.importances_mean), key=lambda t: -t[1])
    model_out[name] = {
        "holdout_accuracy": round(accuracy_score(yte, pred), 4),
        "holdout_f1": round(f1_score(yte, pred), 4),
        "holdout_roc_auc": round(roc_auc_score(yte, proba), 4),
        "cv_roc_auc": round(roc_auc_score(y, cvp), 4),
        "gini_top10": [(f, round(float(v), 4)) for f, v in gini[:10]],
        "perm_top10": [(f, round(float(v), 4)) for f, v in perm[:10]],
        "corporate_scale_gini": round(float(dict(gini).get("Top10", 0)), 4),
        "corporate_scale_perm": round(float(dict(perm).get("Top10", 0)), 4),
    }
results["models_5yr"] = model_out

# ── Rich Cox model ───────────────────────────────────────────────────────────
if HAVE_LIFELINES:
    cox_feats = ["dur", "log_award", "Top10", "Is_DoD", "fy"] + kw_cols
    cd = df[cox_feats].copy(); cd["event"] = 1
    cph = CoxPHFitter(penalizer=0.01).fit(cd, duration_col="dur", event_col="event")
    results["cox_rich"] = {
        "concordance": round(float(cph.concordance_index_), 4),
        "hazard_ratios": {k: round(float(v), 4) for k, v in cph.hazard_ratios_.items()},
        "p_values": {k: round(float(v), 4) for k, v in cph.summary["p"].items()},
    }

# ── Duration by keyword category (descriptive) ───────────────────────────────
results["median_duration_by_keyword"] = {
    c: {"with": float(df[df[c] == 1]["dur"].median() or 0),
        "without": float(df[df[c] == 0]["dur"].median() or 0),
        "n_with": int(df[c].sum())}
    for c in kw_cols}

# ── Figure: model comparison feature importance ──────────────────────────────
rich = model_out["rich"]
labels = [f for f, v in rich["gini_top10"]][::-1]
vals = [v for f, v in rich["gini_top10"]][::-1]
pretty = {"log_award": "Award size (log)", "Top10": "Top-10 vendor", "Is_DoD": "DoD agency",
          "fy": "Fiscal-year cohort"}
labels = [pretty.get(l, l.replace("Ag_", "").replace("KW_", "kw: ").replace("_", " ")[:26])
          for l in labels]
plt.figure(figsize=(7, 4.5))
plt.barh(labels, vals, color="#3b6ba5")
plt.xlabel("Gini importance"); plt.title("Rich Model: Top Feature Importances (5-year survival)")
plt.tight_layout(); plt.savefig(f"{OUT}/rich_importances.png", dpi=150); plt.close()

json.dump(results, open(f"{OUT}/jopp_results.json", "w"), indent=2)
print(json.dumps(results, indent=2)[:3000])
print("\nDONE -> outputs/jopp/jopp_results.json")
