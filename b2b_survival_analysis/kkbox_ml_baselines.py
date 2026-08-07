"""
kkbox_ml_baselines.py — ML baseline suite (RUN ON YOUR OWN MACHINE)
====================================================================
Benchmarks nonlinear ML models against the paper's logistic hazard
models on the 30-day person-period panel, with the same strict temporal
split (train: period starts < 2016-07-01) and the same nested feature
ablation:

  FS0  boundary-blind   (tenure, plan, auto-renew, txn history, member)
  FS1  + days-to-boundary features   (the renewal cliff)
  FS2  + engagement features         (dynamic telemetry)
  FS3  logistic only: FS2 + engagement x near-boundary interactions
       (tests "does engagement matter more at the boundary?")

Models: LogisticRegression, HistGradientBoosting (sklearn), XGBoost
(optional — used if importable). Metrics on the temporal test set:
AUC, PR-AUC, log-loss, Brier, near-boundary (d<32) AUC, and a
10-bin reliability table.

Setup (once):   pip install scikit-learn xgboost
Run:            python3 kkbox_ml_baselines.py            (full panel,
                ~10-30 min, ~6 GB RAM)
                python3 kkbox_ml_baselines.py --sample 3 (quick pass on
                a 1/3 user subsample, matches the sandbox numbers)

Output: ml_results.json (git-tracked) — feeds the paper's Table 3.
"""
import argparse
import json
import os
import time
from datetime import date

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "data_cache")

SPLIT_DAY = (date(2016, 7, 1) - date(1970, 1, 1)).days
MONTH_BASE = (2015 - 1970) * 12
N_MONTHS = 27

DTB_BINS = [(-10**6, 0), (0, 8), (8, 16), (16, 24), (24, 32), (32, 93)]
TEN_BINS = [(1, 2), (2, 3), (3, 6), (6, 12), (12, 10**6)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def month_idx(days):
    m = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return (m - MONTH_BASE).astype(np.int32)


def load_panel(sample_mod):
    cols = ["uid", "t0", "days_to_boundary", "plan_days", "auto_renew",
            "period", "n_prior_trans", "event"]
    parts = {c: [] for c in cols}
    i = 0
    while os.path.exists(os.path.join(CACHE, f"panel_part{i}.npz")):
        z = np.load(os.path.join(CACHE, f"panel_part{i}.npz"))
        keep = (z["uid"] % sample_mod) == 0 if sample_mod > 1 \
            else np.ones(len(z["uid"]), bool)
        for c in cols:
            parts[c].append(z[c][keep])
        i += 1
    return {c: np.concatenate(v) for c, v in parts.items()}


def build_features(p):
    n = len(p["uid"])
    names, cols = [], []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    # ---- FS0: boundary-blind -------------------------------------------
    per = p["period"]
    for lo, hi in TEN_BINS:
        add(f"per_{lo}_{hi}", (per >= lo) & (per < hi))
    pd_ = p["plan_days"]
    add("plan_7d", pd_ < 8)
    add("plan_long", pd_ >= 35)
    add("auto_renew", p["auto_renew"])
    add("log_n_prior", np.log1p(p["n_prior_trans"]))

    mem = np.load(os.path.join(CACHE, "members.npz"))
    u = p["uid"]
    age = mem["bd"][u].astype(np.float32)
    add("age", np.where(age > 0, age, 0) / 40.0)
    add("age_missing", age < 0)
    g = mem["gender"][u]
    add("gender_m", g == 1)
    add("gender_f", g == 2)
    rv = mem["registered_via"][u]
    for code in (7, 9, 3, 4):
        add(f"regvia_{code}", rv == code)
    rd = mem["reg_date"][u].astype(np.float32)
    acct_age = np.where(rd > 0, (p["t0"] - rd) / 365.0, -1)
    add("acct_age_yrs", np.clip(acct_age, 0, 15))
    add("acct_age_missing", acct_age < 0)
    fs0 = len(names)

    # ---- FS1: + boundary distance ---------------------------------------
    dtb = p["days_to_boundary"]
    for lo, hi in DTB_BINS:
        add(f"dtb_{lo}_{hi}", (dtb >= lo) & (dtb < hi))
    add("dtb_linear", np.clip(dtb, 0, 92) / 92.0)
    fs1 = len(names)

    # ---- FS2: + engagement ------------------------------------------------
    eng = np.load(os.path.join(CACHE, "eng_compact.npz"))
    ad, hrs, skp = eng["active_days"], eng["hrs"], eng["skip_share"]
    has_logs = (ad.sum(axis=1, dtype=np.int32) > 0)
    m1 = month_idx(p["t0"]) - 1
    valid = (m1 >= 3) & (m1 < N_MONTHS) & has_logs[u]
    uu, mv = u[valid], m1[valid]

    def eng_col(source, offset=0, scale=1.0):
        out = np.zeros(n, dtype=np.float32)
        out[valid] = source[uu, mv - offset].astype(np.float32) / scale
        return out

    f_ad1 = eng_col(ad, 0, 31.0)
    f_ad2 = eng_col(ad, 1, 31.0)
    f_ad4 = eng_col(ad, 3, 31.0)
    add("eng_ad_m1", f_ad1)
    add("eng_ad_m2", f_ad2)
    add("eng_trend3", f_ad1 - f_ad4)
    add("eng_hrs_m1", np.clip(eng_col(hrs, 0, 100.0), 0, 10))
    add("eng_skip_m1", eng_col(skp, 0, 1.0))
    zero1 = np.zeros(n, dtype=np.float32)
    zero1[valid] = (ad[uu, mv] == 0)
    add("eng_zero_m1", zero1)
    add("eng_missing", ~valid)
    fs2 = len(names)

    # ---- FS3 (logistic only): engagement x near-boundary -----------------
    near = ((dtb >= 0) & (dtb < 32)).astype(np.float32)
    add("nearXad_m1", near * f_ad1)
    add("nearXzero_m1", near * zero1)
    add("nearXtrend3", near * (f_ad1 - f_ad4))
    add("nearXmissing", near * (~valid))
    fs3 = len(names)

    X = np.column_stack(cols)
    return X, names, (fs0, fs1, fs2, fs3)


def reliability(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": b, "n": int(m.sum()),
                    "mean_pred": round(float(prob[m].mean()), 5),
                    "mean_obs": round(float(y[m].mean()), 5)})
    return out


def evaluate(y, prob, near):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 log_loss, brier_score_loss)
    return {
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "pr_auc": round(float(average_precision_score(y, prob)), 4),
        "logloss": round(float(log_loss(y, prob)), 5),
        "brier": round(float(brier_score_loss(y, prob)), 5),
        "auc_near_boundary": round(float(roc_auc_score(y[near],
                                                       prob[near])), 4),
        "reliability": reliability(y, prob),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1,
                    help="keep uid %% SAMPLE == 0 (1 = full panel)")
    args = ap.parse_args()

    log(f"loading panel (sample mod {args.sample}) ...")
    p = load_panel(args.sample)
    log(f"{len(p['uid']):,} person-periods")
    X, names, (fs0, fs1, fs2, fs3) = build_features(p)
    y = p["event"].astype(np.int8)
    train = p["t0"] < SPLIT_DAY
    test = ~train
    dtb_te = p["days_to_boundary"][test]
    near = (dtb_te >= 0) & (dtb_te < 32)
    log(f"train {train.sum():,} | test {test.sum():,} | "
        f"features {X.shape[1]}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except Exception as e:                      # ImportError or XGBoostError
        has_xgb = False
        log(f"xgboost unavailable — skipping ({type(e).__name__}; on macOS "
            f"try `brew install libomp`)")

    R = {"n_train": int(train.sum()), "n_test": int(test.sum()),
         "sample_mod": args.sample, "split_date": "2016-07-01",
         "feature_sets": {"FS0": fs0, "FS1": fs1, "FS2": fs2, "FS3": fs3},
         "models": {}}

    runs = []
    for fs_name, k in [("FS0", fs0), ("FS1", fs1), ("FS2", fs2)]:
        runs.append((f"logit_{fs_name}",
                     LogisticRegression(max_iter=200, C=100.0,
                                        solver="lbfgs"), k))
        runs.append((f"hgb_{fs_name}",
                     HistGradientBoostingClassifier(
                         max_iter=300, learning_rate=0.1,
                         max_leaf_nodes=63, early_stopping=True,
                         validation_fraction=0.05, random_state=0), k))
        if has_xgb:
            runs.append((f"xgb_{fs_name}",
                         XGBClassifier(n_estimators=400, max_depth=7,
                                       learning_rate=0.1, subsample=0.8,
                                       colsample_bytree=0.8, n_jobs=-1,
                                       eval_metric="logloss",
                                       tree_method="hist"), k))
    runs.append(("logit_FS3_interactions",
                 LogisticRegression(max_iter=200, C=100.0,
                                    solver="lbfgs"), fs3))

    for label, model, k in runs:
        t0 = time.time()
        model.fit(X[train][:, :k], y[train])
        prob = model.predict_proba(X[test][:, :k])[:, 1]
        R["models"][label] = evaluate(y[test], prob, near)
        R["models"][label]["fit_minutes"] = round((time.time() - t0) / 60, 2)
        log(f"{label}: auc={R['models'][label]['auc']} "
            f"near={R['models'][label]['auc_near_boundary']} "
            f"({R['models'][label]['fit_minutes']} min)")
        if label == "logit_FS3_interactions":
            R["fs3_coefficients"] = {
                names[j]: round(float(model.coef_[0][j]), 4)
                for j in range(fs2, fs3)}

    out = os.path.join(BASE_DIR, "ml_results.json")
    json.dump(R, open(out, "w"), indent=2)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
