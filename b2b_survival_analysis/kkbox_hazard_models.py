"""
kkbox_hazard_models.py — discrete-time hazard models, boundary-blind vs
boundary-aware vs +engagement (pure NumPy — no sklearn required)
========================================================================
Fits logistic discrete-time hazard models on the 30-day person-period
panel with a strict temporal split (train: periods starting before
2016-07-01; test: after). Nested specification tests the paper's two
core claims:

  M0  boundary-blind: tenure, plan, auto-renew, transaction history
  M1  + days-to-boundary bins        (the renewal cliff)
  M2  + monthly engagement features  (dynamic prediction)

Reports test AUC, log-loss, and the fitted cliff shape (dtb-bin odds
ratios). Appends results to kkbox_results.json.

Run: python3 kkbox_hazard_models.py            (~40 s, ~2.5 GB RAM)
"""
import json
import os
from datetime import date

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "data_cache")

SAMPLE_MOD = 3                # keep uid % 3 == 0  (~8.7M rows)
SPLIT_DAY = (date(2016, 7, 1) - date(1970, 1, 1)).days
MONTH_BASE = (2015 - 1970) * 12
N_MONTHS = 27
RIDGE = 1e-4

DTB_BINS = [(-10**6, 0, "past_due"), (0, 8, "dtb_0_7"), (8, 16, "dtb_8_15"),
            (16, 24, "dtb_16_23"), (24, 32, "dtb_24_31"),
            (32, 93, "dtb_32_92")]          # ref: > 92 days
TEN_BINS = [(1, 2, "per_1"), (2, 3, "per_2"), (3, 6, "per_3_5"),
            (6, 12, "per_6_11"), (12, 10**6, "per_12p")]  # ref: period 0
PLAN_BINS = [(0, 8, "plan_7d"), (35, 10**6, "plan_long")]  # ref: 30d


def load_panel():
    cols = ["uid", "t0", "days_to_boundary", "plan_days", "auto_renew",
            "period", "n_prior_trans", "event"]
    parts = {c: [] for c in cols}
    i = 0
    while os.path.exists(os.path.join(CACHE, f"panel_part{i}.npz")):
        z = np.load(os.path.join(CACHE, f"panel_part{i}.npz"))
        keep = (z["uid"] % SAMPLE_MOD) == 0
        for c in cols:
            parts[c].append(z[c][keep])
        i += 1
    return {c: np.concatenate(v) for c, v in parts.items()}


def month_idx(days):
    m = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return (m - MONTH_BASE).astype(np.int32)


def build_features(p):
    n = len(p["uid"])
    names, cols = ["intercept"], [np.ones(n, dtype=np.float32)]

    def add(name, arr):
        names.append(name)
        cols.append(arr.astype(np.float32))

    per = p["period"]
    for lo, hi, lab in TEN_BINS:
        add(lab, ((per >= lo) & (per < hi)))
    pd_ = p["plan_days"]
    for lo, hi, lab in PLAN_BINS:
        add(lab, ((pd_ >= lo) & (pd_ < hi)))
    add("auto_renew", p["auto_renew"])
    add("log_n_prior", np.log1p(p["n_prior_trans"]))
    m0_k = len(names)

    dtb = p["days_to_boundary"]
    for lo, hi, lab in DTB_BINS:
        add(lab, ((dtb >= lo) & (dtb < hi)))
    m1_k = len(names)

    # engagement (previous calendar month)
    eng = np.load(os.path.join(CACHE, "eng_compact.npz"))
    ad, skp = eng["active_days"], eng["skip_share"]
    has_logs = (ad.sum(axis=1, dtype=np.int32) > 0)
    m1 = month_idx(p["t0"]) - 1
    valid = (m1 >= 3) & (m1 < N_MONTHS) & has_logs[p["uid"]]
    u, mv = p["uid"][valid], m1[valid]

    f_ad = np.zeros(n, dtype=np.float32)
    f_tr = np.zeros(n, dtype=np.float32)
    f_sk = np.zeros(n, dtype=np.float32)
    f_ad[valid] = ad[u, mv].astype(np.float32) / 31.0
    f_tr[valid] = (ad[u, mv].astype(np.float32)
                   - ad[u, mv - 3].astype(np.float32)) / 31.0
    f_sk[valid] = skp[u, mv].astype(np.float32)
    zero_act = np.zeros(n, dtype=np.float32)
    zero_act[valid] = (ad[u, mv] == 0)
    add("eng_active_days", f_ad)
    add("eng_trend3", f_tr)
    add("eng_skip_share", f_sk)
    add("eng_zero_month", zero_act)
    add("eng_missing", (~valid))
    m2_k = len(names)

    X = np.column_stack(cols)
    return X, names, (m0_k, m1_k, m2_k)


def fit_logistic(X, y, ridge=RIDGE, iters=30, tol=1e-8):
    k = X.shape[1]
    beta = np.zeros(k)
    beta[0] = np.log(max(y.mean(), 1e-6) / (1 - y.mean()))
    for it in range(iters):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = pr * (1 - pr) + 1e-12
        g = X.T @ (y - pr) - ridge * beta
        H = np.zeros((k, k))
        for lo in range(0, len(y), 2_000_000):     # chunked X'WX
            Xb = X[lo:lo + 2_000_000]
            wb = w[lo:lo + 2_000_000]
            H += (Xb * wb[:, None]).T @ Xb
        H += ridge * np.eye(k)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta, it + 1


def auc_rank(y, s):
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    s_sorted = s[order]
    ties = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    grp = np.cumsum(ties) - 1
    cnt = np.bincount(grp)
    csum = np.bincount(grp, weights=r[order])
    mid = (csum / cnt)[grp]
    r[order] = mid
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    p = load_panel()
    print(f"panel sample: {len(p['uid']):,} person-periods", flush=True)
    X, names, (k0, k1, k2) = build_features(p)
    y = p["event"].astype(np.float64)
    train = p["t0"] < SPLIT_DAY
    test = ~train
    print(f"train {train.sum():,} | test {test.sum():,} | "
          f"test event rate {y[test].mean():.4f}", flush=True)

    R = {"n_train": int(train.sum()), "n_test": int(test.sum()),
         "test_event_rate": round(float(y[test].mean()), 5),
         "split_date": "2016-07-01", "sample": f"uid%{SAMPLE_MOD}==0",
         "models": {}}
    Xtr, ytr = X[train], y[train]
    Xte, yte = X[test], y[test]
    dtb_te = p["days_to_boundary"][test]
    near = (dtb_te >= 0) & (dtb_te < 32)      # decision-imminent periods
    R["n_test_near_boundary"] = int(near.sum())
    R["near_event_rate"] = round(float(yte[near].mean()), 5)
    betas = {}
    for label, k in [("M0_boundary_blind", k0),
                     ("M1_plus_boundary", k1),
                     ("M2_plus_engagement", k2)]:
        beta, ni = fit_logistic(Xtr[:, :k], ytr)
        pte = 1 / (1 + np.exp(-(Xte[:, :k] @ beta)))
        R["models"][label] = {
            "auc": round(auc_rank(yte, pte), 4),
            "logloss": round(logloss(yte, pte), 5),
            "auc_near_boundary": round(auc_rank(yte[near], pte[near]), 4),
            "logloss_near_boundary": round(logloss(yte[near], pte[near]), 5),
            "n_features": k, "newton_iters": ni}
        betas[label] = beta
        print(label, R["models"][label], flush=True)

    # fitted cliff shape from M2 (odds ratios vs >92d reference)
    b2 = betas["M2_plus_engagement"]
    R["cliff_odds_ratios_M2"] = {
        names[j]: round(float(np.exp(b2[j])), 3)
        for j in range(k0, k1)}
    R["engagement_coefs_M2"] = {
        names[j]: round(float(b2[j]), 4) for j in range(k1, k2)}

    res_path = os.path.join(BASE_DIR, "kkbox_results.json")
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    allres["hazard_models"] = R
    json.dump(allres, open(res_path, "w"), indent=2)
    print(json.dumps({k: v for k, v in R.items()
                      if k in ("cliff_odds_ratios_M2",
                               "engagement_coefs_M2")}, indent=2))


if __name__ == "__main__":
    main()
