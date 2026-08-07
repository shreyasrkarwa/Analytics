"""
kkbox_engagement_features.py — engagement × renewal-boundary empirics
======================================================================
1. Compacts engagement_monthly.npz → eng_compact.npz (int8/float16,
   ~4x smaller in RAM, needed for the 3 GB sandbox).
2. Joins last-full-month engagement onto each renewal decision →
   data_cache/boundary_eng.npz.
3. Empirics for the paper's dynamic-prediction claim:
     (a) churn-by-engagement-decile at the boundary
     (b) event study: mean listening activity in the 6 months before a
         boundary, churned vs renewed (the "fading-out" signature)
     (c) engagement trend (3-month change) vs churn
   → appends to kkbox_results.json, draws figures/fig8_engagement_cliff.png

Run: python3 kkbox_engagement_features.py
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "data_cache")
FIG_DIR = os.path.join(BASE_DIR, "figures")

MONTH_BASE = (2015 - 1970) * 12          # month index of 2015-01
N_MONTHS = 27


def build_compact():
    out = os.path.join(CACHE, "eng_compact.npz")
    if os.path.exists(out):
        return
    z = np.load(os.path.join(CACHE, "engagement_monthly.npz"))
    ad = np.clip(z["active_days"], 0, 31).astype(np.int8)
    hrs = (z["total_secs"] / 3600.0).astype(np.float16)
    plays = z["plays"]
    skips = z["skips"]
    skip_share = (skips / np.maximum(plays + skips, 1)).astype(np.float16)
    np.savez(out, active_days=ad, hrs=hrs, skip_share=skip_share)
    print("eng_compact.npz written", flush=True)


def month_idx(days):
    """days since 1970 → month index relative to 2015-01."""
    m = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return (m - MONTH_BASE).astype(np.int32)


def main():
    build_compact()
    eng = np.load(os.path.join(CACHE, "eng_compact.npz"))
    ad, hrs, skip_share = eng["active_days"], eng["hrs"], eng["skip_share"]
    has_logs = (ad.sum(axis=1, dtype=np.int32) > 0)

    b = np.load(os.path.join(CACHE, "boundaries.npz"))
    uid, out = b["uid"], b["outcome"]
    bm = month_idx(b["boundary_date"])

    # last full month before the boundary month
    m1 = bm - 1
    ok = (m1 >= 3) & (m1 < N_MONTHS) & has_logs[uid]
    u, m1v = uid[ok], m1[ok]
    f_ad1 = ad[u, m1v].astype(np.float32)
    f_hrs1 = hrs[u, m1v].astype(np.float32)
    f_skip1 = skip_share[u, m1v].astype(np.float32)
    f_ad4 = ad[u, m1v - 3].astype(np.float32)
    f_trend = f_ad1 - f_ad4

    np.savez_compressed(
        os.path.join(CACHE, "boundary_eng.npz"),
        row_ok=ok, ad_m1=f_ad1.astype(np.float16),
        hrs_m1=f_hrs1.astype(np.float16),
        skip_m1=f_skip1.astype(np.float16),
        trend3=f_trend.astype(np.float16))

    dec = out[ok] != 2
    churn = (out[ok] == 1)[dec]
    R = {}

    # (a) churn by active-days bins, last full month
    x = f_ad1[dec]
    AD_BINS = [(0, 1, "0"), (1, 3, "1-2"), (3, 6, "3-5"), (6, 11, "6-10"),
               (11, 16, "11-15"), (16, 21, "16-20"), (21, 26, "21-25"),
               (26, 32, "26-31")]
    ad_labels, ad_rates, ad_ns = [], [], []
    for lo, hi, lab in AD_BINS:
        m = (x >= lo) & (x < hi)
        ad_labels.append(lab)
        ad_rates.append(round(float(churn[m].mean()), 4))
        ad_ns.append(int(m.sum()))
    R["churn_by_active_days"] = dict(zip(ad_labels,
                                         [{"churn_rate": r, "n": n}
                                          for r, n in zip(ad_rates, ad_ns)]))

    # zero-activity flag is its own regime
    z0 = x == 0
    R["churn_zero_activity_last_month"] = round(float(churn[z0].mean()), 4)
    R["churn_active_last_month"] = round(float(churn[~z0].mean()), 4)
    R["share_zero_activity"] = round(float(z0.mean()), 4)

    # (b) event study: months -6..0 relative to boundary
    ok_es = ok & (bm >= 6) & (bm < N_MONTHS)
    u_es, bm_es = uid[ok_es], bm[ok_es]
    d_es = out[ok_es]
    traj = {"churned": [], "renewed": []}
    for off in range(-6, 1):
        vals = ad[u_es, bm_es + off].astype(np.float32)
        traj["churned"].append(float(vals[d_es == 1].mean()))
        traj["renewed"].append(float(vals[d_es == 0].mean()))
    R["event_study_active_days"] = {k: [round(v, 2) for v in vv]
                                    for k, vv in traj.items()}
    R["event_study_n"] = {"churned": int((d_es == 1).sum()),
                          "renewed": int((d_es == 0).sum())}

    # (c) trend effect among still-active users
    act = (x > 0)
    t = f_trend[dec]
    trend_bins = [(-32, -8, "falling hard"), (-8, -3, "falling"),
                  (-3, 4, "stable"), (4, 32, "rising")]
    R["churn_by_trend"] = {}
    for lo, hi, lab in trend_bins:
        m = act & (t >= lo) & (t < hi)
        R["churn_by_trend"][lab] = {
            "churn_rate": round(float(churn[m].mean()), 4),
            "n": int(m.sum())}

    res_path = os.path.join(BASE_DIR, "kkbox_results.json")
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    allres["engagement"] = R
    json.dump(allres, open(res_path, "w"), indent=2)
    print(json.dumps(R, indent=2), flush=True)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle("Engagement Dynamics Predict Renewal-Boundary Failure "
                 "(KKBox, decisions with log history)", fontsize=12,
                 fontweight="bold")

    ax = axes[0]
    ax.plot(ad_labels, ad_rates, "o-", color="#2c7fb8")
    ax.set_xlabel("Active days, last full month")
    ax.set_ylabel("P(churn | boundary reached)")
    ax.set_title("(a) Boundary churn by engagement level")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1]
    xs = list(range(-6, 1))
    ax.plot(xs, traj["renewed"], "o-", color="#31a354", label="renewed")
    ax.plot(xs, traj["churned"], "o-", color="#de2d26", label="churned")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_xlabel("Months before boundary")
    ax.set_ylabel("Mean active days / month")
    ax.set_title("(b) The fading-out signature")
    ax.legend(frameon=False)

    ax = axes[2]
    labs = [lab for *_, lab in trend_bins]
    vals = [R["churn_by_trend"][lab]["churn_rate"] for lab in labs]
    ax.bar(labs, vals, color="#756bb1")
    ax.set_ylabel("P(churn | boundary reached)")
    ax.set_title("(c) 3-month engagement trend (active users)")
    ax.tick_params(axis="x", rotation=15)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_p = os.path.join(FIG_DIR, "fig8_engagement_cliff.png")
    fig.savefig(out_p, dpi=300)
    print(f"figure saved: {out_p}", flush=True)


if __name__ == "__main__":
    main()
