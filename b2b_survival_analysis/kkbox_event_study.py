"""
kkbox_event_study.py — calendar-adjusted pre-boundary engagement
=================================================================
Corrects the raw fading-out event study (fig8b) for two artifacts:
  1. Secular platform growth — active days per user rise over calendar
     time, contaminating raw trajectories. Fixed by residualizing on
     calendar-month means estimated from the pooled study sample.
  2. Month-0 truncation — the boundary month is partially observed.
     Fixed by using offsets -6..-1 only.
Sensitivity: decisions with >=180 days of spell tenure (removes the
new-user composition effect).

Outputs: kkbox_results.json["event_study_fe"],
         figures/fig8b_event_study_fe.png
Run: python3 kkbox_event_study.py
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "data_cache")
MONTH_BASE = (2015 - 1970) * 12
N_MONTHS = 27
OFFSETS = list(range(-6, 0))


def month_idx(days):
    m = days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)
    return (m - MONTH_BASE).astype(np.int32)


def main():
    eng = np.load(os.path.join(CACHE, "eng_compact.npz"))
    ad = eng["active_days"]
    has_logs = (ad.sum(axis=1, dtype=np.int32) > 0)

    b = np.load(os.path.join(CACHE, "boundaries.npz"))
    uid, out, ten = b["uid"], b["outcome"], b["tenure_days"]
    bm = month_idx(b["boundary_date"])

    base = (bm >= 7) & (bm < N_MONTHS) & has_logs[uid] & (out != 2)
    samples = {"all": base, "tenure_180d": base & (ten >= 180)}

    R = {"offsets": OFFSETS, "note": ("active days residualized on "
         "calendar-month means; month 0 dropped (truncated)")}
    curves = {}
    for sname, mask in samples.items():
        u, m, o = uid[mask], bm[mask], out[mask]
        # calendar means from pooled (decision, offset) observations
        cal_sum = np.zeros(N_MONTHS)
        cal_n = np.zeros(N_MONTHS)
        vals = {}
        for off in OFFSETS:
            v = ad[u, m + off].astype(np.float64)
            vals[off] = v
            np.add.at(cal_sum, m + off, v)
            np.add.at(cal_n, m + off, 1.0)
        cal_mean = cal_sum / np.maximum(cal_n, 1)

        res = {"churned": [], "renewed": [], "gap": [],
               "n_churned": int((o == 1).sum()),
               "n_renewed": int((o == 0).sum())}
        for off in OFFSETS:
            r = vals[off] - cal_mean[m + off]
            c_mean = float(r[o == 1].mean())
            r_mean = float(r[o == 0].mean())
            res["churned"].append(round(c_mean, 3))
            res["renewed"].append(round(r_mean, 3))
            res["gap"].append(round(r_mean - c_mean, 3))
        R[sname] = res
        curves[sname] = res
        print(sname, json.dumps(res, indent=1), flush=True)

    res_path = os.path.join(BASE_DIR, "kkbox_results.json")
    allres = json.load(open(res_path))
    allres["event_study_fe"] = R
    json.dump(allres, open(res_path, "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for sname, ls in [("all", "-"), ("tenure_180d", "--")]:
        c = curves[sname]
        ax.plot(OFFSETS, c["renewed"], ls, color="#31a354", marker="o",
                ms=4, label=f"renewed ({sname})")
        ax.plot(OFFSETS, c["churned"], ls, color="#de2d26", marker="o",
                ms=4, label=f"churned ({sname})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Months before renewal boundary")
    ax.set_ylabel("Active days vs calendar-month mean")
    ax.set_title("The fading-out signature, calendar-adjusted",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, "figures",
                             "fig8b_event_study_fe.png"), dpi=300)
    print("figure saved", flush=True)


if __name__ == "__main__":
    main()
