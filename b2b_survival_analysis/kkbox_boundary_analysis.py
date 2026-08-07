"""
kkbox_boundary_analysis.py — renewal-boundary decision dataset + empirics
==========================================================================
Constructs the paper's core empirical object from the KKBox transaction
stream: one row per *renewal decision* (a subscription cycle reaching or
closing at its effective expiration), and produces the first real-data
renewal-cliff statistics and figure.

Definitions
-----------
* Within a spell, consecutive transactions i → i+1 define a renewal with
  timing delta = td[i+1] − eff[i] (days relative to the boundary).
  delta ≥ −REACH_WINDOW means the boundary was genuinely reached
  (auto-renew charges fire at/around expiry); large negative deltas are
  early plan changes/top-ups, not renewal decisions.
* The final boundary of each spell is a churn (event=1) or is censored
  when expiry+30d exceeds the data cutoff.
* Churn decomposition: a churned spell whose final transaction has
  is_cancel=1 is an *active cancellation* (decision made mid-cycle);
  otherwise it is a *passive lapse* at the boundary.

Outputs
-------
  data_cache/boundaries.npz        decision-level dataset
  kkbox_results.json               headline statistics (git-tracked)
  figures/fig7_kkbox_renewal_cliff.png

Run: python3 kkbox_boundary_analysis.py
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

GAP = 30
CUTOFF = 17256                       # 2017-03-31, days since epoch
REACH_WINDOW = 7                     # renewal within [eff-7, eff+30] = reached


def load():
    z = np.load(os.path.join(CACHE_DIR, "transactions_sorted.npz"))
    return (z["uid"], z["td"], z["eff"], z["cancel"], z["auto_renew"],
            z["plan_days"])


def main():
    uid, td, eff, cancel, auto_renew, plan_days = load()
    n = len(uid)

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = uid[1:] != uid[:-1]
    prev_eff = np.empty_like(eff)
    prev_eff[0] = 0
    prev_eff[1:] = eff[:-1]
    new_spell = user_start | (td > prev_eff + GAP)
    spell_id = np.cumsum(new_spell, dtype=np.int64) - 1
    spell_first = np.flatnonzero(new_spell)
    spell_start_of = td[spell_first]          # index by spell_id

    # --- renewal transitions (row i -> i+1 within same spell) -------------
    same_spell = ~new_spell[1:]
    i_idx = np.flatnonzero(same_spell)        # transition from row i
    delta = (td[i_idx + 1] - eff[i_idx]).astype(np.int32)
    reached_mask = delta >= -REACH_WINDOW

    # --- decision table ----------------------------------------------------
    # reached renewals
    r_rows = i_idx[reached_mask]
    # final boundaries (last row of each spell)
    last_rows = np.append(spell_first[1:], n) - 1
    ev = (eff[last_rows] + GAP <= CUTOFF)
    # outcome codes: 0 renewed, 1 churned, 2 censored
    d_rows = np.concatenate([r_rows, last_rows])
    d_out = np.concatenate([np.zeros(len(r_rows), dtype=np.int8),
                            np.where(ev, 1, 2).astype(np.int8)])
    d_delta = np.concatenate([delta[reached_mask],
                              np.full(len(last_rows), 127, dtype=np.int32)])
    order = np.argsort(d_rows, kind="stable")
    d_rows, d_out, d_delta = d_rows[order], d_out[order], d_delta[order]

    d_uid = uid[d_rows]
    d_spell = spell_id[d_rows]
    d_bdate = eff[d_rows]
    d_auto = auto_renew[d_rows]
    d_plan = plan_days[d_rows]
    d_cancel = cancel[d_rows]
    d_tenure = (d_bdate - spell_start_of[d_spell]).astype(np.int32)

    # boundary index within spell (0-based)
    sfirst = np.r_[0, np.flatnonzero(np.diff(d_spell)) + 1]
    counts = np.diff(np.r_[sfirst, len(d_spell)])
    d_bindex = (np.arange(len(d_spell))
                - np.repeat(sfirst, counts)).astype(np.int32)

    np.savez_compressed(
        os.path.join(CACHE_DIR, "boundaries.npz"),
        uid=d_uid.astype(np.int32), spell_id=d_spell.astype(np.int32),
        boundary_date=d_bdate.astype(np.int32), outcome=d_out,
        boundary_index=d_bindex, auto_renew=d_auto.astype(np.int8),
        plan_days=d_plan.astype(np.int16), is_cancel=d_cancel.astype(np.int8),
        tenure_days=d_tenure, renew_delta=d_delta.astype(np.int8, copy=False)
        if d_delta.dtype != np.int8 else d_delta)

    # --- headline statistics ------------------------------------------------
    R = {}
    dec = d_out != 2                          # decided (not censored)
    churn = d_out == 1
    R["n_decisions"] = int(dec.sum())
    R["n_censored"] = int((~dec).sum())
    R["overall_boundary_churn_rate"] = round(float(churn[dec].mean()), 4)

    # renewal timing distribution
    hist, edges = np.histogram(delta, bins=np.arange(-45, 32))
    R["renewal_timing"] = {
        "share_on_time_pm3d": round(float(np.mean(np.abs(delta) <= 3)), 4),
        "share_grace_1_30d": round(float(np.mean((delta >= 1)
                                                 & (delta <= 30))), 4),
        "share_early_lt_minus7": round(float(np.mean(delta < -7)), 4),
    }

    # churn prob by boundary index (experience effect)
    by_bi = {}
    for lo, hi, lab in [(0, 1, "1st"), (1, 2, "2nd"), (2, 3, "3rd"),
                        (3, 6, "4-6th"), (6, 12, "7-12th"),
                        (12, 10**9, "13th+")]:
        m = dec & (d_bindex >= lo) & (d_bindex < hi)
        by_bi[lab] = {"churn_rate": round(float(churn[m].mean()), 4),
                      "n": int(m.sum())}
    R["churn_by_boundary_index"] = by_bi

    # by auto-renew
    R["churn_by_auto_renew"] = {
        "auto": round(float(churn[dec & (d_auto == 1)].mean()), 4),
        "manual": round(float(churn[dec & (d_auto == 0)].mean()), 4)}

    # by plan length
    by_pl = {}
    for lo, hi, lab in [(0, 8, "<=7d"), (8, 35, "30d"), (35, 100, "90d"),
                        (100, 200, "180d"), (200, 10**9, "365d+")]:
        m = dec & (d_plan >= lo) & (d_plan < hi)
        by_pl[lab] = {"churn_rate": round(float(churn[m].mean()), 4),
                      "n": int(m.sum())}
    R["churn_by_plan_days"] = by_pl

    # active cancellation vs passive lapse among churned boundaries
    ch = churn
    R["churn_decomposition"] = {
        "active_cancel_share": round(float(d_cancel[ch].mean()), 4),
        "passive_lapse_share": round(float(1 - d_cancel[ch].mean()), 4),
        "auto_renew_share_among_churn": round(float(d_auto[ch].mean()), 4)}

    # cliff-vs-attritional: hazard concentration
    # (share of all churn events occurring within ±3d of a boundary is 1.0
    #  by construction in transaction data; the meaningful split is
    #  active-cancel timing)
    canc_rows = i_idx[(cancel[i_idx + 1] == 1)]
    R["n_mid_cycle_cancel_txns"] = int((cancel == 1).sum())

    with open(os.path.join(BASE_DIR, "kkbox_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))

    # --- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("The Renewal Cliff in 23M KKBox Subscription Transactions",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, hist / hist.sum(), width=1, color="#2c7fb8")
    ax.axvline(0, color="crimson", ls="--", lw=1)
    ax.set_xlabel("Renewal transaction timing vs boundary (days)")
    ax.set_ylabel("Share of renewals")
    ax.set_title("(a) Renewals concentrate at the boundary")

    ax = axes[0, 1]
    labs = list(by_bi)
    ax.bar(labs, [by_bi[k]["churn_rate"] for k in labs], color="#de2d26")
    ax.set_ylabel("P(churn | boundary reached)")
    ax.set_title("(b) Cliff height falls with renewal experience")
    ax.set_xlabel("Boundary number within spell")

    ax = axes[1, 0]
    labs = list(by_pl)
    ax.bar(labs, [by_pl[k]["churn_rate"] for k in labs], color="#756bb1")
    ax.set_ylabel("P(churn | boundary reached)")
    ax.set_xlabel("Plan length")
    ax.set_title("(c) Cliff height by contract length")

    ax = axes[1, 1]
    cd = R["churn_decomposition"]
    ax.bar(["Active cancel\n(mid-cycle decision)", "Passive lapse\n(at boundary)"],
           [cd["active_cancel_share"], cd["passive_lapse_share"]],
           color=["#e6550d", "#31a354"])
    ax.set_ylabel("Share of churned boundaries")
    ax.set_title("(d) Churn decomposition")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, "fig7_kkbox_renewal_cliff.png")
    fig.savefig(out, dpi=300)
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
