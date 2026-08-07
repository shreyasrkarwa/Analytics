"""
kkbox_cliff_model.py — change-point estimation of the renewal cliff
====================================================================
The paper's methodological core. In transaction data, membership
*termination* at the boundary is partly definitional, so the estimable
behavioral object is the DECISION hazard in boundary-relative time:

    h_s(d) = P(decision event on a day at distance d from the next
              scheduled expiration | active), for segment s

where decision events are (i) active cancellation transactions
(mid-cycle, user-chosen timing) and (ii) passive lapses (failure to
renew, occurring at d=0). Exposure is exact person-days at each d,
built from the full transaction timeline.

Model: piecewise-constant hazard with K change-points, locations
estimated by dynamic programming on the Poisson log-likelihood; K
selected by BIC. Null: smooth log-hazard (cubic truncated-power spline,
matched df). Inference: parametric bootstrap sup-LR test (change-point
locations are non-regular, so no χ² tables).

Population: 30-day plans (25–35 d), split by auto-renew status.
Outputs: kkbox_results.json["cliff_model"], figures/fig9_changepoints.png

Run: python3 kkbox_cliff_model.py
"""
import json
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE_DIR, "data_cache")
FIG_DIR = os.path.join(BASE_DIR, "figures")

GAP = 30
CUTOFF = 17256
DMAX = 45                    # build exposure out to 45 d before boundary
AN_D = 30                    # but fit models on d = 0..29 (one full cycle;
                             # d>=30 mixes grace-period renewers, and the
                             # DMAX edge bin accumulates clipped exposure)
KMAX = 8
BOOT_B = 500
SEED = 7


# ---------------------------------------------------------------------------
# 1. exposure & event construction in boundary-relative time
# ---------------------------------------------------------------------------

def build_bins():
    z = np.load(os.path.join(CACHE, "transactions_sorted.npz"))
    uid, td, eff = z["uid"], z["td"], z["eff"]
    cancel, auto, plan = z["cancel"], z["auto_renew"], z["plan_days"]
    n = len(uid)

    user_start = np.empty(n, dtype=bool)
    user_start[0] = True
    user_start[1:] = uid[1:] != uid[:-1]
    prev_eff = np.empty_like(eff)
    prev_eff[0] = 0
    prev_eff[1:] = eff[:-1]
    new_spell = user_start | (td > prev_eff + GAP)

    plan30 = (plan >= 25) & (plan <= 35)
    seg_of = auto.astype(np.int64)            # 0 manual, 1 auto

    exposure = np.zeros((2, DMAX + 2))
    ev_cancel = np.zeros((2, DMAX + 1))
    ev_lapse = np.zeros((2, DMAX + 1))

    # --- cycles: row i -> next row i+1 within same spell ------------------
    pair = ~new_spell[1:]                     # transition (i, i+1)
    i_idx = np.flatnonzero(pair)
    ok = plan30[i_idx] & (eff[i_idx] >= td[i_idx])
    i_ok = i_idx[ok]
    d_hi = np.minimum(eff[i_ok] - td[i_ok], DMAX)
    d_lo = np.maximum(eff[i_ok] - (td[i_ok + 1] - 1), 0)
    keep = d_hi >= d_lo
    s = seg_of[i_ok[keep]]
    np.add.at(exposure, (s, d_lo[keep]), 1.0)
    np.add.at(exposure, (s, d_hi[keep] + 1), -1.0)

    # cancel decision events (timing chosen by the user)
    j = i_ok + 1
    cmask = (cancel[j] == 1) & (eff[i_ok] >= td[j])
    d_c = np.minimum(eff[i_ok[cmask]] - td[j[cmask]], DMAX)
    np.add.at(ev_cancel, (seg_of[i_ok[cmask]], d_c), 1.0)

    # --- terminal rows: last transaction of each spell --------------------
    spell_first = np.flatnonzero(new_spell)
    last = np.append(spell_first[1:], n) - 1
    okL = plan30[last] & (eff[last] >= td[last])
    L = last[okL]
    decided = eff[L] + GAP <= CUTOFF
    d_hiL = np.minimum(eff[L] - td[L], DMAX)
    d_loL = np.where(decided, 0, np.maximum(eff[L] - CUTOFF, 0))
    keepL = d_hiL >= d_loL
    sL = seg_of[L[keepL]]
    np.add.at(exposure, (sL, d_loL[keepL]), 1.0)
    np.add.at(exposure, (sL, d_hiL[keepL] + 1), -1.0)

    # passive lapse events: decided, final txn not a cancel → event at d=0
    lap = decided & (cancel[L] == 0)
    np.add.at(ev_lapse, (seg_of[L[lap]], np.zeros(int(lap.sum()),
                                                  dtype=np.int64)), 1.0)

    exposure = np.cumsum(exposure, axis=1)[:, :DMAX + 1]
    events = ev_cancel + ev_lapse
    return exposure, events, ev_cancel, ev_lapse


# ---------------------------------------------------------------------------
# 2. piecewise-constant hazard via dynamic programming
# ---------------------------------------------------------------------------

def seg_ll(E, X):
    """Poisson log-lik (up to constants) of one constant-rate segment."""
    if E <= 0 or X <= 0:
        return 0.0
    return E * np.log(E / X) - E


def fit_piecewise(ev, ex, kmax=KMAX):
    """DP over bins 0..DMAX (vectorised). Returns list over K of
    (log-lik, break positions)."""
    nb = len(ev)
    cev = np.r_[0.0, np.cumsum(ev)]
    cex = np.r_[0.0, np.cumsum(ex)]

    def ll_vec(m, j):                     # segments bins m..j-1, m array
        E = cev[j] - cev[m]
        X = cex[j] - cex[m]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where((E > 0) & (X > 0), E * np.log(E / np.maximum(X, 1e-300)) - E, 0.0)
        return v

    LL = np.full((kmax + 1, nb + 1), -np.inf)
    BK = np.zeros((kmax + 1, nb + 1), dtype=int)
    js = np.arange(1, nb + 1)
    LL[0][1:] = ll_vec(np.zeros(nb, dtype=int), js)
    for k in range(1, kmax + 1):
        for j in range(k + 1, nb + 1):
            m = np.arange(k, j)
            v = LL[k - 1][m] + ll_vec(m, j)
            a = int(np.argmax(v))
            LL[k][j], BK[k][j] = v[a], m[a]
    out = []
    for k in range(kmax + 1):
        brks, j = [], nb
        for kk in range(k, 0, -1):
            j = BK[kk][j]
            brks.append(j)
        out.append((float(LL[k][nb]), sorted(brks)))
    return out


def piecewise_rates(ev, ex, breaks):
    edges = [0] + list(breaks) + [len(ev)]
    rates = np.empty(len(ev))
    segs = []
    for a, b in zip(edges[:-1], edges[1:]):
        E, X = ev[a:b].sum(), ex[a:b].sum()
        r = E / X if X > 0 else 0.0
        rates[a:b] = r
        segs.append({"d_from": int(a), "d_to": int(b - 1),
                     "rate_per_day": float(r), "events": int(E)})
    return rates, segs


# ---------------------------------------------------------------------------
# 3. smooth null: Poisson GLM with truncated-power cubic spline in d
# ---------------------------------------------------------------------------

def spline_basis(d, knots):
    d = d.astype(np.float64)
    cols = [np.ones_like(d), d / DMAX, (d / DMAX) ** 2, (d / DMAX) ** 3]
    for k in knots:
        cols.append(np.clip(d - k, 0, None) ** 3 / DMAX ** 3)
    return np.column_stack(cols)


def fit_poisson_glm(ev, ex, B):
    beta = np.zeros(B.shape[1])
    with np.errstate(divide="ignore"):
        base = np.log(max(ev.sum() / max(ex.sum(), 1e-9), 1e-12))
    beta[0] = base
    for _ in range(60):
        eta = np.clip(B @ beta, -30, 5)
        mu = ex * np.exp(eta)
        g = B.T @ (ev - mu)
        H = (B * mu[:, None]).T @ B + 1e-8 * np.eye(B.shape[1])
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(B @ beta, -30, 5)
    mu = ex * np.exp(eta)
    ll = float(np.sum(ev[mu > 0] * np.log(mu[mu > 0])) - mu.sum()
               - np.sum(ev * np.log(np.maximum(ex, 1e-12))))
    return beta, ll


# ---------------------------------------------------------------------------
# 4. bootstrap sup-LR test
# ---------------------------------------------------------------------------

def sup_lr_test(ev, ex, k_hat, knots, rng):
    B = spline_basis(np.arange(len(ev)), knots)
    beta0, ll0 = fit_poisson_glm(ev, ex, B)
    ll1 = fit_piecewise(ev, ex, k_hat)[k_hat][0]
    lr_obs = 2 * (ll1 - ll0)
    mu0 = ex * np.exp(np.clip(B @ beta0, -30, 5))
    exceed = 0
    for _ in range(BOOT_B):
        ev_b = rng.poisson(mu0).astype(np.float64)
        _, ll0b = fit_poisson_glm(ev_b, ex, B)
        ll1b = fit_piecewise(ev_b, ex, k_hat)[k_hat][0]
        if 2 * (ll1b - ll0b) >= lr_obs:
            exceed += 1
    return lr_obs, (exceed + 1) / (BOOT_B + 1)


# ---------------------------------------------------------------------------

def analyze_segment(name, ev, ex, rng, R):
    ev, ex = ev[:AN_D], ex[:AN_D]
    n_ev = ev.sum()
    fits = fit_piecewise(ev, ex)
    bics = [(-2 * ll + (2 * k + 1) * np.log(max(n_ev, 1)), k, brks)
            for k, (ll, brks) in enumerate(fits)]
    _, k_hat, breaks = min(bics)
    rates, segs = piecewise_rates(ev, ex, breaks)
    knots = [7, 15, 22]
    lr, pval = sup_lr_test(ev, ex, k_hat, knots, rng)

    # concentration index
    csum = np.cumsum(ev) / n_ev
    R[name] = {
        "n_events": int(n_ev),
        "person_days_exposure": float(ex.sum()),
        "K_selected_by_BIC": int(k_hat),
        "changepoints_days_before_boundary": [int(b) for b in breaks],
        "segments": segs,
        "sup_LR_vs_smooth_spline": round(float(lr), 1),
        "bootstrap_p_value": round(float(pval), 5),
        "bic_by_K": {str(k): round(b, 1) for b, k, _ in sorted(
            bics, key=lambda t: t[1])},
        "renewal_cliff_index": {
            f"share_within_{w}d": round(float(csum[w]), 4)
            for w in (0, 1, 3, 7, 14)},
    }
    return rates


def main():
    rng = np.random.default_rng(SEED)
    exposure, events, ev_cancel, ev_lapse = build_bins()
    R = {"population": "30-day plans (25-35d)",
         "note": ("decision events = active cancel txns + passive lapses; "
                  "d = days before scheduled expiration")}
    rates = {}
    for s, name in [(1, "auto_renew"), (0, "manual_renew")]:
        rates[name] = analyze_segment(name, events[s], exposure[s], rng, R)
        print(name, json.dumps({k: R[name][k] for k in
                                ["K_selected_by_BIC",
                                 "changepoints_days_before_boundary",
                                 "sup_LR_vs_smooth_spline",
                                 "bootstrap_p_value",
                                 "renewal_cliff_index"]}, indent=1),
              flush=True)

    res_path = os.path.join(BASE_DIR, "kkbox_results.json")
    allres = json.load(open(res_path)) if os.path.exists(res_path) else {}
    allres["cliff_model"] = R
    json.dump(allres, open(res_path, "w"), indent=2)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.arange(DMAX + 1)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    fig.suptitle("Estimated Change-Points in the Renewal-Decision Hazard "
                 "(KKBox, 30-day plans)", fontsize=12, fontweight="bold")
    for ax, (s, name, c) in zip(
            axes[:2], [(1, "auto_renew", "#2c7fb8"),
                       (0, "manual_renew", "#de2d26")]):
        h = np.divide(events[s], exposure[s],
                      out=np.zeros(DMAX + 1), where=exposure[s] > 0)
        ax.semilogy(d, h, "o", ms=3.5, color=c, alpha=0.7,
                    label="empirical daily hazard")
        ax.semilogy(d[:AN_D], rates[name], "-", color="black", lw=1.8,
                    label="piecewise fit (BIC), d<30")
        for b in R[name]["changepoints_days_before_boundary"]:
            ax.axvline(b - 0.5, color="gray", ls=":", lw=1)
        ax.invert_xaxis()
        ax.set_xlabel("Days before scheduled expiration")
        ax.set_ylabel("Decision hazard (events / person-day)")
        ax.set_title(f"{name}  (K={R[name]['K_selected_by_BIC']}, "
                     f"p={R[name]['bootstrap_p_value']:.4f})")
        ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for s, name, c in [(1, "auto_renew", "#2c7fb8"),
                       (0, "manual_renew", "#de2d26")]:
        csum = np.cumsum(events[s]) / events[s].sum()
        ax.plot(d, csum, color=c, label=name)
    ax.invert_xaxis()
    ax.set_xlabel("Days before scheduled expiration")
    ax.set_ylabel("Cumulative share of decision events")
    ax.set_title("Concentration: the Renewal Cliff Index")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG_DIR, "fig9_changepoints.png")
    fig.savefig(out, dpi=300)
    print(f"figure saved: {out}", flush=True)


if __name__ == "__main__":
    main()
