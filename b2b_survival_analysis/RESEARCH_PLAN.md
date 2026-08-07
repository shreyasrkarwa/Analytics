# Research Plan: The Renewal Cliff — Contract-Boundary Hazard Dynamics in Subscription Churn

**Goal:** A peer-reviewed publication in a reputed journal within 12–18 months, strong enough to support an EB1A "original contributions" and "authorship of scholarly articles" claim.

**Working title:** *The Renewal Cliff: Change-Point Hazard Dynamics and Dynamic Churn Prediction at Contract Boundaries in Subscription Businesses*

---

## 1. Honest assessment of current state

| Component | Status | Verdict |
|---|---|---|
| Synthetic data generator (5,000 accounts) | Done | Keep only as simulation appendix — cannot be primary evidence (author-defined DGP) |
| Time-varying Cox + landmarking | Done | Keep as baseline; C-index 0.62 is weak |
| Stratified Cox + Schoenfeld tests | Done | Keep as robustness check; stratification lowered C-index |
| Renewal cliff (piecewise exponential) | Done | Core idea is good; current results non-significant (all p > 0.32) because effect was simulated weakly |
| Telco validation | Done | **Has leakage**: `TotalCharges` ≈ tenure × MonthlyCharges encodes the outcome → C=0.90 is inflated. Must drop TotalCharges. |
| evaluation_metrics.py (Uno's C, Brier, IBS, td-AUC) | Written, not wired in | Wire into main pipeline |
| SSRN PDF | Done | Retire; rewrite around new results |

**Bottom line:** solid engineering scaffold, but as a paper it would be desk-rejected: synthetic primary data, textbook methods, weak/non-significant results, leaky validation.

## 2. The contribution (what makes it publishable)

Literature confirms renewal-period churn spikes are *observed* (e.g., sharp survival drops at 6/12-month contract points; concordance differing sharply pre- vs. post-contract-expiration) but not *formalized*. The gap we claim:

1. **Formal model:** a change-point / piecewise hazard framework where the hazard has estimated (not assumed) discontinuities at contract boundaries — magnitude, decay, and covariate modulation of the "cliff" are parameters. Discrete-time hazard formulation with boundary-proximity splines; change-point locations estimated and tested (score/likelihood-ratio tests against smooth-hazard null).
2. **Cliff-aware dynamic prediction:** show that landmarking/dynamic models that encode time-to-renewal-boundary materially beat boundary-blind models (Cox-TV, RSF, DeepSurv, DeepHit) on time-dependent AUC and IBS — especially in the 1–3 months before a boundary, which is exactly the actionable window for retention teams.
3. **Managerial quantification:** decompose churn into "cliff churn" vs. "attritional churn" per segment; translate into intervention-timing value (expected revenue saved by targeting the pre-cliff window).

## 3. Data strategy (public, reproducible)

- **Primary: KKBox (WSDM 2018 Kaggle challenge).** ~1M+ subscribers, 2 years of transaction logs with `membership_expire_date`, `is_auto_renew`, `is_cancel`, plus daily listening logs → **real renewal boundaries and real time-varying engagement**. This is the standard academic benchmark for dynamic churn survival (used by DeepSurv/DeepHit/Cox-Time papers), so results are directly comparable to published numbers. Churn = no resubscription within 30 days of expiry.
- **Secondary: IBM Telco** (fixed: drop TotalCharges; use Contract type to infer boundary months) — small but universally recognized.
- **Tertiary (generalization):** one more subscription dataset with renewal timing (candidates to evaluate: a telecom dataset with contract end dates, or a public SaaS/CRM dataset if one with renewal dates exists; fall back to KKBox subsamples by plan length: 30/90/180/365-day plans give four distinct boundary structures — arguably better than a third dataset).
- **Simulation appendix:** repurpose the existing generator as a *controlled recovery study* — show the estimator recovers known cliff parameters. This is the legitimate use of synthetic data.

## 4. Methods & evaluation plan

**Models:** (a) smooth-hazard null (Cox-TV, penalized spline hazard); (b) proposed cliff model (discrete-time hazard w/ boundary-distance basis + estimated change-points); (c) ML baselines: RSF, XGBoost-AFT, DeepSurv, DeepHit/Dynamic-DeepHit, Cox-Time. All baselines get the same features *except* boundary-distance, then again *with* it (ablation showing the feature vs. model contribution).

**Evaluation:** Harrell's C and Uno's C, cumulative/dynamic AUC at horizons {1,3,6,12} months, Brier score + IBS, calibration plots, landmark analysis anchored at k months before each renewal boundary. Statistical tests: likelihood-ratio for change-points, bootstrap CIs on metric differences. Strict temporal train/test split (train on year 1, test on year 2) — no random splits.

**Reproducibility:** public GitHub repo, pinned environment, one-command pipeline (extend `run_all_models.py`), released feature-engineering code. Reviewers and citers reward this; it also drives citations (relevant for EB1A).

## 5. Target venues (in order)

1. **European Journal of Operational Research** — strong OR/analytics fit, high reputation, publishes churn/survival work.
2. **Expert Systems with Applications** — high IF, faster review, very citable; strong fit for method+benchmark papers.
3. **International Journal of Forecasting** or **Journal of Business Research** — if framing tilts empirical/managerial.
4. Backup: **Machine Learning with Applications**, **Journal of Marketing Analytics** (published closely related churn-survival work in 2025).

Post immediately to **arXiv/SSRN** on submission (establishes priority + starts citation clock while under review).

## 6. Timeline (18-month EB1A window)

| Months | Milestone |
|---|---|
| 1 | Acquire KKBox data (Kaggle), build survival-format ETL (counting-process with real expiry dates); fix Telco leakage |
| 2–3 | Implement cliff model + change-point tests; wire evaluation_metrics.py into pipeline |
| 3–4 | ML baselines + ablations; simulation recovery appendix |
| 5 | Full results, figures, robustness (temporal splits, plan-length subgroups) |
| 6–7 | Write paper; internal red-team review; arXiv preprint |
| 7 | Submit to EJOR or ESWA |
| 8–13 | Review cycle; meanwhile: present at a conference/workshop, promote preprint (citations) |
| 13–18 | Revision → acceptance; second short paper (e.g., benchmark note) optional |

## 7. EB1A-specific notes (not legal advice)

- Peer-reviewed acceptance + citations >> preprint alone. Start the citation clock early via arXiv.
- Judging/reviewing for journals in this space is a separate EB1A criterion — after the preprint is public, volunteer as a reviewer (ESWA/JBR-tier journals accept reviewer sign-ups).
- Keep the independent-research disclaimer (no employer data/IP) — you already do this.
- A public, starred, cited GitHub repo supports the "original contribution of major significance" narrative.

## 8. Immediate next actions

1. ~~Download KKBox dataset from Kaggle~~ ✅
2. Fix `validation_telco.py`: drop `TotalCharges`, re-run, report honest C-index.
3. Wire `evaluation_metrics.py` (Uno's C, IBS, td-AUC) into `run_all_models.py`.
4. ~~Build `kkbox_etl.py`: transactions → counting-process format~~ ✅
5. Prototype the cliff model on KKBox 30-day-plan cohort.

## 9. Progress log

**2026-07-22/23 — KKBox data backbone built**

- `sevenz_stream.py`: streams the .7z archives via libarchive (no 30 GB extraction).
- `kkbox_etl.py` (resumable): 23.0M transactions → 3.03M spells / 2.43M users; 26M-row 30-day person-period panel with `days_to_boundary`; WSDM churn rule replicated with **98.8% agreement** vs official `train.csv` labels (859,865 evaluable users; disagreements skew "official says churn, we say renewed" → cancel-handling nuance, footnote material).
- `kkbox_members_etl.py`: demographics joined (82% coverage; age/gender known for ~40%).
- `kkbox_boundary_analysis.py` → `boundaries.npz` (20.8M renewal decisions), `kkbox_results.json`, `figures/fig7_kkbox_renewal_cliff.png`. Headlines:
  - Overall P(churn | boundary reached) = 9.25%; hazard gradient vs. days-to-boundary spans ~3 orders of magnitude (0.0002 → 0.14 per 30d).
  - Experience effect: 1st boundary 33.3% → 2nd 20.3% → 3rd 6.1% → 13th+ 2.4%.
  - Auto-renew 4.9% vs manual 33.0% churn per boundary.
  - Contract length U-shape: 30d plans 6.3%, 180d 27.5%, 365d 31.1% (long-plan boundaries are rarer but far riskier — the B2B analogy).
  - Churn decomposition: 70% passive lapse at boundary, 30% active mid-cycle cancellation.
- `user_logs_aggregate.py`: ready for a one-shot local run (~45–90 min) to add monthly engagement covariates (30.5 GB streamed).

Next: engagement join → discrete-time cliff model with estimated change-points vs smooth-hazard null → ML baselines.

**2026-07-23 — engagement features + first hazard models**

- `user_logs_aggregate.py` run locally by author: 410.5M daily log rows → monthly engagement, 79.9% coverage, clean.
- `kkbox_engagement_features.py` → `boundary_eng.npz`, `fig8_engagement_cliff.png`:
  - Zero-activity month before boundary → 20.3% churn vs 6.7% (any activity), 3.9% (daily listeners). Monotone gradient.
  - Event study shows a level gap (~2×) between renewers and churners for 6+ months pre-boundary; *caveat*: raw trajectories contaminated by secular platform growth and month-0 truncation → redo with calendar-month fixed effects for the paper.
  - Raw 3-month trend is U-shaped (rising-engagement users churn more) — new-user composition effect; must condition on boundary index.
- `kkbox_hazard_models.py` (NumPy Newton logistic, 8.7M person-period sample, temporal split at 2016-07):
  - M0 boundary-blind AUC **0.840** → M1 +days-to-boundary **0.880** → M2 +engagement **0.882** (log-loss 0.1400/0.1346/0.1331). Near-boundary subset: 0.830/0.857/0.858.
  - Fitted cliff odds ratios (vs >92d): 65× at 32–92d, ~1,800–4,400× within 31d, 831× past-due.
  - Interpretation so far: boundary proximity is the dominant source of discrimination; linear engagement terms add mainly calibration. Engagement's real test needs richer trajectories, eng×boundary interactions, and nonlinear ML baselines → next stage (local machine with sklearn/xgboost/lifelines).

Next: (1) change-point estimation replacing fixed dtb bins + LR test vs smooth hazard; (2) eng×dtb interactions; (3) calendar-FE event study; (4) ML baseline suite locally.

**2026-07-23 — change-point cliff model (methodological core)**

- Key design decision: in transaction data, *termination at expiry* is partly definitional, so the estimable behavioral object is the **decision hazard in boundary-relative time** — active cancel transactions (user-chosen timing) + passive lapses at d=0, over exact person-day exposure at each distance d from scheduled expiration. This framing is honest, novel, and portable across datasets.
- `kkbox_cliff_model.py`: piecewise-constant hazard, change-points by DP on Poisson likelihood, K by BIC, parametric-bootstrap sup-LR vs cubic-spline smooth null (B=500). Population: 30-day plans, fit window d=0..29.
- Results (`kkbox_results.json["cliff_model"]`, `fig9_changepoints.png`):
  - **Auto-renew**: flat mid-cycle hazard ~8e-4/day; escalation change-points at d=8,6,5,3,2,1; expiry-day spike ~8e-2 (≈100×). K̂ hit KMAX=8 — near-boundary shape is steep enough that piecewise wants many segments; consider parametric spike+decay form for the paper. sup-LR=53,256, bootstrap p=0.002 (floor). RCI: 31% of decisions on expiry day, 61% within 7d.
  - **Manual-renew**: near-pure point mass at boundary (99.8% of decisions at d=0); K̂=3 (breaks 1, 26, 29). sup-LR=312, p=0.002.
  - Change-points at d=26–29 in both = cycle-start dynamics (renew-then-immediately-cancel behavior) — interesting in its own right.
  - Smooth-hazard null decisively rejected in both segments → the cliff is a genuine discontinuity, not a steep smooth trend. This is the paper's central statistical claim.
- Caveats logged: BIC with 631M person-days saturates K (use out-of-sample or penalized selection for final paper); bootstrap p floor at 1/(B+1); RCI denominators use the d<30 window for model fits, full window for concentration curve.

Next: ML baseline suite + lifelines-based Cox/AFT on local machine (script to prepare); eng×dtb interactions; calendar-FE event study; then paper drafting.

**2026-07-23 — ML baseline suite prepared for local run**

- `kkbox_ml_baselines.py`: LogisticRegression + HistGradientBoosting (+ XGBoost if installed) on the person-period panel; FS0/FS1/FS2 nested ablation matching the sandbox models, plus FS3 = engagement×near-boundary interactions (tests whether engagement matters more at the boundary). Temporal split, AUC/PR-AUC/log-loss/Brier/near-boundary AUC + reliability tables → `ml_results.json` (paper Table 3). Feature pipeline smoke-tested (37 features, no NaN/inf).
- Author to run locally: `pip install scikit-learn xgboost` then `python3 kkbox_ml_baselines.py` (~10–30 min full panel; `--sample 3` for a quick pass).

**2026-07-24 — ML baselines run (author's machine, full 26M panel) → `ml_results.json`, `fig10_model_comparison_kkbox.png`**

| Model | FS0 blind | FS1 +boundary | FS2 +engagement |
|---|---|---|---|
| Logistic | 0.8432 | 0.8813 | 0.8832 |
| HistGB | 0.8675 | 0.9076 | 0.9104 |
| XGBoost | 0.8678 | 0.9068 | 0.9095 |

- **Headline: boundary-awareness (+0.038–0.040 AUC) is worth more than model nonlinearity (+0.024–0.026), and the two are additive** → best 0.910. Near-boundary AUC peaks at 0.892 (hgb_FS2). PR-AUC 0.69 at a 5.5% base rate.
- FS3 interactions: `nearXad_m1 = +1.35` (vs negative main effect) → engagement's protective effect is concentrated **mid-cycle**: it predicts *who actively cancels*, while boundary distance predicts *when passive churn happens*. Coheres exactly with the cancel/lapse two-process decomposition — a unifying thread for the paper.
- Calibration drift: test-period reliability shows overprediction (train era churnier than test era; platform maturing). Report + recalibrate (Platt on trailing window) in final models.

Remaining before drafting: calendar-FE event study; Telco secondary analysis (fix leakage, apply RCI + boundary decomposition); optional DeepSurv/DeepHit for reviewer completeness; then paper.

**2026-07-24 — final pre-drafting analyses**

- `kkbox_event_study.py` → `fig8b_event_study_fe.png`, `kkbox_results.json["event_study_fe"]`: with calendar-month FE and tenure≥180d restriction, churners decline ~1 active day/month relative to norms in the final 2 months (gap: 0.0 at t−4 → 1.06 at t−1). The fading-out signal is real but **late and modest**; engagement *levels* carry most information — coheres with ML finding (trend features weak).
- `telco_boundary_analysis.py` → `telco_results.json`, `fig11_telco_boundaries.png`:
  - Leakage fixed: honest holdout AUC **0.879** (deprecates the old C≈0.905 with TotalCharges in `results.json`).
  - **No significant anniversary spikes**: one-year obs/exp = 1.21 (p=0.27, min detectable ratio at 80% power = 1.73); two-year underpowered (1 event); month-to-month placebo 0.95. Hazard-model boundary terms LR p=0.61.
  - Interpretation (paper section 6): US telecom contracts auto-convert to month-to-month — the anniversary never forces a payment re-authorization decision, unlike KKBox. **Cliffs arise where boundaries force decisions** — a testable heterogeneity prediction across subscription designs, and it converts the Telco null into theory-strengthening evidence rather than a failed replication.

STATUS: all planned analyses complete. Next: paper drafting (target EJOR/ESWA), then arXiv.

Paper asset inventory: figs 7–11, kkbox_results.json, telco_results.json, ml_results.json, 9 analysis scripts, resumable ETL, 98.8% label validation. Deprecated (synthetic era): results.json, ssrn_submission.pdf, figures 1–6.
