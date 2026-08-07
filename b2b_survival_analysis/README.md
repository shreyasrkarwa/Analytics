# The Renewal Cliff: Change-Point Hazards and Boundary-Aware Churn Prediction in Subscription Markets

Research code and manuscript for a large-scale empirical study of subscription churn at contract renewal boundaries, using 23.0M transactions / 2.43M subscribers from the public KKBox (WSDM Cup 2018) dataset and the IBM Telco dataset.

**Manuscript:** [`paper/manuscript.md`](paper/manuscript.md) (working draft)
**Research plan & progress log:** [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)

## Headline findings

1. **The renewal cliff is a genuine discontinuity, not a steep slope.** A piecewise-constant hazard with change-points estimated by dynamic programming rejects a smooth-spline null via parametric bootstrap (sup-LR = 53,256; p ≈ 0.002). Auto-renewers' daily decision hazard is flat mid-cycle (≈8×10⁻⁴), escalates from 8 days out, and spikes ~105× on the expiration day.
2. **Boundary-awareness beats model complexity for churn prediction.** Adding days-to-boundary features gains +0.038–0.040 AUC in every model class — more than upgrading logistic regression to XGBoost (+0.024–0.026). Best AUC 0.910 on a 26M person-period panel with a strict temporal split.
3. **Engagement predicts who actively cancels; the calendar determines when passive churn happens.** 70% of churn is passive lapse at the boundary; usage telemetry discriminates mainly mid-cycle.
4. **Cliffs arise where boundaries force decisions.** Telco contracts that auto-convert at term end show no anniversary spikes (obs/exp = 1.21, p = 0.27) — a design-based theory with falsifiable predictions.
5. **The Renewal Cliff Index (RCI)** — share of churn decisions within w days of a boundary — summarizes cliff geometry in one number computable from any billing system (KKBox auto-renew: RCI(0) = 0.31; manual: 0.998).

## Reproducing the results

Data are not committed (see `.gitignore`). Download the KKBox archives from the [WSDM-KKBox Kaggle competition](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) into `kkbox_churn_prediction_kaggle_data/`, then run in order:

```
python3 kkbox_etl.py --budget 100000   # transactions → spells → panel (resumable)
python3 kkbox_members_etl.py           # demographics
python3 user_logs_aggregate.py         # 30 GB listening logs → monthly (~1 h)
python3 kkbox_boundary_analysis.py     # renewal decisions + Fig 1
python3 kkbox_engagement_features.py   # engagement features + Fig 4
python3 kkbox_event_study.py           # calendar-adjusted event study + Fig 5
python3 kkbox_hazard_models.py         # discrete-time hazard models
python3 kkbox_cliff_model.py           # change-point model + bootstrap + Fig 2
python3 kkbox_ml_baselines.py          # ML suite + Fig 3 inputs
python3 telco_boundary_analysis.py     # Telco secondary analysis + Fig 6
```

Requires only pandas/numpy/matplotlib for the core pipeline (scikit-learn/xgboost for `kkbox_ml_baselines.py`). The `.7z` archives are streamed via the system libarchive (`sevenz_stream.py`) — nothing is extracted to disk. Numeric outputs are committed: `kkbox_results.json`, `ml_results.json`, `telco_results.json`.

Churn-label reconstruction is validated at **98.77% agreement** against the competition's official labels (859,865 users).

## Repository layout

| Path | Contents |
|---|---|
| `paper/manuscript.md` | Working paper draft (figures embedded) |
| `kkbox_*.py`, `telco_boundary_analysis.py`, `user_logs_aggregate.py`, `sevenz_stream.py` | Analysis pipeline (see above) |
| `*_results.json` | All numeric results cited in the paper |
| `figures/fig7–fig11*` | Paper figures (Figs 1–6 in manuscript numbering) |
| `RESEARCH_PLAN.md` | Plan, decisions, and dated progress log |
| `data_generator.py`, `run_all_models.py`, `results.json`, `ssrn_submission.pdf`, figures 1–6 | **Deprecated** early synthetic-data prototype, kept for provenance; superseded by the KKBox analyses (the old Telco validation contains a known leakage issue, corrected in `telco_boundary_analysis.py`) |

## Author

Shreyas Karwa — independent research. Contact: shreyasrkarwa@gmail.com
