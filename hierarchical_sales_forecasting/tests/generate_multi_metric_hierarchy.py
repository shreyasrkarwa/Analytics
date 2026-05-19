"""
Generate a synthetic multi-metric B2B hierarchy.

Each IC carries 4 quarters of historical performance for several metrics:
  - NetNewACV       (the metric being cascaded; "proportional")
  - CloudSeats      (more cloud seats -> more NetNewACV; "proportional")
  - DCSeats         (more on-prem / data-center seats -> LESS NetNewACV;
                     "inverse" because DC accounts skew toward maintenance
                     rather than new ARR)
  - ExpansionSpent  (already-spent expansion dollars in the LTM -> LESS
                     headroom for new ACV; single LTM column, "inverse")

The relationships are seeded with controlled correlations so that
MetricSpec.suggest_from_data() should recover sensible weights/directions.
"""
import pandas as pd
import numpy as np
import random
import os


def generate_multi_metric_hierarchy():
    np.random.seed(7)
    random.seed(7)

    regions = ['NA', 'EMEA', 'APAC']
    rows = []

    for region in regions:
        num_rvps = random.randint(1, 2)
        for rvp_idx in range(num_rvps):
            rvp = f"RVP_{region}_{rvp_idx+1}"
            num_dirs = random.randint(2, 3)
            for d_idx in range(num_dirs):
                director = f"Dir_{rvp}_{d_idx+1}"
                num_mgrs = random.randint(2, 4)
                for m_idx in range(num_mgrs):
                    manager = f"Mgr_{director}_{m_idx+1}"
                    num_ics = random.randint(4, 7)
                    for ic_idx in range(num_ics):
                        ic = f"IC_{manager}_{ic_idx+1}"

                        # --- Latent IC strength drives NetNewACV
                        baseline_acv = np.random.uniform(80_000, 400_000)
                        q_acv = [baseline_acv * np.random.uniform(0.7, 1.4)
                                 for _ in range(4)]
                        # Q4 hockey stick
                        q_acv[3] *= np.random.uniform(1.1, 1.6)

                        # --- CloudSeats: positively correlated with ACV
                        # Roughly 1 cloud seat per $5K of ACV with noise
                        q_cloud = [max(0, int(q / 5000 * np.random.uniform(0.6, 1.4)))
                                   for q in q_acv]

                        # --- DCSeats: NEGATIVELY correlated with ACV
                        # Stronger DC presence => more legacy work, less new ACV
                        # We invert the latent strength to drive this metric.
                        dc_baseline = max(0, 60 - baseline_acv / 8000)
                        q_dc = [max(0, int(dc_baseline * np.random.uniform(0.5, 1.5)))
                                for _ in range(4)]

                        # --- ExpansionSpent (LTM): negatively correlated
                        # ICs who already spent a lot on expansion have less
                        # headroom for new ACV next quarter.
                        ltm_expansion = max(
                            0.0,
                            (500_000 - baseline_acv) * np.random.uniform(0.2, 1.5)
                        )

                        # Current pipeline (kept for backward-compat tests)
                        current_pipe = sum(q_acv) / 4 * np.random.uniform(1.5, 4.0)

                        rows.append({
                            'Global': 'Global_Corp',
                            'Region': region,
                            'RVP': rvp,
                            'Director': director,
                            'Manager': manager,
                            'IC': ic,
                            # NetNewACV (the cascade target)
                            'Q1_NetNewACV': round(q_acv[0], 2),
                            'Q2_NetNewACV': round(q_acv[1], 2),
                            'Q3_NetNewACV': round(q_acv[2], 2),
                            'Q4_NetNewACV': round(q_acv[3], 2),
                            # Proportional signal
                            'Q1_CloudSeats': q_cloud[0],
                            'Q2_CloudSeats': q_cloud[1],
                            'Q3_CloudSeats': q_cloud[2],
                            'Q4_CloudSeats': q_cloud[3],
                            # Inverse signal #1
                            'Q1_DCSeats': q_dc[0],
                            'Q2_DCSeats': q_dc[1],
                            'Q3_DCSeats': q_dc[2],
                            'Q4_DCSeats': q_dc[3],
                            # Inverse signal #2 (single LTM column)
                            'LTM_ExpansionSpent': round(ltm_expansion, 2),
                            # For PipelineAdjuster compatibility
                            'Current_Pipeline': round(current_pipe, 2),
                            # Aggregated convenience columns for suggest_from_data()
                            'NetNewACV_4Q_sum': round(sum(q_acv), 2),
                            'CloudSeats_4Q_sum': sum(q_cloud),
                            'DCSeats_4Q_sum': sum(q_dc),
                        })

    df = pd.DataFrame(rows)
    os.makedirs('tests/data', exist_ok=True)
    out = 'tests/data/synthetic_multi_metric.csv'
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} ICs with multi-metric history at {out}")
    return df


if __name__ == '__main__':
    generate_multi_metric_hierarchy()
