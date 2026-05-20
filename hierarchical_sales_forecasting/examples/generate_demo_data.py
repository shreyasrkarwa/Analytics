"""
Generate a realistic synthetic dataset that exercises every feature of
b2b_revenue_forecasting (v0.3+).

Produces TWO CSVs in examples/data/:

  1. demo_hierarchy.csv  — one row per IC. Columns:

        Hierarchy:
          Global, Region, RVP, Director, Manager, IC

        Historical attainment metrics (4 quarters each):
          Q1..Q4_NetNewACV     — primary signal, "proportional" (the target)
          Q1..Q4_CloudSeats    — secondary signal, "proportional"
          Q1..Q4_DCSeats       — secondary signal, "inverse"

        Historical attainment metric (single LTM column):
          LTM_ExpansionSpent   — secondary signal, "inverse"

        Boolean metric (4 quarters):
          Q1..Q4_Has_Active_Cert — "proportional" (1 if certified that Q)

        Multi-source open pipeline (for PipelineAdjuster):
          Open_Pipeline, Late_Stage_Commit, Best_Case_Adds

        Forward-looking columns (for white-space-planning cascade):
          Current_Seats_ProductX     — current installed-base, proportional
          Knowledge_Workers_Count    — total knowledge workers in territory
          Unmigrated_Seats           — gate metric: 0 => no migration target

        Brand-new flag (for QuotaCascader new_ic_attr):
          Is_Brand_New         — 'yes' / 'no'

        Pre-aggregated convenience columns (for MetricSpec.suggest_weights):
          NetNewACV_4Q_sum, CloudSeats_4Q_sum, DCSeats_4Q_sum

     Includes intentional variety:
       - Mostly veteran ICs with full 4Q history
       - A handful of partial-history ICs (joined in Q3 / Q4)
       - A handful of brand-new ICs (zero historical attainment)
       - A few ICs flagged Is_Brand_New=True despite having some history
         (e.g., transferred-in reps the analyst wants treated as new)

  2. demo_commit_history.csv — manager-level historical commits and actuals,
     for CommitReconciler. Columns:

        Manager_ID, Historical_Commit, Historical_Actual_Closed

     Includes a sandbagger, a happy-ears optimist, and a truth-teller.
"""
import os
import random
import numpy as np
import pandas as pd

DEMO_DATA_DIR = 'examples/data'


def generate_hierarchy_csv():
    np.random.seed(11)
    random.seed(11)

    regions = ['NA', 'EMEA', 'APAC']
    rows = []
    flagged_count = 0

    for region in regions:
        num_rvps = {'NA': 2, 'EMEA': 1, 'APAC': 1}[region]
        for rvp_idx in range(1, num_rvps + 1):
            rvp = f'RVP_{region}_{rvp_idx}'
            num_dirs = random.randint(2, 3)
            for d_idx in range(1, num_dirs + 1):
                director = f'Dir_{rvp}_{d_idx}'
                num_mgrs = random.randint(2, 4)
                for m_idx in range(1, num_mgrs + 1):
                    manager = f'Mgr_{director}_{m_idx}'
                    num_ics = random.randint(4, 7)
                    for ic_idx in range(1, num_ics + 1):
                        ic = f'IC_{manager}_{ic_idx}'

                        # --- Decide the IC's tenure profile
                        roll = random.random()
                        if roll < 0.80:
                            tenure = 'veteran'        # full 4Q history
                        elif roll < 0.90:
                            tenure = 'partial'        # joined in Q3 or Q4
                        else:
                            tenure = 'brand_new'      # all zeros

                        # --- Generate Net New ACV by quarter
                        baseline_acv = np.random.uniform(80_000, 400_000)
                        q_acv = [baseline_acv * np.random.uniform(0.7, 1.4)
                                 for _ in range(4)]
                        q_acv[3] *= np.random.uniform(1.1, 1.6)  # Q4 hockey stick

                        if tenure == 'partial':
                            # First two quarters are zero (joined mid-year)
                            join_quarter = random.choice([2, 3])  # joined Q3 or Q4
                            q_acv = [0.0] * join_quarter + q_acv[join_quarter:]
                        elif tenure == 'brand_new':
                            q_acv = [0.0, 0.0, 0.0, 0.0]
                            baseline_acv = 0.0

                        # --- CloudSeats: positively correlated with ACV
                        q_cloud = []
                        for q in q_acv:
                            if q == 0:
                                q_cloud.append(0)
                            else:
                                q_cloud.append(max(0, int(q / 5000 * np.random.uniform(0.6, 1.4))))

                        # --- DCSeats: negatively correlated with ACV
                        dc_baseline = max(0, 60 - baseline_acv / 8000) if tenure != 'brand_new' else 0
                        q_dc = []
                        for q in q_acv:
                            if q == 0 and tenure == 'brand_new':
                                q_dc.append(0)
                            else:
                                q_dc.append(max(0, int(dc_baseline * np.random.uniform(0.5, 1.5))))

                        # --- ExpansionSpent (LTM): negatively correlated
                        if tenure == 'brand_new':
                            ltm_expansion = 0.0
                        else:
                            ltm_expansion = max(0.0, (500_000 - baseline_acv) *
                                                     np.random.uniform(0.2, 1.5))

                        # --- Has_Active_Cert (boolean per quarter)
                        # Veterans more likely to be certified; certification
                        # itself correlates with attainment.
                        cert_prob = 0.7 if tenure == 'veteran' else 0.3
                        q_cert = []
                        for q_idx, q in enumerate(q_acv):
                            if q == 0 and tenure != 'veteran':
                                q_cert.append(False)
                            else:
                                # Once certified, tend to stay certified
                                if q_cert and q_cert[-1]:
                                    q_cert.append(random.random() < 0.95)
                                else:
                                    q_cert.append(random.random() < cert_prob)

                        # --- Multi-source open pipeline (Q1 forward-looking)
                        avg_acv = sum(q_acv) / 4 if any(q_acv) else 50_000
                        open_pipe = avg_acv * np.random.uniform(1.5, 3.5)
                        late_stage = avg_acv * np.random.uniform(0.3, 0.9)
                        best_case = avg_acv * np.random.uniform(0.1, 0.6)

                        # --- Forward-looking columns (white-space planning)
                        # Current installed seats of Product X (proportional)
                        current_seats = max(0, int(np.random.uniform(20, 500)))
                        # Total knowledge workers in the territory
                        knowledge_workers = max(current_seats + 50,
                                                int(np.random.uniform(200, 5000)))
                        # Unmigrated seats — the gate metric. ~12% of ICs
                        # are fully migrated (0 unmigrated => gate fails).
                        # For an extra demo touch, the entire team under
                        # Mgr_Dir_RVP_APAC_1_1_1 is fully migrated so we
                        # can show whole-subtree gating propagation.
                        force_zero = (manager == 'Mgr_Dir_RVP_APAC_1_1_1'
                                      or np.random.random() < 0.12)
                        if force_zero:
                            unmigrated_seats = 0
                        else:
                            unmigrated_seats = max(1, int(np.random.uniform(5, 200)))

                        # --- Is_Brand_New flag
                        if tenure == 'brand_new':
                            is_brand_new = 'yes'
                        elif tenure == 'partial' and random.random() < 0.3:
                            # 30% of partial-history reps are explicitly
                            # flagged as new by the analyst (e.g., the rep
                            # had a slow ramp and shouldn't be benchmarked
                            # against historical attainment)
                            is_brand_new = 'yes'
                            flagged_count += 1
                        else:
                            is_brand_new = 'no'

                        rows.append({
                            'Global': 'Global_Corp',
                            'Region': region,
                            'RVP': rvp,
                            'Director': director,
                            'Manager': manager,
                            'IC': ic,
                            # NetNewACV by quarter (primary historical signal)
                            'Q1_NetNewACV': round(q_acv[0], 2),
                            'Q2_NetNewACV': round(q_acv[1], 2),
                            'Q3_NetNewACV': round(q_acv[2], 2),
                            'Q4_NetNewACV': round(q_acv[3], 2),
                            # CloudSeats by quarter (proportional secondary)
                            'Q1_CloudSeats': q_cloud[0],
                            'Q2_CloudSeats': q_cloud[1],
                            'Q3_CloudSeats': q_cloud[2],
                            'Q4_CloudSeats': q_cloud[3],
                            # DCSeats by quarter (inverse secondary)
                            'Q1_DCSeats': q_dc[0],
                            'Q2_DCSeats': q_dc[1],
                            'Q3_DCSeats': q_dc[2],
                            'Q4_DCSeats': q_dc[3],
                            # Single-column LTM metric (inverse secondary)
                            'LTM_ExpansionSpent': round(ltm_expansion, 2),
                            # Boolean cert per quarter (proportional secondary)
                            'Q1_Has_Active_Cert': q_cert[0],
                            'Q2_Has_Active_Cert': q_cert[1],
                            'Q3_Has_Active_Cert': q_cert[2],
                            'Q4_Has_Active_Cert': q_cert[3],
                            # Multi-source pipeline (for PipelineAdjuster)
                            'Open_Pipeline':     round(open_pipe, 2),
                            'Late_Stage_Commit': round(late_stage, 2),
                            'Best_Case_Adds':    round(best_case, 2),
                            # Forward-looking columns (white-space planning)
                            'Current_Seats_ProductX':  current_seats,
                            'Knowledge_Workers_Count': knowledge_workers,
                            'Unmigrated_Seats':        unmigrated_seats,
                            # Brand-new flag (for cascade_quota new_ic_attr)
                            'Is_Brand_New': is_brand_new,
                            # Pre-aggregated convenience columns for
                            # MetricSpec.suggest_weights (one row per IC)
                            'NetNewACV_4Q_sum':   round(sum(q_acv), 2),
                            'CloudSeats_4Q_sum':  sum(q_cloud),
                            'DCSeats_4Q_sum':     sum(q_dc),
                        })

    df = pd.DataFrame(rows)
    os.makedirs(DEMO_DATA_DIR, exist_ok=True)
    out = os.path.join(DEMO_DATA_DIR, 'demo_hierarchy.csv')
    df.to_csv(out, index=False)

    # Tenure breakdown for sanity
    n_total = len(df)
    n_zero_acv = int((df['NetNewACV_4Q_sum'] == 0).sum())
    n_flagged = int((df['Is_Brand_New'] == 'yes').sum())
    n_zero_unmigrated = int((df['Unmigrated_Seats'] == 0).sum())
    print(f"Hierarchy CSV: {out}")
    print(f"  Total ICs: {n_total}")
    print(f"  ICs with zero 4Q NetNewACV: {n_zero_acv}")
    print(f"  ICs flagged Is_Brand_New=yes: {n_flagged}")
    print(f"  ICs with zero Unmigrated_Seats (will be gated): {n_zero_unmigrated}")
    return df


def generate_commit_history_csv():
    """Manager-level commit vs actual history for CommitReconciler."""
    rows = [
        # Sandbagger: closes ~1.4x what they commit
        {'Manager_ID': 'Mgr_Dir_RVP_NA_1_1_1',
         'Historical_Commit':        400_000, 'Historical_Actual_Closed': 560_000},
        {'Manager_ID': 'Mgr_Dir_RVP_NA_1_1_1',
         'Historical_Commit':        500_000, 'Historical_Actual_Closed': 700_000},
        # Truth-teller: closes ~1.0x
        {'Manager_ID': 'Mgr_Dir_RVP_NA_1_1_2',
         'Historical_Commit':        300_000, 'Historical_Actual_Closed': 305_000},
        {'Manager_ID': 'Mgr_Dir_RVP_NA_1_1_2',
         'Historical_Commit':        400_000, 'Historical_Actual_Closed': 395_000},
        # Happy ears: closes only ~0.7x
        {'Manager_ID': 'Mgr_Dir_RVP_EMEA_1_1_1',
         'Historical_Commit':        500_000, 'Historical_Actual_Closed': 360_000},
        {'Manager_ID': 'Mgr_Dir_RVP_EMEA_1_1_1',
         'Historical_Commit':        600_000, 'Historical_Actual_Closed': 410_000},
    ]
    df = pd.DataFrame(rows)
    out = os.path.join(DEMO_DATA_DIR, 'demo_commit_history.csv')
    df.to_csv(out, index=False)
    print(f"Commit history CSV: {out}")
    print(f"  Managers profiled: {df['Manager_ID'].nunique()}")
    return df


if __name__ == '__main__':
    generate_hierarchy_csv()
    generate_commit_history_csv()
