"""
Tests for issue #33 — on_degenerate: small/low-variance slices must not
silently equal-split.

Covers:
  - the issue's exact France-South repro: default 'proportional' keeps
    declared weights; the cascade splits ~70/30 instead of 50/50
  - single-metric proportional: 6x seat difference -> ~6x share
  - 'raise' mode; invalid mode
  - missing column is NOT degenerate (stays 0 in every mode, no raise)
  - report carries degenerate/fallback fields in all paths
  - cascade_many(weights_mode='per_group') on tiny combos: no equal split
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy,
    QuotaCascader,
    MetricSpec,
    cascade_many,
)

SEPARATOR = "=" * 90

FRANCE = pd.DataFrame({
    'node_4_team': ['France-South1', 'France-South2'],
    'knowledge_workers': [1_453_360, 1_213_025],   # ~20% apart
    'cloud_seats':       [326_220, 56_070],        # ~6x apart
})
FRANCE_CANDS = [
    {'name': 'knowledge_workers', 'column': 'knowledge_workers',
     'direction': 'proportional'},
    {'name': 'cloud_seats', 'column': 'cloud_seats',
     'direction': 'proportional'},
]


def _france_suggest(**kw):
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter('always')
        specs, report = MetricSpec.suggest_weights(
            FRANCE, target_column='knowledge_workers',
            candidate_metrics=FRANCE_CANDS, **kw)
    return specs, report, wlog


# ----------------------------------------------------------------------
# 1. Issue #33 repro — default keeps declared weights, cascade ~70/30
# ----------------------------------------------------------------------
def test_issue33_repro_proportional_default():
    print(SEPARATOR)
    print("TEST 1: France-South repro — declared weights kept; cascade "
          "splits ~70/30, NOT 50/50")
    print(SEPARATOR)
    specs, report, wlog = _france_suggest()
    weights = {s.name: s.weight for s in specs}
    print(f"  weights: {weights}")
    assert weights == {'knowledge_workers': 1.0, 'cloud_seats': 1.0}
    assert all(report[n]['degenerate'] for n in weights)
    assert all(report[n]['fallback'] == 'proportional' for n in weights)
    # Loud but not the equal-split warning
    assert any('statistically undefined' in str(w.message) for w in wlog)
    assert not any('ALL suggested weights' in str(w.message) for w in wlog)

    # End-to-end: cascade the two teams with the suggested specs
    df = FRANCE.assign(regional='France')
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['regional', 'node_4_team'],
                     metrics_cols=['knowledge_workers', 'cloud_seats'])
    c = QuotaCascader(h)
    q = c.cascade_quota('France', 1_000_000.0, metrics=specs, verbose=False)
    s1, s2 = q['France-South1'], q['France-South2']
    # Blend: kw shares .5451/.4549, seats shares .8534/.1466 -> ~.699/.301
    print(f"  South1: ${s1:,.2f} · South2: ${s2:,.2f} "
          f"(pre-fix: both $500,000.00)")
    assert abs(s1 / 1_000_000.0 - 0.699) < 0.005
    assert abs(s2 / 1_000_000.0 - 0.301) < 0.005


# ----------------------------------------------------------------------
# 2. Single metric: 6x seats -> ~6x share
# ----------------------------------------------------------------------
def test_single_metric_proportional():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: single degenerate candidate — 6x seat base -> ~6x share")
    print(SEPARATOR)
    specs, _, _ = _france_suggest()
    seats_only = [s for s in specs if s.name == 'cloud_seats']
    df = FRANCE.assign(regional='France')
    h = SalesHierarchy()
    h.from_dataframe(df, path_cols=['regional', 'node_4_team'],
                     metrics_cols=['knowledge_workers', 'cloud_seats'])
    c = QuotaCascader(h)
    q = c.cascade_quota('France', 1_000_000.0, metrics=seats_only,
                        verbose=False)
    ratio = q['France-South1'] / q['France-South2']
    print(f"  share ratio: {ratio:.2f}x (seat ratio {326_220/56_070:.2f}x)")
    assert abs(ratio - 326_220 / 56_070) < 0.01


# ----------------------------------------------------------------------
# 3. 'raise' mode + invalid mode
# ----------------------------------------------------------------------
def test_raise_and_invalid_modes():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: on_degenerate='raise' raises; invalid mode rejected")
    print(SEPARATOR)
    try:
        _france_suggest(on_degenerate='raise')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'statistically undefined' in str(e)
        print(f"  raise mode: {str(e)[:80]}...")
    try:
        _france_suggest(on_degenerate='panic')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'on_degenerate' in str(e)
        print(f"  invalid mode: {str(e)[:70]}...")


# ----------------------------------------------------------------------
# 4. Missing column is absent data, not thin data
# ----------------------------------------------------------------------
def test_missing_column_not_degenerate():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: missing column stays weight 0 in every mode (and never "
          "raises under 'raise')")
    print(SEPARATOR)
    df = pd.DataFrame({'kw': [1, 2, 3, 4], 'dc': [1, 2, 3, 4]})
    for mode in ['proportional', 'equal', 'raise']:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            specs, report = MetricSpec.suggest_weights(
                df, target_column='kw',
                candidate_metrics=[
                    {'name': 'dc', 'column': 'dc', 'direction': 'proportional'},
                    {'name': 'ghost', 'column': 'ghost',
                     'direction': 'proportional'},
                ],
                on_degenerate=mode)
        w = {s.name: s.weight for s in specs}
        assert w['ghost'] == 0.0 and w['dc'] > 0, mode
        assert report['ghost']['degenerate'] is False
        print(f"  {mode:>12}: ghost=0.0 (not degenerate), dc={w['dc']:.3f}")


# ----------------------------------------------------------------------
# 5. cascade_many per_group on tiny combos — the field scenario
# ----------------------------------------------------------------------
def test_cascade_many_per_group_tiny_slices():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: cascade_many(weights_mode='per_group') with 2-team "
          "combos — proportional, not equal")
    print(SEPARATOR)
    rows = []
    for reg, seats in [('France', [326_220, 56_070]),
                       ('Iberia', [10_000, 90_000])]:
        for i, s in enumerate(seats):
            rows.append(dict(regional=reg, team=f'{reg}-T{i+1}',
                             rep=f'{reg}-T{i+1}-r', knowledge_workers=100,
                             cloud_seats=s))
    hdf = pd.DataFrame(rows)
    targets = pd.DataFrame([dict(regional='France', q=1_000_000.0),
                            dict(regional='Iberia', q=1_000_000.0)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        quotas, weights = cascade_many(
            hdf, targets, group_keys=['regional'], target_col='q',
            taxonomy=['regional', 'team', 'rep'],
            suggest_config=dict(
                target_column='cloud_seats',
                candidate_metrics=[{'name': 'cloud_seats',
                                    'column': 'cloud_seats',
                                    'direction': 'proportional'}],
            ),
            weights_mode='per_group',
        )
    fr = quotas[(quotas.regional == 'France') & quotas.is_leaf]
    ratio = (fr[fr.node_id == 'France-T1-r']['cascaded_quota'].iloc[0]
             / fr[fr.node_id == 'France-T2-r']['cascaded_quota'].iloc[0])
    ib = quotas[(quotas.regional == 'Iberia') & quotas.is_leaf]
    ib_t2 = ib[ib.node_id == 'Iberia-T2-r']['cascaded_quota'].iloc[0]
    print(f"  France ratio: {ratio:.2f}x (seats 5.82x) · "
          f"Iberia T2: ${ib_t2:,.2f} (expected $900,000)")
    assert abs(ratio - 326_220 / 56_070) < 0.01     # NOT 1.0 (equal)
    assert abs(ib_t2 - 900_000.0) < 0.01
    # per-combo weights table shows the kept declared weight
    assert (weights['input_weight'] == 1.0).all()


if __name__ == '__main__':
    test_issue33_repro_proportional_default()
    test_single_metric_proportional()
    test_raise_and_invalid_modes()
    test_missing_column_not_degenerate()
    test_cascade_many_per_group_tiny_slices()

    print(f"\n\n{SEPARATOR}")
    print("ALL DEGENERATE-FALLBACK TESTS PASSED")
    print(SEPARATOR)
