"""
Tests for issues #50 / #38 — weights_long as the authoritative record
+ per-node share_of_parent.

Covers:
  - weights_long: blend rows + gate rows (role, gate_threshold,
    gate_mode), legacy '_Attainment' combos no longer silently absent,
    weights_source provenance, per-metric degenerate flag
  - share_of_parent: root=1.0, sums to 1 per sibling group (base
    layer), gated node shows 0, present in single-cascade
    quotas_to_dataframe output too
  - the #38 decomposition recipe: share_of_parent vs metric-subtree
    share agree for a single-metric cascade
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    SalesHierarchy, QuotaCascader, cascade_many, MetricSpec,
    rollup_metrics,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=2.0,
                 columns=['kw']),
      MetricSpec('seats', direction='inverse', weight=1.0,
                 columns=['seats'])]
GATE = [MetricSpec('seats', columns=['seats'], gate_threshold=15.0)]


def _hdf():
    return pd.DataFrame([
        dict(region=rg, team=f'{rg}_T{i//2+1}', rep=f'{rg}_r{i+1}',
             kw=[100, 220, 310, 450][i], seats=[10, 20, 30, 40][i],
             Q1_Attainment=1.0)
        for rg in ('EMEA', 'APAC') for i in range(4)])


# ----------------------------------------------------------------------
# 1. weights_long: roles, gates, provenance
# ----------------------------------------------------------------------
def test_weights_long_roles_and_gates():
    print(SEPARATOR)
    print("TEST 1: blend + gate rows, per combo, with provenance")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                            dict(region='APAC', tgt=900_000.0)])
    _, w = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=KW, gate_metrics=GATE)
    print(w.to_string(index=False))
    emea = w[w.region == 'EMEA']
    blend = emea[emea.role == 'blend'].set_index('metric')
    assert abs(blend.loc['kw', 'normalized_share'] - 2 / 3) < 1e-9
    gate = emea[emea.role == 'gate'].iloc[0]
    assert gate['metric'] == 'seats'
    assert gate['gate_threshold'] == 15.0 and gate['gate_mode'] == 'gt'
    assert (w['weights_source'] == 'fixed').all()
    assert set(w.region) == {'EMEA', 'APAC'}


# ----------------------------------------------------------------------
# 2. Legacy combos recorded; degenerate flag on suggested slates
# ----------------------------------------------------------------------
def test_legacy_and_degenerate():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: '_Attainment' row for legacy combos; degenerate flag "
          "from suggest")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    _, w = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'])
    assert len(w) == 1 and w.iloc[0]['metric'] == '_Attainment'
    assert w.iloc[0]['weights_source'] == 'default_attainment'
    # degenerate: constant metric under per_group suggestion
    hdf = _hdf()
    hdf['flat'] = 5.0
    hdf['perf'] = hdf['kw'] * 2 + 1
    _, w2 = cascade_many(
        hdf, targets, group_keys=['region'], target_col='tgt',
        taxonomy=['region', 'team', 'rep'],
        suggest_config=dict(target_column='perf',
                            candidate_metrics=[
                                MetricSpec('kw', columns=['kw']),
                                MetricSpec('flat', columns=['flat'])]),
        weights_mode='per_group')
    d = w2.set_index('metric')['degenerate']
    print(f"  degenerate flags: {d.to_dict()}")
    assert bool(d.loc['flat']) and not bool(d.loc['kw'])
    assert (w2['weights_source'] == 'suggested_per_group').all()


# ----------------------------------------------------------------------
# 3. share_of_parent: root, sibling sums, gated zero; single-cascade too
# ----------------------------------------------------------------------
def test_share_of_parent():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: share_of_parent — root 1.0, sibling groups sum to 1, "
          "gated rep = 0")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=KW, gate_metrics=GATE)
    ix = q.set_index('node_id')
    assert ix.loc['EMEA', 'share_of_parent'] == 1.0
    for parent in ('EMEA', 'EMEA_T1', 'EMEA_T2'):
        kids = q[q.parent == parent]
        assert abs(kids['share_of_parent'].sum() - 1.0) < 1e-4, parent
    assert bool(ix.loc['EMEA_r1', 'is_gated'])          # seats=10 <= 15
    assert ix.loc['EMEA_r1', 'share_of_parent'] == 0.0
    # single-cascade path (quotas_to_dataframe) carries it too
    h = SalesHierarchy()
    h.from_dataframe(_hdf()[lambda d: d.region == 'EMEA'],
                     path_cols=['region', 'team', 'rep'],
                     metrics_cols=['kw', 'seats'])
    c = QuotaCascader(h)
    quotas = c.cascade_quota('EMEA', 500_000.0, metrics=KW,
                             verbose=False)
    out = c.quotas_to_dataframe(quotas, unhedged_quotas='auto')
    assert 'share_of_parent' in out.columns
    assert out.set_index('node_id').loc['EMEA', 'share_of_parent'] == 1.0
    print("  root 1.0; all sibling groups sum to 1; gated rep 0; "
          "single-cascade output included")


# ----------------------------------------------------------------------
# 4. The #38 decomposition: quota share == metric share (single metric)
# ----------------------------------------------------------------------
def test_share_decomposition_recipe():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: share_of_parent == kw_subtree share for a pure-kw "
          "cascade (the debugging recipe)")
    print(SEPARATOR)
    targets = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    q, _ = cascade_many(_hdf(), targets, group_keys=['region'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'],
                        metrics=[MetricSpec('kw',
                                            direction='proportional',
                                            weight=1.0, columns=['kw'])],
                        metadata_cols=['kw'], attach_metrics=True)
    ix = q.set_index('node_id')
    for node, parent in [('EMEA_T1', 'EMEA'), ('EMEA_r3', 'EMEA_T2')]:
        metric_share = (ix.loc[node, 'kw_subtree']
                        / ix.loc[parent, 'kw_subtree'])
        assert abs(ix.loc[node, 'share_of_parent'] - metric_share) < 1e-4
    print("  quota shares equal metric-subtree shares — 'why' answered "
          "in two columns")


if __name__ == '__main__':
    test_weights_long_roles_and_gates()
    test_legacy_and_degenerate()
    test_share_of_parent()
    test_share_decomposition_recipe()

    print(f"\n\n{SEPARATOR}")
    print("ALL WEIGHTS-EXPLAINABILITY TESTS PASSED")
    print(SEPARATOR)
