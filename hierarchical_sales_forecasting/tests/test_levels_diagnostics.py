"""
Tests for issue #56 — cascade_levels diagnostics parity: per-transition
weights_long / combo_report / id_map on the output's attrs, tagged with
transition + level.

Covers:
  - distinct fixed slates per transition visible in the reconstructed
    weights table (kw at t0, dc at t1), correctly tagged
  - a CALLABLE transition inherits v0.33.0 per-row provenance
    (sub-target tags + weights_source='policy')
  - combo_report: one record per (transition x parent combo);
    skipped combos carry reasons
  - id_map records on a collision, tagged with the transition
  - attrs stay concat-safe
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_levels, MetricSpec

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]
DC = [MetricSpec('dc', direction='proportional', weight=1.0,
                 columns=['dc'])]


def _hdf():
    return pd.DataFrame([
        dict(region='EMEA', rvp=f'V{i//2+1}', director=f'D{i+1}',
             kw=100 * (i + 1), dc=[9, 7, 5, 3][i]) for i in range(4)])


# ----------------------------------------------------------------------
# 1. Per-transition slates + combo records, tagged
# ----------------------------------------------------------------------
def test_per_transition_records():
    print(SEPARATOR)
    print("TEST 1: kw at t0, dc at t1 — attrs carry both, tagged")
    print(SEPARATOR)
    rt = pd.DataFrame([dict(region='EMEA', st1_sales_type='Migration',
                            tgt=1_000_000.0)])
    res = cascade_levels(_hdf(), rt,
                         taxonomy=['region', 'rvp', 'director'],
                         target_col='tgt',
                         level_kwargs=[dict(metrics=KW),
                                       dict(metrics=DC)])
    w = pd.DataFrame(res.attrs['weights_long'])
    print(w[['transition', 'level', 'metric', 'weights_source']]
          .to_string(index=False))
    t0 = w[w.transition == 0]
    t1 = w[w.transition == 1]
    assert set(t0['metric']) == {'kw'} and set(t1['metric']) == {'dc'}
    assert (t0['level'] == 'region').all()
    assert (t1['level'] == 'rvp').all()
    assert len(t1) == 2                       # one slate per V-parent
    rep = pd.DataFrame(res.attrs['combo_report'])
    assert len(rep[rep.transition == 0]) == 1     # EMEA
    assert len(rep[rep.transition == 1]) == 2     # V1, V2
    assert (rep['weights_source'] == 'fixed').all()
    assert not rep['skipped'].any()


# ----------------------------------------------------------------------
# 2. Callable transition inherits per-row provenance (v0.33.0)
# ----------------------------------------------------------------------
def test_callable_transition_provenance():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: policy at t1 -> weights records carry sub-target "
          "tags + source='policy'")
    print(SEPARATOR)
    rt = pd.DataFrame([dict(region='EMEA', st1_sales_type='Migration',
                            tgt=1_000_000.0)])
    pol = lambda g: (DC if g.get('st1_sales_type') == 'Migration'
                     else KW)
    res = cascade_levels(_hdf(), rt,
                         taxonomy=['region', 'rvp', 'director'],
                         target_col='tgt',
                         level_kwargs=[dict(metrics=KW),
                                       dict(metrics=pol)])
    w = pd.DataFrame(res.attrs['weights_long'])
    t1 = w[w.transition == 1]
    assert (t1['weights_source'] == 'policy').all()
    assert set(t1['metric']) == {'dc'}                # routed correctly
    assert (t1['st1_sales_type'] == 'Migration').all()  # per-row tag
    rep = pd.DataFrame(res.attrs['combo_report'])
    assert (rep[rep.transition == 1]['weights_source'] == 'policy').all()
    print("  t1 slates chosen by sub-target, recorded per row")


# ----------------------------------------------------------------------
# 3. Skipped combos carry reasons; id_map on collision
# ----------------------------------------------------------------------
def test_skips_and_id_map():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: GHOST root skipped with reason; collision id_map "
          "tagged")
    print(SEPARATOR)
    hdf = _hdf()
    # a director sharing its rvp's id -> collision rename at t1
    hdf.loc[0, 'director'] = 'V1'
    rt = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0),
                       dict(region='GHOST', tgt=500_000.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = cascade_levels(hdf, rt,
                             taxonomy=['region', 'rvp', 'director'],
                             target_col='tgt',
                             level_kwargs=[dict(metrics=KW),
                                           dict(metrics=KW)])
    rep = pd.DataFrame(res.attrs['combo_report'])
    ghost = rep[(rep.transition == 0) & (rep.region == 'GHOST')]
    assert len(ghost) == 1 and bool(ghost.iloc[0]['skipped'])
    assert 'no rows' in ghost.iloc[0]['reason']
    im = pd.DataFrame(res.attrs['id_map'])
    assert len(im) == 1
    assert im.iloc[0]['sanitized'] == 'V1__director'
    assert im.iloc[0]['original'] == 'V1'
    assert im.iloc[0]['transition'] == 1
    print(f"  ghost reason recorded; id_map: {im.iloc[0].to_dict()}")


# ----------------------------------------------------------------------
# 4. Concat safety
# ----------------------------------------------------------------------
def test_concat_safe():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: pd.concat of two outputs is fine")
    print(SEPARATOR)
    rt = pd.DataFrame([dict(region='EMEA', tgt=1_000_000.0)])
    res = cascade_levels(_hdf(), rt,
                         taxonomy=['region', 'rvp', 'director'],
                         target_col='tgt',
                         level_kwargs=[dict(metrics=KW),
                                       dict(metrics=KW)])
    both = pd.concat([res, res.copy()], ignore_index=True)
    assert len(both) == 2 * len(res)
    print("  concat ok")


if __name__ == '__main__':
    test_per_transition_records()
    test_callable_transition_provenance()
    test_skips_and_id_map()
    test_concat_safe()

    print(f"\n\n{SEPARATOR}")
    print("ALL LEVELS-DIAGNOSTICS TESTS PASSED")
    print(SEPARATOR)
