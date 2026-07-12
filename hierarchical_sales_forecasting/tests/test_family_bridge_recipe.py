"""
Pinned recipe for issue #27 — family/edition gate-bridge (Cloud<->DC).

The package deliberately does NOT ship a ProductFamily/edition_map
utility (the #36 precedent: catalog knowledge and the sum-vs-max
entitlement choice are source-data semantics that belong to the
consumer). What it DOES own — conditional gates per combination — is
what this recipe composes with. This test keeps the documented recipe
working forever.

Covers:
  - single-counterpart bridge: Cloud rep inherits its DC counterpart's
    seats; rep without any is gated at $0
  - multi-counterpart bridge (Teamwork -> Jira DC + Confluence DC,
    summed): inheritance through EITHER counterpart passes the gate
  - non-family combos are untouched (no gate applied)
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec

SEPARATOR = "=" * 90


def test_family_bridge_recipe():
    print(SEPARATOR)
    print("TEST: the #27 recipe — 6-line bridge + v0.15.0 conditional "
          "gates")
    print(SEPARATOR)
    df = pd.DataFrame([
        ('Jira Cloud',          'EMEA', 'T1', 'r1', 100, 0),
        ('Jira Cloud',          'EMEA', 'T1', 'r2', 200, 0),
        ('Jira DC',             'EMEA', 'T1', 'r1',  50, 9),
        ('Jira DC',             'EMEA', 'T1', 'r2',  60, 0),
        ('Teamwork Collection', 'EMEA', 'T1', 'r1',  80, 0),
        ('Teamwork Collection', 'EMEA', 'T1', 'r2',  90, 0),
        ('Confluence DC',       'EMEA', 'T1', 'r2',  40, 7),
    ], columns=['product', 'region', 'team', 'rep', 'kw', 'dc_seats'])

    # ---- the recipe: consumer-owned map + one lookup ----------------
    FAMILY = {'Jira Cloud': ['Jira DC'],
              'Teamwork Collection': ['Jira DC', 'Confluence DC']}
    seats = df.set_index(['rep', 'product'])['dc_seats']
    df['family_dc_seats'] = df.apply(
        lambda r: sum(seats.get((r['rep'], cp), 0)   # sum? max? YOUR call
                      for cp in FAMILY.get(r['product'], [])), axis=1)

    targets = pd.DataFrame([dict(product=p, tgt=1_000_000.0)
                            for p in ('Jira Cloud', 'Teamwork Collection',
                                      'Jira DC')])
    gates = lambda g: ([MetricSpec('family_dc_seats',
                                   columns=['family_dc_seats'],
                                   gate_threshold=1.0)]
                       if g['product'] in FAMILY else None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q, _ = cascade_many(
            df, targets, group_keys=['product'], target_col='tgt',
            taxonomy=['region', 'team', 'rep'],
            metrics=[MetricSpec('kw', direction='proportional',
                                weight=1.0, columns=['kw'])],
            gate_metrics=gates)
    ix = q.set_index(['product', 'node_id'])

    # Cloud edition inherits its DC counterpart's entitlement per rep
    assert bool(ix.loc[('Jira Cloud', 'r1'), 'is_gated']) is False
    assert abs(ix.loc[('Jira Cloud', 'r1'), 'base_quota']
               - 1_000_000.0) < 0.05          # sole survivor
    assert bool(ix.loc[('Jira Cloud', 'r2'), 'is_gated']) is True
    assert ix.loc[('Jira Cloud', 'r2'), 'base_quota'] == 0.0

    # Multi-counterpart: r2 inherits via Confluence DC (7 seats)
    tw = q[q['product'] == 'Teamwork Collection'].set_index('node_id')
    assert (tw['is_gated'].fillna(False) == False).all()  # noqa: E712
    assert abs(tw.loc['r2', 'base_quota']
               - 1_000_000.0 * 90 / 170) < 0.5

    # Non-family combo (Jira DC itself): no gate applied
    jd = q[q['product'] == 'Jira DC'].set_index('node_id')
    assert abs(jd.loc['r1', 'base_quota']
               - 1_000_000.0 * 50 / 110) < 0.5
    print("  r1 inherits Jira DC seats; r2 gated at $0; Teamwork "
          "bridges via Confluence DC; DC combo ungated")


if __name__ == '__main__':
    test_family_bridge_recipe()
    print(f"\n\n{SEPARATOR}")
    print("FAMILY-BRIDGE RECIPE PINNED")
    print(SEPARATOR)
