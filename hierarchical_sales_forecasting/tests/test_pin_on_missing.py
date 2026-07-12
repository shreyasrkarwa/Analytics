"""
Tests for issue #48 — apply_pins(on_missing='error'|'skip'|'warn').

Covers:
  - default 'error' preserves historic behavior (message now points at
    on_missing)
  - 'skip': mixed batch — valid pins land; skipped pins recorded with
    reason='node_absent' | 'empty_scope', in INPUT order
  - 'warn': one summary warning naming the skipped pins
  - no ghost side-effects: a node whose pin was skipped still absorbs
    for other pins (frame identical to a batch without the vacuous pin)
  - all pins skipped -> values untouched
  - invalid on_missing rejected; missing scope COLUMN always raises
"""
import warnings
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import cascade_many, MetricSpec, Pin, apply_pins

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _quotas():
    hdf = pd.DataFrame([dict(st='NN', region='EMEA', team=f'T{i//2+1}',
                             rep=f'r{i+1}', kw=[100, 200, 300, 400][i])
                        for i in range(4)])
    targets = pd.DataFrame([dict(st='NN', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    q, _ = cascade_many(hdf, targets, group_keys=['st'],
                        target_col='tgt',
                        taxonomy=['region', 'team', 'rep'], metrics=KW)
    return q


BATCH = lambda: [Pin('T1', 900_000.0),
                 Pin('GHOST', 100.0),                            # absent
                 Pin('T2', 100.0, scope={'fiscal_quarter': 9}),  # empty
                 Pin('r3', 400_000.0)]


# ----------------------------------------------------------------------
# 1. Default 'error' unchanged
# ----------------------------------------------------------------------
def test_default_error():
    print(SEPARATOR)
    print("TEST 1: default still raises; message points at on_missing")
    print(SEPARATOR)
    q = _quotas()
    try:
        apply_pins(q, BATCH())
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'GHOST' in str(e) and "on_missing='skip'" in str(e)
    print("  raises as before, with the new hint")


# ----------------------------------------------------------------------
# 2. 'skip': valid pins land, skipped pins recorded in input order
# ----------------------------------------------------------------------
def test_skip_records_reasons():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: 'skip' — T1/r3 land; GHOST/T2 recorded with reasons")
    print(SEPARATOR)
    q = _quotas()
    e, rep = apply_pins(q, BATCH(), on_missing='skip')
    ix = e.set_index(['node_id', 'fiscal_quarter'])
    assert abs(ix.loc['T1', 'base_quota'].sum() - 900_000.0) < 0.05
    assert abs(ix.loc['r3', 'base_quota'].sum() - 400_000.0) < 0.05
    assert list(rep['pin_node']) == ['T1', 'GHOST', 'T2', 'r3']  # input order
    r = rep.set_index('pin_node')
    assert bool(r.loc['GHOST', 'skipped'])
    assert r.loc['GHOST', 'reason'] == 'node_absent'
    assert bool(r.loc['T2', 'skipped'])
    assert r.loc['T2', 'reason'] == 'empty_scope'
    assert not bool(r.loc['GHOST', 'feasible'])
    # pd.isna, not `is None`: newer pandas (py3.11/3.12 CI) coerces
    # None -> NaN in object columns when building the report frame.
    assert not bool(r.loc['T1', 'skipped'])
    assert pd.isna(r.loc['T1', 'reason'])
    assert bool(r.loc['T1', 'feasible']) and bool(r.loc['r3', 'feasible'])
    print(f"  report: {list(zip(rep['pin_node'], rep['skipped'], rep['reason']))}")


# ----------------------------------------------------------------------
# 3. 'warn': one summary warning naming the pins
# ----------------------------------------------------------------------
def test_warn_summary():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: 'warn' — single summary warning")
    print(SEPARATOR)
    q = _quotas()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        _, rep = apply_pins(q, BATCH(), on_missing='warn')
    hits = [x for x in w if 'skipped 2 pin(s)' in str(x.message)]
    assert len(hits) == 1
    msg = str(hits[0].message)
    assert 'GHOST (node_absent)' in msg and 'T2 (empty_scope)' in msg
    assert rep['skipped'].sum() == 2
    print(f"  {msg[:90]}...")


# ----------------------------------------------------------------------
# 4. No ghost side-effects: skipped pin's node still absorbs
# ----------------------------------------------------------------------
def test_no_ghost_protection():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: T2's empty-scope pin skipped -> T2 still absorbs "
          "T1's delta (identical to batch without the vacuous pin)")
    print(SEPARATOR)
    q = _quotas()
    with_ghost, _ = apply_pins(
        q, [Pin('T1', 900_000.0),
            Pin('T2', 100.0, scope={'fiscal_quarter': 9})],
        on_missing='skip')
    without, _ = apply_pins(q, [Pin('T1', 900_000.0)])
    a = with_ghost.set_index(['node_id', 'fiscal_quarter']) \
                  .sort_index()['base_quota'].round(2)
    b = without.set_index(['node_id', 'fiscal_quarter']) \
               .sort_index()['base_quota'].round(2)
    assert a.equals(b)
    # T2 genuinely absorbed (shrank from baseline 1.4M)
    assert a.loc['T2'].sum() < 1_400_000.0 - 0.05
    print("  frames identical; T2 absorbed normally")


# ----------------------------------------------------------------------
# 5. All pins skipped -> values untouched; edge validation
# ----------------------------------------------------------------------
def test_all_skipped_and_validation():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: all-skipped no-op; bad on_missing; missing scope "
          "COLUMN always raises")
    print(SEPARATOR)
    q = _quotas()
    e, rep = apply_pins(q, [Pin('GHOST', 100.0)], on_missing='skip')
    assert e.set_index(['node_id', 'fiscal_quarter'])['base_quota'].equals(
        q.set_index(['node_id', 'fiscal_quarter'])['base_quota'])
    assert rep['skipped'].all() and len(rep) == 1
    try:
        apply_pins(q, [Pin('T1', 1.0)], on_missing='ignore')
        raise AssertionError('expected ValueError')
    except ValueError as e2:
        assert 'on_missing' in str(e2)
    try:  # scope-column typo is a programming error even under 'skip'
        apply_pins(q, [Pin('T1', 1.0, scope={'nope_col': 1})],
                   on_missing='skip')
        raise AssertionError('expected ValueError')
    except ValueError as e3:
        assert 'nope_col' in str(e3)
    print("  no-op exact; both validations raise")


if __name__ == '__main__':
    test_default_error()
    test_skip_records_reasons()
    test_warn_summary()
    test_no_ghost_protection()
    test_all_skipped_and_validation()

    print(f"\n\n{SEPARATOR}")
    print("ALL ON-MISSING TESTS PASSED")
    print(SEPARATOR)
