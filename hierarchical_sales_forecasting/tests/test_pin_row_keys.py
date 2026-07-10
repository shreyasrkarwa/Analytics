"""
Tests for issue #40 — cascade-key handling in apply_pins.

Before v0.21.0, apply_pins with no row_keys inferred the cascade key
from EVERY non-structural column; per-node columns (metadata_cols)
poisoned it, so parent and child rows landed in different key tuples,
manager pins silently became pin_type='leaf', and sibling absorption
mis-grouped. Covers:
  - cascade_many stamps .attrs['cascade_row_keys']
  - apply_pins uses the stamp: subtree pins Just Work with
    metadata_cols present and no row_keys
  - orphan guard: stripped attrs + poisoned inference -> ValueError
    naming the poison column and suggesting corrected keys
  - wrong EXPLICIT row_keys -> same guard
  - correct explicit row_keys still work (the filer's workaround)
  - leaf-only frames (no manager rows) still allowed
"""
import pandas as pd
import sys
sys.path.insert(0, '.')

from b2b_revenue_forecasting import (
    cascade_many, MetricSpec, Pin, apply_pins,
)

SEPARATOR = "=" * 90
KW = [MetricSpec('kw', direction='proportional', weight=1.0,
                 columns=['kw'])]


def _frames():
    hdf = pd.DataFrame([
        dict(product='cloud', regional='EMEA', team=f'T{i//2+1}',
             rep=f'r{i+1}', kw=[100, 200, 300, 400][i],
             rep_name=f'Rep {i+1}') for i in range(4)])
    targets = pd.DataFrame([dict(product='cloud', fiscal_quarter=fq,
                                 tgt=1_000_000.0) for fq in (1, 2)])
    return hdf, targets


def _quotas(**kw):
    hdf, targets = _frames()
    q, _ = cascade_many(hdf, targets, group_keys=['product'],
                        target_col='tgt',
                        taxonomy=['regional', 'team', 'rep'],
                        metrics=KW, **kw)
    return q


# ----------------------------------------------------------------------
# 1. cascade_many stamps the cascade identity
# ----------------------------------------------------------------------
def test_attrs_stamped():
    print(SEPARATOR)
    print("TEST 1: cascade_many output carries "
          ".attrs['cascade_row_keys']")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['rep_name'])
    print(f"  cascade_row_keys = {q.attrs['cascade_row_keys']}")
    assert q.attrs['cascade_row_keys'] == ['product', 'fiscal_quarter']


# ----------------------------------------------------------------------
# 2. The #40 scenario now Just Works: no row_keys, metadata present
# ----------------------------------------------------------------------
def test_manager_pin_without_row_keys():
    print(f"\n\n{SEPARATOR}")
    print("TEST 2: manager pin, metadata_cols present, NO row_keys -> "
          "subtree pin via attrs")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['rep_name'])
    edited, rep = apply_pins(q, [Pin('T1', 900_000.0)])
    assert rep.iloc[0]['pin_type'] == 'subtree'        # was 'leaf' pre-fix
    e = edited[edited.fiscal_quarter == 1].set_index('node_id')
    print(f"  pin_type={rep.iloc[0]['pin_type']}  "
          f"T1={e.loc['T1', 'base_quota']:,.2f}  "
          f"r1+r2={e.loc[['r1', 'r2'], 'base_quota'].sum():,.2f}")
    # descendants follow the pin; parent conserved; absorber shed
    assert abs(e.loc[['r1', 'r2'], 'base_quota'].sum()
               - e.loc['T1', 'base_quota']) < 0.05
    assert abs(e.loc['T1', 'base_quota'] - 450_000.0) < 0.05
    assert abs(e.loc[['T1', 'T2'], 'base_quota'].sum()
               - e.loc['EMEA', 'base_quota']) < 0.05
    assert bool(rep.iloc[0]['feasible'])


# ----------------------------------------------------------------------
# 3. Orphan guard: poisoned inference is a hard error, never silent
# ----------------------------------------------------------------------
def test_orphan_guard_names_poison_column():
    print(f"\n\n{SEPARATOR}")
    print("TEST 3: attrs stripped -> poisoned inference raises, naming "
          "'rep_name'")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['rep_name'])
    q.attrs = {}                       # simulate pre-v0.21.0 frame
    try:
        apply_pins(q, [Pin('T1', 900_000.0)])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        print(f"  message names poison column: {'rep_name' in msg}")
        assert 'rep_name' in msg
        assert 'row_keys' in msg
        # suggested keys exclude the poison column
        assert "'product'" in msg and "'fiscal_quarter'" in msg


# ----------------------------------------------------------------------
# 4. Wrong EXPLICIT row_keys hit the same guard
# ----------------------------------------------------------------------
def test_wrong_explicit_keys_raise():
    print(f"\n\n{SEPARATOR}")
    print("TEST 4: explicit row_keys including a per-node column raise")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['rep_name'])
    try:
        apply_pins(q, [Pin('T1', 900_000.0)],
                   row_keys=['product', 'fiscal_quarter', 'rep_name'])
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'rep_name' in str(e)
        print("  raised as expected")


# ----------------------------------------------------------------------
# 5. The filer's workaround (correct explicit keys) still works
# ----------------------------------------------------------------------
def test_correct_explicit_keys():
    print(f"\n\n{SEPARATOR}")
    print("TEST 5: explicit correct row_keys unchanged")
    print(SEPARATOR)
    q = _quotas(metadata_cols=['rep_name'])
    edited, rep = apply_pins(q, [Pin('T1', 900_000.0)],
                             row_keys=['product', 'fiscal_quarter'])
    assert rep.iloc[0]['pin_type'] == 'subtree'
    e = edited[edited.fiscal_quarter == 1].set_index('node_id')
    assert abs(e.loc[['r1', 'r2'], 'base_quota'].sum()
               - e.loc['T1', 'base_quota']) < 0.05
    print("  subtree pin conserved")


# ----------------------------------------------------------------------
# 6. Leaf-only frames (no manager rows present) are still allowed
# ----------------------------------------------------------------------
def test_leaf_only_frame_allowed():
    print(f"\n\n{SEPARATOR}")
    print("TEST 6: leaf-only frame (parents absent) -> no orphan error, "
          "leaf pin + sibling absorption work")
    print(SEPARATOR)
    q = _quotas()
    leaves = q[q.is_leaf].copy()
    edited, rep = apply_pins(leaves, [Pin('r1', 200_000.0)],
                             row_keys=['product', 'fiscal_quarter'])
    e = edited[edited.fiscal_quarter == 1].set_index('node_id')
    print(f"  r1={e.loc['r1', 'base_quota']:,.2f}  "
          f"r2={e.loc['r2', 'base_quota']:,.2f}")
    assert rep.iloc[0]['pin_type'] == 'leaf'
    assert abs(e.loc['r1', 'base_quota'] - 100_000.0) < 0.05
    assert bool(rep.iloc[0]['feasible'])


if __name__ == '__main__':
    test_attrs_stamped()
    test_manager_pin_without_row_keys()
    test_orphan_guard_names_poison_column()
    test_wrong_explicit_keys_raise()
    test_correct_explicit_keys()
    test_leaf_only_frame_allowed()

    print(f"\n\n{SEPARATOR}")
    print("ALL PIN ROW-KEY TESTS PASSED")
    print(SEPARATOR)
