"""
Aggregate pinning on cascade outputs (issues #22 / #31 / #24).

`new_ic_overrides` (v0.13.0) pins a node WITHIN one cascade. Real
planning also needs the aggregate version: "this territory carries
exactly $2.6M in TOTAL across all (product x sales_type x quarter)
cascades" — with the mix preserved, siblings absorbing the delta inside
each cascade, parents conserved, and certain territories frozen.

`apply_pins` is that operation, performed post-cascade on the tidy long
frame (all math on the BASE layer, hedged values derived from each
row's own ratio — the issue #21 contract).
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Columns that are structure/values, not cascade-identifying keys.
_STRUCTURAL_COLS = {
    "node_id", "parent", "depth", "level", "is_leaf",
    "base_quota", "cascaded_quota", "unhedged_quota", "hedge_buffer",
    "overassignment_pct", "is_gated", "gate_relaxed", "is_unallocated",
    "original_id", "routed", "reason", "is_pinned", "pin_type",
}

_PIN_BASES = ("base", "cascaded")


class Pin:
    """
    One aggregate pin (issues #22/#31): hold `node` (leaf OR manager) at
    an exact `total` across every cascade row it appears in.

    Parameters
    ----------
    node : str
        Node id to pin. A manager pin rescales its whole subtree
        proportionally within each cascade (pin_type='subtree').
    total : float
        The exact combined amount across all matched rows (>= 0).
    basis : str
        "base" (default) — total refers to the un-hedged plan layer;
        hedged values derive from each row's own ratio. "cascaded" —
        total refers to the hedged layer; base derives by dividing the
        ratio back out.
    scope : Optional[Dict[str, Any]]
        Column=value filters restricting WHICH of the node's rows count
        toward the total (e.g. {'fiscal_quarter': 1} pins Q1 only;
        unmatched rows are untouched).
    exclude : Optional[List[str]]
        Sibling node ids that must not absorb this pin's delta (#24).
    """

    def __init__(self, node: str, total: float, basis: str = "base",
                 scope: Optional[Dict[str, Any]] = None,
                 exclude: Optional[List[str]] = None):
        if not node or not isinstance(node, str):
            raise ValueError(f"Pin.node must be a node id string, "
                             f"got {node!r}.")
        if not isinstance(total, (int, float)) or total < 0:
            raise ValueError(f"Pin.total must be a non-negative number, "
                             f"got {total!r}.")
        if basis not in _PIN_BASES:
            raise ValueError(f"Pin.basis must be one of {_PIN_BASES}, "
                             f"got '{basis}'.")
        self.node = node
        self.total = float(total)
        self.basis = basis
        self.scope = dict(scope or {})
        self.exclude = list(exclude or [])

    def __repr__(self):  # pragma: no cover — debugging nicety
        return (f"Pin(node={self.node!r}, total={self.total:,.2f}, "
                f"basis={self.basis!r})")


def apply_pins(
    quotas_long: pd.DataFrame,
    pins: List[Pin],
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply aggregate pins to a cascade_many output (issues #22/#31/#24).

    For each Pin, across every cascade row of the pinned node (optionally
    narrowed by Pin.scope):
      1. the node's per-row values are scaled so they SUM to Pin.total,
         proportional to the node's baseline mix (a $2.6M total lands on
         products/quarters in the same shape the cascade produced),
      2. within EACH cascade, the row's delta is absorbed by eligible
         siblings proportional to their base (they shed when the pin
         grows, gain when it shrinks) — frozen/excluded/pinned nodes
         never absorb and are never modified,
      3. manager pins (and manager absorbers) rescale their whole
         subtree proportionally, so every depth stays consistent,
      4. parents are conserved exactly wherever absorption succeeds;
         where it CANNOT (siblings floor at $0 on a shed, or no eligible
         absorber exists on a gain), the unabsorbed remainder is
         reported in the feasibility frame — never hidden, never a
         negative quota,
      5. all math runs on the BASE layer; each modified row's
         cascaded_quota is re-derived from its own original hedge ratio
         (never re-hedged) per the issue #21 contract. Pin.basis
         controls which layer Pin.total refers to.

    Pins are applied in order; later pins see earlier edits. A node
    pinned by ANY pin never absorbs for another.

    Parameters
    ----------
    quotas_long : pd.DataFrame
        cascade_many output (or equivalent long frame with node_id,
        parent, depth, base_quota, cascaded_quota).
    pins : list[Pin]
    freeze_nodes : Optional[List[str]]
        Global freeze list (#24): these nodes are never absorbers and
        never modified, for every pin.
    row_keys : Optional[List[str]]
        Columns identifying ONE cascade (group keys + sub-target columns
        like fiscal_quarter). Inferred as "every non-structural column"
        when omitted — pass explicitly if your frame carries extra
        per-node columns (e.g. metadata_cols), which would confuse the
        inference.

    Returns
    -------
    (edited_df, feasibility_report)
        edited_df — a copy with updated base_quota / cascaded_quota,
            plus `is_pinned` and `pin_type` ('leaf'/'subtree')
            provenance on the pinned nodes' rows.
        feasibility_report — one row per pin: pin_node, pin_type,
            basis, requested_total, baseline_total, achieved_total,
            rows_affected, absorbed, unabsorbed, feasible.
    """
    required = {"node_id", "parent", "depth", "base_quota",
                "cascaded_quota"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)}.")
    df = quotas_long.copy()
    if "is_pinned" not in df.columns:
        df["is_pinned"] = False
    if "pin_type" not in df.columns:
        df["pin_type"] = None

    keys = (list(row_keys) if row_keys is not None
            else [c for c in df.columns if c not in _STRUCTURAL_COLS])
    key_of = (df[keys].apply(tuple, axis=1) if keys
              else pd.Series([()] * len(df), index=df.index))

    # Uniqueness: one row per (cascade, node)
    if df.groupby([key_of, df["node_id"]]).size().max() > 1:
        raise ValueError(
            "quotas_long has multiple rows per (cascade, node) under the "
            f"inferred keys {keys} — pass row_keys= listing the columns "
            "that identify one cascade (group keys + e.g. fiscal_quarter)."
        )

    # Original hedge ratio per row index (edits preserve it)
    ratio = {}
    for idx in df.index:
        b = df.at[idx, "base_quota"]
        ratio[idx] = (float(df.at[idx, "cascaded_quota"]) / float(b)
                      if pd.notna(b) and float(b) != 0.0 else 1.0)

    # Row lookup: (key, node_id) -> index ; children: (key, parent) -> [idx]
    row_ix: Dict[Tuple[Any, str], Any] = {}
    child_ix: Dict[Tuple[Any, Any], List[Any]] = {}
    for idx in df.index:
        k = key_of.at[idx]
        row_ix[(k, df.at[idx, "node_id"])] = idx
        child_ix.setdefault((k, df.at[idx, "parent"]), []).append(idx)

    def _descendants(k, node_id) -> List[Any]:
        out, stack = [], [node_id]
        while stack:
            for cidx in child_ix.get((k, stack.pop()), []):
                out.append(cidx)
                stack.append(df.at[cidx, "node_id"])
        return out

    def _set_base(idx, new_base: float) -> None:
        df.at[idx, "base_quota"] = round(new_base, 2)
        df.at[idx, "cascaded_quota"] = round(new_base * ratio[idx], 2)

    def _scale_subtree(k, node_id, factor: float) -> None:
        for didx in _descendants(k, node_id):
            _set_base(didx, float(df.at[didx, "base_quota"]) * factor)

    frozen = set(freeze_nodes or [])
    all_pinned = {p.node for p in pins}
    report_rows = []

    for pin in pins:
        basis_col = "base_quota" if pin.basis == "base" else "cascaded_quota"
        mask = df["node_id"] == pin.node
        for col, val in pin.scope.items():
            if col not in df.columns:
                raise ValueError(f"Pin scope column '{col}' not in "
                                 f"quotas_long.")
            mask &= df[col] == val
        node_idx = list(df.index[mask])
        if not node_idx:
            raise ValueError(f"Pin node '{pin.node}' matches no rows "
                             f"(after scope {pin.scope}).")
        is_subtree = any(_descendants(key_of.at[i], pin.node)
                         for i in node_idx)
        pin_type = "subtree" if is_subtree else "leaf"

        baseline_total = float(df.loc[node_idx, basis_col].sum())
        # Per-row allocation: proportional to baseline mix, equal if flat 0
        if baseline_total > 0:
            alloc = {i: pin.total * float(df.at[i, basis_col]) / baseline_total
                     for i in node_idx}
        else:
            warnings.warn(
                f"Pin '{pin.node}': baseline total is 0 across matched rows "
                f"— splitting {pin.total:,.2f} equally across "
                f"{len(node_idx)} row(s).",
                UserWarning, stacklevel=2,
            )
            alloc = {i: pin.total / len(node_idx) for i in node_idx}

        absorbed_sum, unabsorbed_sum = 0.0, 0.0
        for idx in node_idx:
            k = key_of.at[idx]
            old_base = float(df.at[idx, "base_quota"])
            # Convert the allocated (basis-layer) value to a BASE value
            new_base = (alloc[idx] if pin.basis == "base"
                        else alloc[idx] / ratio[idx])
            delta = new_base - old_base       # >0: siblings must shed

            # Set the pinned node (+ subtree) — pins are sacrosanct
            if old_base > 0:
                _scale_subtree(k, pin.node, new_base / old_base)
            elif is_subtree and new_base != 0:
                warnings.warn(
                    f"Pin '{pin.node}': subtree baseline is 0 in cascade "
                    f"{dict(zip(keys, k)) if keys else '<single>'} — the "
                    f"pinned amount sits on the manager row only (no "
                    f"leaf mix to scale).",
                    UserWarning, stacklevel=2,
                )
            _set_base(idx, new_base)

            # Eligible absorbers: same cascade, same parent, not the pin,
            # not frozen/excluded/pinned (#24)
            blocked = frozen | set(pin.exclude) | all_pinned
            sibs = [s for s in child_ix.get((k, df.at[idx, "parent"]), [])
                    if df.at[s, "node_id"] != pin.node
                    and df.at[s, "node_id"] not in blocked]
            pool = sum(float(df.at[s, "base_quota"]) for s in sibs)

            if abs(delta) < 0.005:
                continue
            if delta > 0:                     # siblings shed, floor $0
                absorb = min(delta, pool)
                if pool > 0:
                    for s in sibs:
                        s_base = float(df.at[s, "base_quota"])
                        if s_base <= 0:
                            continue
                        shed = absorb * (s_base / pool)
                        _scale_subtree(k, df.at[s, "node_id"],
                                       (s_base - shed) / s_base)
                        _set_base(s, s_base - shed)
                absorbed_sum += absorb
                unabsorbed_sum += delta - absorb
            else:                             # siblings gain
                gain = -delta
                if pool > 0:
                    for s in sibs:
                        s_base = float(df.at[s, "base_quota"])
                        if s_base <= 0:
                            continue
                        add = gain * (s_base / pool)
                        _scale_subtree(k, df.at[s, "node_id"],
                                       (s_base + add) / s_base)
                        _set_base(s, s_base + add)
                    absorbed_sum += gain
                else:
                    leaf_sibs = [s for s in sibs if df.at[s, "is_leaf"]] \
                        if "is_leaf" in df.columns else sibs
                    if leaf_sibs:
                        for s in leaf_sibs:
                            _set_base(s, float(df.at[s, "base_quota"])
                                      + gain / len(leaf_sibs))
                        absorbed_sum += gain
                    else:
                        unabsorbed_sum += gain

        df.loc[node_idx, "is_pinned"] = True
        df.loc[node_idx, "pin_type"] = pin_type

        achieved = float(df.loc[node_idx, basis_col].sum())
        report_rows.append({
            "pin_node": pin.node,
            "pin_type": pin_type,
            "basis": pin.basis,
            "requested_total": round(pin.total, 2),
            "baseline_total": round(baseline_total, 2),
            "achieved_total": round(achieved, 2),
            "rows_affected": len(node_idx),
            "absorbed": round(absorbed_sum, 2),
            "unabsorbed": round(unabsorbed_sum, 2),
            "feasible": abs(unabsorbed_sum) <= 0.01,
        })
        if unabsorbed_sum > 0.01:
            warnings.warn(
                f"Pin '{pin.node}': {unabsorbed_sum:,.2f} could not be "
                f"absorbed by eligible siblings (floors at $0 / no "
                f"absorbers). Parents will not fully conserve — see the "
                f"feasibility report.",
                UserWarning, stacklevel=2,
            )

    return df, pd.DataFrame(report_rows)
