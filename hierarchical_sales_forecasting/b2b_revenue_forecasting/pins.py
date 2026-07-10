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
        Node ids protected from this pin (#24, #39): siblings listed
        here never absorb the pin's delta, and descendants listed here
        keep their current values when the pinned subtree rescales
        (free siblings inside the subtree stretch to fill instead).
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
         subtree PROTECTION-AWARE (#39): descendants that are pinned by
         another pin, in freeze_nodes, or in this pin's exclude list
         keep their current values; free descendants scale to fill the
         remainder proportional to their base (equal split if the free
         baseline is 0). With nothing protected this is exactly the
         proportional rescale. If protected values alone exceed the new
         subtree total, free rows floor at $0 and the shortfall lands
         in the feasibility report ('subtree_shortfall') with a
         warning — never hidden,
      4. parents are conserved exactly wherever absorption succeeds;
         where it CANNOT (siblings' FREE capacity floors at $0 on a
         shed, or no eligible absorber exists on a gain), the
         unabsorbed remainder is reported in the feasibility frame —
         never hidden, never a negative quota. Absorption is
         distributed proportional to each sibling's free capacity
         (base minus protected mass inside it),
      5. all math runs on the BASE layer; each modified row's
         cascaded_quota is re-derived from its own original hedge ratio
         (never re-hedged) per the issue #21 contract. Pin.basis
         controls which layer Pin.total refers to. In particular,
         pinning a manager on basis='cascaded' under HedgeByDepth makes
         its descendants roll up to pinned x the cross-level hedge —
         each depth keeps its hedge ratio relative to the pinned value.

    Pins are applied in order for absorption, but every pin's node is
    protected from every OTHER pin's rescales (#39) — so pin order no
    longer changes where pinned nodes land. A node pinned by ANY pin
    never absorbs for another.

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

    def _protected_sum(k, node_id, protected) -> float:
        """Immovable base mass inside `node_id`'s subtree: the maximal
        protected descendants (their own subtrees are never entered)."""
        total = 0.0
        for cidx in child_ix.get((k, node_id), []):
            cid = df.at[cidx, "node_id"]
            if cid in protected:
                total += float(df.at[cidx, "base_quota"])
            else:
                total += _protected_sum(k, cid, protected)
        return total

    def _has_gain_room(k, idx, protected) -> bool:
        """True if the subtree at row `idx` contains at least one FREE
        leaf that could take a gain (any free leaf, even at $0)."""
        node = df.at[idx, "node_id"]
        kids = child_ix.get((k, node), [])
        if not kids:
            return True
        return any(df.at[c, "node_id"] not in protected
                   and _has_gain_room(k, c, protected) for c in kids)

    def _rescale_subtree(k, node_id, new_base, protected) -> float:
        """
        Protection-aware subtree rescale (issue #39). Top-down: at each
        level, PROTECTED children (pinned by another pin, frozen, or in
        Pin.exclude) keep their current values and their subtrees are
        untouched; FREE children scale to fill the remainder,
        proportional to their current base (equal split when the free
        baseline is 0 — no mix to preserve). With nothing protected
        this reproduces the uniform factor scale exactly. Returns the
        SHORTFALL (> 0 when protected values alone exceed `new_base`,
        so free children floor at $0 and the subtree cannot sum to the
        parent — reported, never hidden).
        """
        kids = child_ix.get((k, node_id), [])
        if not kids:
            return 0.0
        free = [c for c in kids if df.at[c, "node_id"] not in protected]
        prot_sum = sum(float(df.at[c, "base_quota"]) for c in kids
                       if df.at[c, "node_id"] in protected)
        remainder = new_base - prot_sum
        shortfall = max(-remainder, 0.0)
        remainder = max(remainder, 0.0)
        if not free:
            return shortfall + (remainder if remainder > 0.005 else 0.0)
        free_sum = sum(float(df.at[c, "base_quota"]) for c in free)
        for c in free:
            c_base = float(df.at[c, "base_quota"])
            share = (remainder * c_base / free_sum if free_sum > 0
                     else remainder / len(free))
            _set_base(c, share)
            shortfall += _rescale_subtree(k, df.at[c, "node_id"], share,
                                          protected)
        return shortfall

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

        # Protection set (issue #39): nodes pinned by ANY pin, frozen
        # nodes, and this pin's exclude list keep their current values
        # everywhere — including INSIDE a pinned manager's subtree and
        # inside absorbing siblings' subtrees. This makes pin order
        # irrelevant: a later manager pin rescales AROUND an earlier
        # descendant pin instead of trampling it.
        protected = (all_pinned | frozen | set(pin.exclude)) - {pin.node}

        absorbed_sum, unabsorbed_sum, shortfall_sum = 0.0, 0.0, 0.0
        for idx in node_idx:
            k = key_of.at[idx]
            old_base = float(df.at[idx, "base_quota"])
            # Convert the allocated (basis-layer) value to a BASE value
            new_base = (alloc[idx] if pin.basis == "base"
                        else alloc[idx] / ratio[idx])
            delta = new_base - old_base       # >0: siblings must shed

            # Set the pinned node (+ subtree) — pins are sacrosanct.
            # Free descendants scale to fill; protected ones hold.
            if old_base <= 0 and is_subtree and new_base != 0:
                warnings.warn(
                    f"Pin '{pin.node}': subtree baseline is 0 in cascade "
                    f"{dict(zip(keys, k)) if keys else '<single>'} — no "
                    f"mix to preserve, splitting equally among free "
                    f"children at each level.",
                    UserWarning, stacklevel=2,
                )
            shortfall_sum += _rescale_subtree(k, pin.node, new_base,
                                              protected)
            _set_base(idx, new_base)

            # Eligible absorbers: same cascade, same parent, not the pin,
            # not frozen/excluded/pinned (#24). Their capacity is their
            # FREE base (total minus protected mass inside — #39): a
            # subtree can only shed what its unprotected rows carry.
            sibs = [s for s in child_ix.get((k, df.at[idx, "parent"]), [])
                    if df.at[s, "node_id"] != pin.node
                    and df.at[s, "node_id"] not in protected]
            free_cap = {
                s: max(float(df.at[s, "base_quota"])
                       - _protected_sum(k, df.at[s, "node_id"], protected),
                       0.0)
                for s in sibs
            }
            pool = sum(free_cap.values())

            if abs(delta) < 0.005:
                continue
            if delta > 0:                     # siblings shed, floor $0
                absorb = min(delta, pool)
                if pool > 0:
                    for s in sibs:
                        if free_cap[s] <= 0:
                            continue
                        s_base = float(df.at[s, "base_quota"])
                        s_new = s_base - absorb * (free_cap[s] / pool)
                        _set_base(s, s_new)
                        shortfall_sum += _rescale_subtree(
                            k, df.at[s, "node_id"], s_new, protected)
                absorbed_sum += absorb
                unabsorbed_sum += delta - absorb
            else:                             # siblings gain
                gain = -delta
                if pool > 0:
                    for s in sibs:
                        if free_cap[s] <= 0:
                            continue
                        s_base = float(df.at[s, "base_quota"])
                        s_new = s_base + gain * (free_cap[s] / pool)
                        _set_base(s, s_new)
                        shortfall_sum += _rescale_subtree(
                            k, df.at[s, "node_id"], s_new, protected)
                    absorbed_sum += gain
                else:
                    roomy = [s for s in sibs
                             if _has_gain_room(k, s, protected)]
                    if roomy:
                        for s in roomy:
                            s_new = (float(df.at[s, "base_quota"])
                                     + gain / len(roomy))
                            _set_base(s, s_new)
                            shortfall_sum += _rescale_subtree(
                                k, df.at[s, "node_id"], s_new, protected)
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
            "subtree_shortfall": round(shortfall_sum, 2),
            "feasible": (abs(unabsorbed_sum) <= 0.01
                         and abs(shortfall_sum) <= 0.01),
        })
        if unabsorbed_sum > 0.01:
            warnings.warn(
                f"Pin '{pin.node}': {unabsorbed_sum:,.2f} could not be "
                f"absorbed by eligible siblings (floors at $0 / no "
                f"absorbers). Parents will not fully conserve — see the "
                f"feasibility report.",
                UserWarning, stacklevel=2,
            )
        if shortfall_sum > 0.01:
            warnings.warn(
                f"Pin '{pin.node}': protected descendants (other pins / "
                f"freeze_nodes / exclude) hold {shortfall_sum:,.2f} more "
                f"than the pinned subtree can contain — free descendants "
                f"floored at $0 and the subtree will not sum to the pin. "
                f"See feasibility report column 'subtree_shortfall'.",
                UserWarning, stacklevel=2,
            )

    return df, pd.DataFrame(report_rows)
