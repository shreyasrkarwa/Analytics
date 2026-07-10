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
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Columns that are structure/values, not cascade-identifying keys.
_STRUCTURAL_COLS = {
    "node_id", "parent", "depth", "level", "is_leaf",
    "base_quota", "cascaded_quota", "unhedged_quota", "hedge_buffer",
    "overassignment_pct", "is_gated", "gate_relaxed", "is_unallocated",
    "original_id", "routed", "reason", "is_pinned", "pin_type",
}

_PIN_BASES = ("base", "cascaded")
_ON_MISSING = ("error", "skip", "warn")


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
    on_missing: str = "error",
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

    REMAINDER PINS (issue #42) — "these children are fixed, the
    unpinned one(s) take the rest" needs NO special mode; it is plain
    pin composition::

        # remainder to unpinned siblings, parent conserved as-is:
        apply_pins(quotas, [Pin('T1', 500_000), Pin('T2', 1_500_000)])

        # the parent's total is pinned too (the requested
        # Pin(parent, total, children={...}, remainder='auto')):
        apply_pins(quotas, [Pin('EMEA', 5_000_000),
                            Pin('T1', 500_000), Pin('T2', 1_500_000)])

    Why this is exactly the requested semantics: pinned nodes never
    absorb for other pins (#24), so every pin's delta lands entirely on
    the UNPINNED siblings; proportional absorption preserves their
    ratios, so the remainder splits at baseline proportions (the same
    identity as issue #37); and depth-ordered application (#41) makes
    the parent pin land before its children regardless of list order.
    If the pinned children exceed the parent's pin, free siblings floor
    at $0 and every unplaced dollar appears in the feasibility report
    (unabsorbed / subtree_shortfall) — never hidden.

    LIST ORDER NEVER MATTERS (issue #41). Pinned values have been
    order-independent since v0.20.0 (every pin's node is protected from
    every other pin's rescales, #39); since v0.22.0 application is also
    canonical — pins are applied by DEPTH of the pinned node,
    shallowest first (managers before leaves), stable within a depth —
    so the whole output frame, absorber rows included, is identical for
    any ordering of `pins`. Leaf-pin allocations are computed against
    post-manager-rescale baselines. The feasibility report is returned
    in the INPUT pin order. A node pinned by ANY pin never absorbs for
    another.

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
        like fiscal_quarter). When omitted (issue #40): cascade_many
        outputs (v0.21.0+) carry the exact keys in
        .attrs['cascade_row_keys'] and those are used automatically;
        otherwise every non-structural column is inferred. Either way a
        wrong key set can no longer corrupt silently — if any row's
        parent exists in the frame but not under the same cascade key,
        apply_pins raises, naming the per-node columns poisoning the
        identity and suggesting the corrected row_keys.
    on_missing : str
        What to do with a pin that matches ZERO rows (issue #48) — a
        product a region doesn't sell, a team with no reps in a
        quarter. 'error' (default): raise, preserving historic
        behavior. 'skip': drop the pin and record it in the
        feasibility report (skipped=True, reason='node_absent' when
        the node id is nowhere in the frame, 'empty_scope' when the
        node exists but Pin.scope matched nothing). 'warn': like
        'skip', plus one summary warning naming the skipped pins.
        A skipped pin is dropped ENTIRELY — it does not enter the
        protection set, so its node can still absorb for other pins
        and still rescales inside pinned subtrees (no ghost
        side-effects). A missing Pin.scope COLUMN is a programming
        error and always raises, regardless of on_missing.

    Returns
    -------
    (edited_df, feasibility_report)
        edited_df — a copy with updated base_quota / cascaded_quota,
            plus `is_pinned` and `pin_type` ('leaf'/'subtree')
            provenance on the pinned nodes' rows.
        feasibility_report — one row per pin, in INPUT order:
            pin_node, pin_type, basis, requested_total,
            baseline_total, achieved_total, rows_affected, absorbed,
            unabsorbed, subtree_shortfall, feasible, plus (v0.25.0)
            skipped and reason for on_missing bookkeeping.
    """
    if on_missing not in _ON_MISSING:
        raise ValueError(f"on_missing must be one of {_ON_MISSING}, "
                         f"got '{on_missing}'.")
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

    if row_keys is not None:
        keys = list(row_keys)
    else:
        # Prefer the cascade-identity stamp cascade_many leaves on its
        # output (issue #40) — exact group_keys + sub-target columns.
        stamped = quotas_long.attrs.get("cascade_row_keys")
        if (isinstance(stamped, (list, tuple))
                and all(isinstance(c, str) and c in df.columns
                        for c in stamped)):
            keys = list(stamped)
        else:
            keys = [c for c in df.columns if c not in _STRUCTURAL_COLS]
    # NaN normalized to None so tuples compare by value (NaN != NaN
    # would silently split key groups — issue #40 hygiene).
    key_of = (df[keys].apply(
                  lambda r: tuple(None if pd.isna(v) else v for v in r),
                  axis=1) if keys
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

    # Orphan guard (issue #40): under CORRECT cascade keys, every row's
    # parent (when the parent node exists in the frame at all) has a row
    # in the SAME key tuple — a cascade always contains the parent. An
    # "orphan" (parent present elsewhere in the frame but not in this
    # row's key group) proves the keys are wrong: per-node columns (e.g.
    # metadata_cols) are poisoning the cascade identity. Before v0.21.0
    # this silently downgraded manager pins to leaf pins and mis-grouped
    # absorbers; now it's a hard error naming the poison columns.
    all_node_ids = set(df["node_id"])
    orphans = [idx for idx in df.index
               if pd.notna(df.at[idx, "parent"])
               and df.at[idx, "parent"] in all_node_ids
               and (key_of.at[idx], df.at[idx, "parent"]) not in row_ix]
    if orphans:
        poison = set()
        for idx in orphans[:20]:
            parent_rows = df.index[df["node_id"] == df.at[idx, "parent"]]
            for col in keys:
                v = df.at[idx, col]
                if all(not (df.at[p, col] == v
                            or (pd.isna(df.at[p, col]) and pd.isna(v)))
                       for p in parent_rows):
                    poison.add(col)
        suggested = [c for c in keys if c not in poison]
        raise ValueError(
            f"apply_pins: {len(orphans)} row(s) have a parent that exists "
            f"in the frame but NOT under the same cascade key — the "
            f"row_keys are wrong, so manager/subtree pins and sibling "
            f"absorption would silently misbehave. Keys in use: {keys}. "
            f"Columns that vary per node and are poisoning the cascade "
            f"identity: {sorted(poison)}. Pass "
            f"row_keys={suggested or '[<group keys + sub-target columns>]'} "
            f"(group keys + sub-target columns like fiscal_quarter only). "
            f"cascade_many outputs since v0.21.0 carry the correct keys "
            f"in .attrs['cascade_row_keys'] and need no row_keys at all."
        )

    # Missing-pin pre-pass (issue #48): classify BEFORE the protection
    # set and application order are built, so a skipped pin has no
    # ghost side-effects (its node may still absorb / rescale freely).
    skipped_reason: Dict[int, str] = {}
    for i, pin in enumerate(pins):
        pmask = df["node_id"] == pin.node
        for col, val in pin.scope.items():
            if col not in df.columns:
                raise ValueError(f"Pin scope column '{col}' not in "
                                 f"quotas_long.")
            pmask &= df[col] == val
        if not pmask.any():
            if on_missing == "error":
                raise ValueError(
                    f"Pin node '{pin.node}' matches no rows (after "
                    f"scope {pin.scope}). Pass on_missing='skip' (or "
                    f"'warn') to drop such pins into the feasibility "
                    f"report instead of aborting the batch.")
            skipped_reason[i] = ("node_absent"
                                 if pin.node not in all_node_ids
                                 else "empty_scope")
    if skipped_reason and on_missing == "warn":
        named = [f"{pins[i].node} ({r})"
                 for i, r in sorted(skipped_reason.items())]
        warnings.warn(
            f"apply_pins: skipped {len(skipped_reason)} pin(s) with no "
            f"matching rows: {', '.join(named)}. See the feasibility "
            f"report (skipped / reason columns).",
            UserWarning, stacklevel=2)

    frozen = set(freeze_nodes or [])
    all_pinned = {p.node for i, p in enumerate(pins)
                  if i not in skipped_reason}

    # Canonical application order (issue #41): shallowest pinned node
    # first — managers before leaves — stable within a depth (same-depth
    # pins keep their list order). Pinned VALUES are order-independent
    # since v0.20.0 (protection); this makes the ABSORBER rows
    # deterministic too, so the output frame is identical for any pin
    # list order. Leaf-pin allocations are computed against
    # post-manager-rescale baselines, the natural reading. The
    # feasibility report is returned in the INPUT pin order.
    def _pin_depth(pin: Pin) -> float:
        mask = df["node_id"] == pin.node
        for col, val in pin.scope.items():
            if col in df.columns:      # missing columns raise in the loop
                mask &= df[col] == val
        depths = df.loc[mask, "depth"]
        return float(depths.min()) if len(depths) else float("inf")

    application_order = sorted(
        (i for i in range(len(pins)) if i not in skipped_reason),
        key=lambda i: (_pin_depth(pins[i]), i))
    report_rows: List[Optional[dict]] = [None] * len(pins)
    for i, reason in skipped_reason.items():
        report_rows[i] = {
            "pin_node": pins[i].node, "pin_type": None,
            "basis": pins[i].basis,
            "requested_total": round(pins[i].total, 2),
            "baseline_total": 0.0, "achieved_total": 0.0,
            "rows_affected": 0, "absorbed": 0.0, "unabsorbed": 0.0,
            "subtree_shortfall": 0.0, "feasible": False,
            "skipped": True, "reason": reason,
        }

    for pin_i in application_order:
        pin = pins[pin_i]
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
        report_rows[pin_i] = ({
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
            "skipped": False, "reason": None,
        })  # slot pin_i: report emitted in INPUT pin order (issue #41)
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


def _scope_mask(df: pd.DataFrame,
                scope: Optional[Dict[str, Any]]) -> pd.Series:
    """Row mask for Pin-style scope filters (shared by redistribute /
    concentrate)."""
    mask = pd.Series(True, index=df.index)
    for col, val in (scope or {}).items():
        if col not in df.columns:
            raise ValueError(f"scope column '{col}' not in quotas_long.")
        mask &= df[col] == val
    return mask


def _run_pins_quietly(df: pd.DataFrame, pins: List[Pin],
                      freeze_nodes: Optional[List[str]],
                      row_keys: Optional[List[str]]) -> pd.DataFrame:
    """apply_pins, with internal absorption noise suppressed.

    redistribute/concentrate emit pin packages that are conservation-
    neutral BY CONSTRUCTION; when every sibling is pinned there is no
    buffer to 'absorb', so apply_pins' unabsorbed warnings would be
    false alarms. The callers verify conservation explicitly instead
    (the report's `exact` column). All other warnings re-emit.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edited, _ = apply_pins(df, pins, freeze_nodes=freeze_nodes,
                               row_keys=row_keys)
    for w in caught:
        if "could not be absorbed" not in str(w.message):
            warnings.warn(w.message, w.category, stacklevel=3)
    return edited


def redistribute(
    quotas_long: pd.DataFrame,
    from_node: str,
    to_nodes: Optional[List[str]] = None,
    weights: Union[str, Dict[str, float]] = "proportional",
    scope: Optional[Dict[str, Any]] = None,
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Move a node's ENTIRE (optionally scoped) quota to its siblings,
    reshaping source and destination subtrees at every depth (#43).

    "MM_AMER_EAST gets zero Migration; move it to CENTRAL/WEST
    proportionally" is::

        edited, report = redistribute(quotas_long, 'EAST',
                                      scope={'st1_sales_type': 'Migration'})

    Custom split (the issue's `weights=`)::

        edited, report = redistribute(quotas_long, 'EAST',
                                      weights={'CENTRAL': .7, 'WEST': .3},
                                      scope={'st1_sales_type': 'Migration'})

    This is a thin convenience over `apply_pins` — it writes the pins
    for you (source pinned to $0, each destination pinned to
    baseline + share x source), so it inherits the whole pin contract:
    subtrees rescale at every depth, parents conserve, each row's
    hedged value re-derives from its own hedge ratio (#21), other
    scopes untouched, frozen nodes never move, floors at $0. Siblings
    that are neither source nor destination end EXACTLY at baseline
    (sequential proportional absorption cancels — verified, not
    assumed: the returned report checks it).

    Parameters
    ----------
    quotas_long : pd.DataFrame
        cascade_many output (or equivalent long frame).
    from_node : str
        Source node (leaf or manager; not a root). Its scoped subtree
        goes to $0.
    to_nodes : list[str], optional
        Restrict recipients. Default: every unfrozen sibling. Ignored
        for dict `weights` (the keys are the recipients).
    weights : 'proportional' | 'equal' | dict[node, weight]
        How the source total splits across recipients.
        'proportional' (default) — by the recipients' baseline mix;
        'equal' — evenly; dict — explicit shares (normalized). All-zero
        recipient baselines fall back to equal, with a warning.
    scope : dict, optional
        Column=value filters (Pin.scope): only matching cascades move.
    freeze_nodes : list[str], optional
        Passed to apply_pins; frozen nodes are never recipients and
        never modified.
    row_keys : list[str], optional
        Passed to apply_pins (rarely needed — cascade_many outputs
        carry .attrs['cascade_row_keys']).

    Returns
    -------
    (edited_df, report)
        report — one row per involved node: node, role
        ('source'/'destination'/'bystander'), baseline_total,
        target_total, achieved_total, exact. Destinations land on
        their targets by construction; bystander rows verify the
        cancellation identity. Any inexact row warns.

    Notes
    -----
    Recipients must be SIBLINGS of `from_node` (same parent) — moving
    value across different parents changes both parents' totals, which
    is route_targets' job, not a redistribution.
    """
    df = quotas_long
    if not from_node or not isinstance(from_node, str):
        raise ValueError(f"from_node must be a node id string, "
                         f"got {from_node!r}.")
    if isinstance(weights, str):
        if weights not in ("proportional", "equal"):
            raise ValueError("weights must be 'proportional', 'equal', "
                             f"or a dict, got '{weights}'.")
    elif not isinstance(weights, dict):
        raise ValueError("weights must be 'proportional', 'equal', or a "
                         f"dict of node->share, got {type(weights)}.")

    mask = _scope_mask(df, scope)

    src = df[(df["node_id"] == from_node) & mask]
    if src.empty:
        raise ValueError(f"redistribute: '{from_node}' matches no rows "
                         f"(after scope {scope or {}}).")
    src_parents = {p for p in src["parent"] if pd.notna(p)}
    if not src_parents:
        raise ValueError(
            f"redistribute: '{from_node}' is a root — there is no "
            f"sibling group to conserve against. Reduce the root's "
            f"target (or use route_targets) instead.")
    e0 = float(src["base_quota"].sum())

    frozen = set(freeze_nodes or [])
    sib_rows = df[mask & df["parent"].isin(src_parents)
                  & (df["node_id"] != from_node)]
    all_sibs = [n for n in sib_rows["node_id"].unique()
                if n not in frozen]

    # ---- Resolve recipients + shares -------------------------------
    if isinstance(weights, dict):
        if to_nodes is not None and set(to_nodes) != set(weights):
            raise ValueError("to_nodes and dict weights disagree — pass "
                             "one or the other (the dict keys are the "
                             "recipients).")
        dests = list(weights)
        raw = {d: float(weights[d]) for d in dests}
        if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError("dict weights must be non-negative and sum "
                             "to a positive number.")
    else:
        dests = list(to_nodes) if to_nodes is not None else all_sibs
        raw = None
    if not dests:
        raise ValueError(f"redistribute: no eligible recipients for "
                         f"'{from_node}' (all siblings frozen?).")
    bad = [d for d in dests
           if d == from_node or d in frozen
           or df[(df["node_id"] == d) & mask].empty
           or set(df.loc[(df["node_id"] == d) & mask, "parent"].dropna())
           != src_parents]
    if bad:
        raise ValueError(
            f"redistribute: {bad} are not eligible recipients — each "
            f"must be an unfrozen SIBLING of '{from_node}' (same "
            f"parent, present in the scoped rows). For cross-parent "
            f"moves use route_targets.")

    base = {d: float(df.loc[(df["node_id"] == d) & mask,
                            "base_quota"].sum()) for d in dests}
    if raw is not None:
        total_w = sum(raw.values())
        share = {d: raw[d] / total_w for d in dests}
    elif weights == "equal":
        share = {d: 1.0 / len(dests) for d in dests}
    else:                                     # proportional
        pool = sum(base.values())
        if pool > 0:
            share = {d: base[d] / pool for d in dests}
        else:
            warnings.warn(
                f"redistribute: recipients of '{from_node}' have an "
                f"all-zero baseline — splitting equally.",
                UserWarning, stacklevel=2)
            share = {d: 1.0 / len(dests) for d in dests}

    # ---- Compose the pins and run ----------------------------------
    pins = [Pin(from_node, 0.0, scope=scope)]
    pins += [Pin(d, base[d] + share[d] * e0, scope=scope) for d in dests]
    edited = _run_pins_quietly(df, pins, freeze_nodes, row_keys)

    # ---- Verify + report --------------------------------------------
    emask = _scope_mask(edited, scope)

    def _after(node):
        return float(edited.loc[(edited["node_id"] == node) & emask,
                                "base_quota"].sum())

    rows = [{"node": from_node, "role": "source",
             "baseline_total": round(e0, 2), "target_total": 0.0,
             "achieved_total": round(_after(from_node), 2)}]
    for d in dests:
        rows.append({"node": d, "role": "destination",
                     "baseline_total": round(base[d], 2),
                     "target_total": round(base[d] + share[d] * e0, 2),
                     "achieved_total": round(_after(d), 2)})
    for b in all_sibs:
        if b in dests:
            continue
        b0 = float(df.loc[(df["node_id"] == b) & mask,
                          "base_quota"].sum())
        rows.append({"node": b, "role": "bystander",
                     "baseline_total": round(b0, 2),
                     "target_total": round(b0, 2),
                     "achieved_total": round(_after(b), 2)})
    report = pd.DataFrame(rows)
    report["exact"] = (report["achieved_total"]
                       - report["target_total"]).abs() <= 0.05
    if not report["exact"].all():
        off = report.loc[~report["exact"], "node"].tolist()
        warnings.warn(
            f"redistribute('{from_node}'): {off} did not land exactly "
            f"on target (frozen mass or $0 floors in the way) — see "
            f"the returned report.",
            UserWarning, stacklevel=2)
    return edited, report


def concentrate(
    quotas_long: pd.DataFrame,
    to_node: str,
    from_nodes: Optional[List[str]] = None,
    scope: Optional[Dict[str, Any]] = None,
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse siblings' (scoped) quota ONTO one sibling, zeroing them —
    the inverse of `redistribute` (issue #47).

    "All of CENTRAL's Migration lands on the CENTRAL6-MIGRATION team;
    every other CENTRAL team = 0" is::

        edited, report = concentrate(quotas_long, 'CENTRAL6-MIGRATION',
                                     scope={'st1_sales_type': 'Migration'})

    This is also the "route 100% of a parent's quota to a single
    child" mode (issue #29) and the "concentrate a group onto one
    team/subtree" helper (issue #44): the survivor gets the FULL
    parent pool (computed for you — no pre-computed hedged pool, no
    sibling enumeration), nothing leaks, and because hedged values
    re-derive from each row's own ratio the survivor carries the
    hedged pool automatically. A manager destination is detected from
    the graph (pin_type='subtree'), so team-vs-rep misclassification
    of ids like 'WEST6-MIGRATION' cannot happen.

    Thin sugar over `apply_pins`, exactly like `redistribute`: it pins
    `to_node` to the summed scoped baseline of `to_node + from_nodes`
    and pins each source to $0, so the whole pin contract applies —
    source subtrees zero and the destination subtree grows AT EVERY
    DEPTH (its internal mix preserved), the parent conserves, hedged
    values re-derive from each row's own ratio (#21), other scopes stay
    untouched, frozen nodes never move. No per-rep pins, no `exclude`,
    no hand-computed group total.

    Parameters
    ----------
    quotas_long : pd.DataFrame
        cascade_many output (or equivalent long frame).
    to_node : str
        The sibling that receives the whole group total (not a root).
    from_nodes : list[str], optional
        Which siblings to zero. Default: every unfrozen sibling of
        `to_node`. Siblings NOT listed become bystanders and are
        verified to stay exactly at baseline.
    scope : dict, optional
        Column=value filters (Pin.scope): only matching cascades move.
    freeze_nodes : list[str], optional
        Passed to apply_pins; frozen nodes are never zeroed and never
        modified (and are excluded from the default `from_nodes`).
    row_keys : list[str], optional
        Passed to apply_pins (rarely needed — cascade_many outputs
        carry .attrs['cascade_row_keys']).

    Returns
    -------
    (edited_df, report)
        report — one row per involved node: node, role
        ('destination'/'source'/'bystander'), baseline_total,
        target_total, achieved_total, exact. Any inexact row warns.

    Notes
    -----
    Sources must be SIBLINGS of `to_node` (same parent) — pulling value
    across different parents changes both parents' totals, which is
    route_targets' job, not a concentration.
    """
    df = quotas_long
    if not to_node or not isinstance(to_node, str):
        raise ValueError(f"to_node must be a node id string, "
                         f"got {to_node!r}.")
    mask = _scope_mask(df, scope)

    dest = df[(df["node_id"] == to_node) & mask]
    if dest.empty:
        raise ValueError(f"concentrate: '{to_node}' matches no rows "
                         f"(after scope {scope or {}}).")
    dest_parents = {p for p in dest["parent"] if pd.notna(p)}
    if not dest_parents:
        raise ValueError(
            f"concentrate: '{to_node}' is a root — there is no sibling "
            f"group to collapse. Use route_targets for cross-tree "
            f"moves.")

    frozen = set(freeze_nodes or [])
    sib_rows = df[mask & df["parent"].isin(dest_parents)
                  & (df["node_id"] != to_node)]
    all_sibs = [n for n in sib_rows["node_id"].unique()
                if n not in frozen]

    sources = list(from_nodes) if from_nodes is not None else all_sibs
    if not sources:
        raise ValueError(f"concentrate: no eligible sources for "
                         f"'{to_node}' (no unfrozen siblings).")
    bad = [s for s in sources
           if s == to_node or s in frozen
           or df[(df["node_id"] == s) & mask].empty
           or set(df.loc[(df["node_id"] == s) & mask, "parent"].dropna())
           != dest_parents]
    if bad:
        raise ValueError(
            f"concentrate: {bad} are not eligible sources — each must "
            f"be an unfrozen SIBLING of '{to_node}' (same parent, "
            f"present in the scoped rows). For cross-parent moves use "
            f"route_targets.")
    if len(set(sources)) != len(sources):
        raise ValueError("concentrate: duplicate nodes in from_nodes.")

    def _base(node, frame=df, m=mask):
        return float(frame.loc[(frame["node_id"] == node) & m,
                               "base_quota"].sum())

    d0 = _base(to_node)
    src_base = {s: _base(s) for s in sources}
    group_total = d0 + sum(src_base.values())

    # Sources FIRST, destination last (all same depth, so list order is
    # application order): with a bystander buffer present, zeroing the
    # sources first INFLATES the buffer and the destination pin then
    # sheds it back — the buffer never floors at $0, so its internal
    # mix survives. Destination-first would transiently floor the
    # buffer and equal-split its reps on the way back up.
    pins = [Pin(s, 0.0, scope=scope) for s in sources]
    pins += [Pin(to_node, group_total, scope=scope)]
    edited = _run_pins_quietly(df, pins, freeze_nodes, row_keys)

    # ---- Verify + report --------------------------------------------
    emask = _scope_mask(edited, scope)

    def _after(node):
        return float(edited.loc[(edited["node_id"] == node) & emask,
                                "base_quota"].sum())

    rows = [{"node": to_node, "role": "destination",
             "baseline_total": round(d0, 2),
             "target_total": round(group_total, 2),
             "achieved_total": round(_after(to_node), 2)}]
    for s in sources:
        rows.append({"node": s, "role": "source",
                     "baseline_total": round(src_base[s], 2),
                     "target_total": 0.0,
                     "achieved_total": round(_after(s), 2)})
    for b in all_sibs:
        if b in sources:
            continue
        b0 = _base(b)
        rows.append({"node": b, "role": "bystander",
                     "baseline_total": round(b0, 2),
                     "target_total": round(b0, 2),
                     "achieved_total": round(_after(b), 2)})
    report = pd.DataFrame(rows)
    report["exact"] = (report["achieved_total"]
                       - report["target_total"]).abs() <= 0.05
    if not report["exact"].all():
        off = report.loc[~report["exact"], "node"].tolist()
        warnings.warn(
            f"concentrate('{to_node}'): {off} did not land exactly on "
            f"target (frozen mass or $0 floors in the way) — see the "
            f"returned report.",
            UserWarning, stacklevel=2)
    return edited, report
