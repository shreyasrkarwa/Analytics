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
_ON_CONFLICT = ("error", "warn", "allow", "narrower_wins")
_REASON_RANK = {"no_siblings": 1, "all_blocked": 2,
                "floors_at_zero": 3}   # genuine trumps intentional (#45)


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


def _row_ratios(df, key_of, child_ix, hedge=None):
    """Per-row cascaded/base hedge ratio (the #21 contract's carrier).

    Rows whose base is 0 cannot supply a ratio; before v0.40.0 they
    silently got 1.0, so a pin landing on a zeroed slice set
    base = cascaded, breaking the depth's hedge identity three levels
    away from the cause (issue #67). Now the ratio is DERIVED:
      1. explicit `hedge=` (float | {depth: cum_ratio} | HedgeByDepth,
         resolved like reconcile) — authoritative when given,
      2. else the mean ratio of same-(cascade, parent) siblings with
         base > 0,
      3. else the mean ratio of same-(cascade, depth) rows,
      4. else 1.0.
    Returns (ratio_dict, derived_dict) — derived_dict maps row index
    -> source string for rows whose ratio was derived, so callers can
    warn when such a row actually receives money."""
    ratio, zero_rows = {}, []
    for idx in df.index:
        b = df.at[idx, "base_quota"]
        if pd.notna(b) and float(b) != 0.0:
            ratio[idx] = float(df.at[idx, "cascaded_quota"]) / float(b)
        else:
            zero_rows.append(idx)
    derived: Dict[Any, str] = {}
    if not zero_rows:
        return ratio, derived

    has_depth = "depth" in df.columns
    expected = None
    if hedge is not None:
        from b2b_revenue_forecasting.quota_cascader import HedgeByDepth
        if isinstance(hedge, HedgeByDepth):
            import networkx as nx
            expected = {}
            groups: Dict[Any, List[Any]] = {}
            for idx in df.index:
                groups.setdefault(key_of.at[idx], []).append(idx)
            for k, idxs in groups.items():
                g = nx.DiGraph()
                node_of = {df.at[i, "node_id"]: i for i in idxs}
                g.add_nodes_from(node_of)
                for i in idxs:
                    p_ = df.at[i, "parent"]
                    if pd.notna(p_) and p_ in node_of:
                        g.add_edge(p_, df.at[i, "node_id"])
                mult = hedge.resolve(g)
                cum: Dict[str, float] = {}

                def _cum(n):
                    if n in cum:
                        return cum[n]
                    p_ = df.at[node_of[n], "parent"]
                    if pd.isna(p_) or p_ not in node_of:
                        cum[n] = 1.0
                    else:
                        cum[n] = _cum(p_) * float(mult.get(p_, 1.0))
                    return cum[n]
                for i in idxs:
                    expected[i] = _cum(df.at[i, "node_id"])
        elif isinstance(hedge, dict):
            expected = {idx: float(hedge.get(
                int(df.at[idx, "depth"]) if has_depth else 0, 1.0))
                for idx in zero_rows}
        elif isinstance(hedge, (int, float)):
            expected = {idx: float(hedge) ** (
                int(df.at[idx, "depth"]) if has_depth else 0)
                for idx in zero_rows}
        else:
            raise ValueError("hedge must be a float, a {depth: "
                             "cum_ratio} dict, or a HedgeByDepth spec.")

    depth_pool: Dict[Tuple[Any, int], List[Any]] = {}
    if has_depth:
        for idx in df.index:
            if idx in ratio:
                depth_pool.setdefault(
                    (key_of.at[idx], int(df.at[idx, "depth"])),
                    []).append(idx)
    for idx in zero_rows:
        if expected is not None and idx in expected:
            ratio[idx] = expected[idx]
            derived[idx] = "hedge"
            continue
        k = key_of.at[idx]
        sibs = [s for s in child_ix.get((k, df.at[idx, "parent"]), [])
                if s in ratio]
        if sibs:
            ratio[idx] = sum(ratio[s] for s in sibs) / len(sibs)
            derived[idx] = "siblings"
            continue
        pool = (depth_pool.get((k, int(df.at[idx, "depth"])), [])
                if has_depth else [])
        if pool:
            ratio[idx] = sum(ratio[s] for s in pool) / len(pool)
            derived[idx] = "same-depth rows"
        else:
            ratio[idx] = 1.0
            derived[idx] = "none (1.0)"
    return ratio, derived


def _expand_subset(df, seeds, pins=None, caller="apply_pins"):
    """Subset fast path (issue #64): expand user seed nodes to the
    operation's CLOSURE so slice-and-stitch can never truncate a
    family. Seeds expand to their full subtrees; for pins, each pin's
    parent's full subtree (its absorption domain) is added
    mechanically. A pin whose node lies outside the seeds' subtrees
    raises — the caller scoped the wrong region. Returns the expanded
    node-id set."""
    all_nodes = set(df["node_id"])
    unknown = [s for s in seeds if s not in all_nodes]
    if unknown:
        raise ValueError(f"{caller}: subset node(s) not in "
                         f"quotas_long: {unknown}")
    pairs = df[["node_id", "parent"]].drop_duplicates()
    kids: Dict[Any, List[str]] = {}
    parent_of: Dict[str, Any] = {}
    for n, p in zip(pairs["node_id"], pairs["parent"]):
        parent_of.setdefault(n, p if pd.notna(p) else None)
        if pd.notna(p):
            kids.setdefault(p, []).append(n)

    def _subtree(n):
        out, stack = set(), [n]
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(kids.get(cur, []))
        return out

    seed_exp: set = set()
    for s in seeds:
        seed_exp |= _subtree(s)
    expanded = set(seed_exp)
    for pin in (pins or []):
        if pin.node not in all_nodes:
            continue                        # on_missing's business
        if pin.node not in seed_exp:
            raise ValueError(
                f"{caller}: pin node '{pin.node}' is outside "
                f"subset={sorted(seeds)} — widen the subset to a "
                f"region containing it, or drop subset=.")
        p = parent_of.get(pin.node)
        expanded |= _subtree(p) if p is not None else _subtree(pin.node)
    return expanded


def _warn_derived_used(derived_used: Dict[Any, str], df, caller: str):
    """One summary warning when zero-baseline rows actually received
    money via a derived ratio (issue #67)."""
    if not derived_used:
        return
    by_src: Dict[str, List[str]] = {}
    for idx, src in derived_used.items():
        by_src.setdefault(src, []).append(str(df.at[idx, "node_id"]))
    bits = "; ".join(
        f"{len(nodes)} row(s) from {src} (e.g. {sorted(set(nodes))[:3]})"
        for src, nodes in by_src.items())
    warnings.warn(
        f"{caller}: zero-baseline row(s) received money — base/cascaded "
        f"derived via inferred hedge ratio: {bits}. Pass hedge= to "
        f"apply_pins to make the ratio authoritative.",
        UserWarning, stacklevel=3)


def apply_pins(
    quotas_long: pd.DataFrame,
    pins: List[Pin],
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
    on_missing: str = "error",
    on_overshoot: str = "allow",
    hedge: Any = None,
    on_conflict: str = "error",
    subset: Optional[List[str]] = None,
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

    on_overshoot : str
        Cross-pin envelope policy (issue #55). Individually-valid pins
        can COLLECTIVELY push a parent's children past its total in
        some cascade slice — a gap no per-pin row can see. After all
        pins land, apply_pins checks every (cascade, parent) identity
        and always attaches edited.attrs['overshoot_report'] (records:
        cascade keys, parent node_id, gap). 'allow' (default): keep
        the result, emit ONE summary warning when gaps exist.
        'scale_pins': resolve by running enforce_identities (#54) —
        pinned children scale down proportionally to fit, free rows
        floor first; scaled pins get overshoot_scaled=True and a
        recomputed achieved_total in the feasibility report.
        'error': raise, naming the offending (parent, cascade) slices.

    hedge : float | dict | HedgeByDepth, optional
        Authoritative hedge spec for ZERO-BASELINE rows (issue #67).
        A row with base==0 cannot supply its own cascaded/base ratio;
        before v0.40.0 it silently got 1.0, so pinning a zeroed slice
        set base = cascaded and broke the depth's hedge identity.
        Now the ratio is DERIVED — from `hedge=` when given (resolved
        exactly like reconcile: float compounds per depth, dict is
        {depth: cumulative_ratio}, HedgeByDepth resolves per cascade),
        else inferred from same-(cascade, parent) siblings with
        base > 0, else same-depth rows, else 1.0. Whenever such a row
        actually receives money, ONE summary warning names the rows
        and the ratio source. Rows with base > 0 always use their own
        ratio — hedge= never re-hedges existing rows.

    on_conflict : str
        Overlapping-pin policy (issue #63). Two pins CONFLICT when
        their matched row sets intersect — same node, overlapping (or
        identical/unscoped) scopes; the later-applied pin silently
        overwrites the earlier one's rows, so the result satisfies
        only the last writer. Cross-node parent/child pins are NOT
        conflicts (deliberate composition — #39/#42 — with the #55
        envelope check covering their arithmetic). 'error' (default):
        raise before applying anything, listing each family's pins,
        scopes, row counts and relation (subset/identical/partial).
        'warn': same text as one warning, then last-writer-wins.
        'allow': proceed silently. 'narrower_wins': the scoped pins
        stand and the broadest pin constrains only its REMAINDER rows
        with total − Σ(narrower totals) — requires clean nesting
        (every narrower pin a strict subset of one broadest pin,
        narrower pins mutually disjoint, one basis, non-negative
        remainder; violations raise naming the numbers). In every
        non-error mode the report gains `conflict` (family
        description) and `adjusted_total` (narrower_wins remainder),
        and EVERY pin's achieved_total is recomputed over its
        original rows on the FINAL frame — a defeated pin can never
        report feasible=True.

    subset : Optional[List[str]]
        Fast path for large frames (issue #64): seed node ids; the
        run is confined to their full subtrees PLUS each pin's
        absorption domain (its parent's full subtree), computed by
        the library so the slice can never truncate a family. The
        stitch preserves the original index, row order, and attrs —
        the three traps of hand-rolled slice-and-stitch (pd.concat
        drops attrs['cascade_row_keys'], silently changing key
        resolution downstream). A pin outside the seeds' subtrees
        raises. Identical output to the full-frame run, only faster.

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
            skipped and reason for on_missing bookkeeping, and
            (v0.29.0, issue #45) unabsorbed_reason + intentional:
            'no_siblings' (root pin — nothing to absorb, by
            construction), 'all_blocked' (the caller's own pins /
            excludes / freezes emptied the absorber set — e.g. a fully
            specified partition whose deltas cancel), or
            'floors_at_zero' (GENUINE: free capacity existed but hit
            $0). Only 'floors_at_zero' warns; intentional=True marks
            unabsorbed money fully explained by the caller's
            construction. Real problems are
            report[~report.intentional & ~report.feasible].
    """
    if on_missing not in _ON_MISSING:
        raise ValueError(f"on_missing must be one of {_ON_MISSING}, "
                         f"got '{on_missing}'.")
    if on_conflict not in _ON_CONFLICT:
        raise ValueError(f"on_conflict must be one of {_ON_CONFLICT}, "
                         f"got '{on_conflict}'.")
    if on_overshoot not in _ON_OVERSHOOT:
        raise ValueError(f"on_overshoot must be one of {_ON_OVERSHOOT}, "
                         f"got '{on_overshoot}'.")
    required = {"node_id", "parent", "depth", "base_quota",
                "cascaded_quota"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)}.")
    # ---- Subset fast path (issue #64): slice -> run -> stitch --------
    # The library computes the closure (pin absorption domains), keeps
    # the original index and row order, and carries attrs through — the
    # three traps of hand-rolled slice-and-stitch.
    if subset is not None:
        expanded = _expand_subset(quotas_long, list(subset), pins=pins,
                                  caller="apply_pins")
        sub = quotas_long[quotas_long["node_id"].isin(expanded)]
        sub.attrs = dict(quotas_long.attrs)
        edited, report = apply_pins(
            sub, pins, freeze_nodes=freeze_nodes, row_keys=row_keys,
            on_missing=on_missing, on_overshoot=on_overshoot,
            hedge=hedge, on_conflict=on_conflict)
        out = quotas_long.copy()
        for col, default in (("is_pinned", False), ("pin_type", None)):
            if col not in out.columns:
                out[col] = default
        out.loc[edited.index, list(edited.columns)] = edited
        out.attrs = {**dict(quotas_long.attrs), **dict(edited.attrs)}
        return out, report

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
    # would silently split key groups — issue #40 hygiene). Built from
    # column arrays (not df.apply) — ~30x faster on wide frames (#64).
    key_of = (pd.Series(
                  [tuple(None if pd.isna(v) else v for v in row)
                   for row in zip(*(df[c].tolist() for c in keys))],
                  index=df.index) if keys
              else pd.Series([()] * len(df), index=df.index))

    # Uniqueness: one row per (cascade, node)
    if df.groupby([key_of, df["node_id"]]).size().max() > 1:
        raise ValueError(
            "quotas_long has multiple rows per (cascade, node) under the "
            f"inferred keys {keys} — pass row_keys= listing the columns "
            "that identify one cascade (group keys + e.g. fiscal_quarter)."
        )

    # Row lookup: (key, node_id) -> index ; children: (key, parent) -> [idx]
    row_ix: Dict[Tuple[Any, str], Any] = {}
    child_ix: Dict[Tuple[Any, Any], List[Any]] = {}
    for idx in df.index:
        k = key_of.at[idx]
        row_ix[(k, df.at[idx, "node_id"])] = idx
        child_ix.setdefault((k, df.at[idx, "parent"]), []).append(idx)

    # Original hedge ratio per row index (edits preserve it). Rows with
    # base==0 get a DERIVED ratio (#67): hedge= > siblings > same-depth.
    ratio, ratio_derived = _row_ratios(df, key_of, child_ix, hedge=hedge)
    derived_used: Dict[Any, str] = {}

    def _descendants(k, node_id) -> List[Any]:
        out, stack = [], [node_id]
        while stack:
            for cidx in child_ix.get((k, stack.pop()), []):
                out.append(cidx)
                stack.append(df.at[cidx, "node_id"])
        return out

    def _set_base(idx, new_base: float) -> None:
        if idx in ratio_derived and abs(new_base) > 0.005:
            derived_used[idx] = ratio_derived[idx]
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
    pin_row_sets: Dict[int, frozenset] = {}
    # One node_id index instead of a full-frame mask per pin (#64):
    # 400 pins on a 100K-row frame is 400 O(rows-of-node) lookups, not
    # 400 full-column scans.
    node_pos = df.groupby("node_id").indices
    for i, pin in enumerate(pins):
        for col in pin.scope:
            if col not in df.columns:
                raise ValueError(f"Pin scope column '{col}' not in "
                                 f"quotas_long.")
        pos = node_pos.get(pin.node)
        if pos is None:
            sel = df.iloc[:0]
        else:
            sel = df.iloc[pos]
            for col, val in pin.scope.items():
                sel = sel[sel[col] == val]
        if not len(sel):
            if on_missing == "error":
                raise ValueError(
                    f"Pin node '{pin.node}' matches no rows (after "
                    f"scope {pin.scope}). Pass on_missing='skip' (or "
                    f"'warn') to drop such pins into the feasibility "
                    f"report instead of aborting the batch.")
            skipped_reason[i] = ("node_absent"
                                 if pin.node not in all_node_ids
                                 else "empty_scope")
        else:
            pin_row_sets[i] = frozenset(sel.index)
    if skipped_reason and on_missing == "warn":
        named = [f"{pins[i].node} ({r})"
                 for i, r in sorted(skipped_reason.items())]
        warnings.warn(
            f"apply_pins: skipped {len(skipped_reason)} pin(s) with no "
            f"matching rows: {', '.join(named)}. See the feasibility "
            f"report (skipped / reason columns).",
            UserWarning, stacklevel=2)

    # ---- Overlap detection (issue #63) ------------------------------
    # Two pins CONFLICT when their matched row sets intersect (same
    # node, overlapping scopes) — the later-applied one silently
    # overwrites the earlier. Cross-node parent/child pins are NOT
    # conflicts: that composition is deliberate (#39/#42) and the #55
    # envelope check covers its arithmetic.
    eff_rows: Dict[int, List[Any]] = {i: sorted(rs)
                                      for i, rs in pin_row_sets.items()}
    eff_total: Dict[int, float] = {i: pins[i].total for i in pin_row_sets}
    conflict_note: Dict[int, str] = {}
    adjusted: Dict[int, float] = {}
    by_node: Dict[str, List[int]] = {}
    for i in pin_row_sets:
        by_node.setdefault(pins[i].node, []).append(i)
    families: List[Tuple[str, List[int]]] = []
    for node, idxs_ in by_node.items():
        remaining = set(idxs_)
        while remaining:
            comp = {remaining.pop()}
            grew = True
            while grew:
                grew = False
                for j in list(remaining):
                    if any(pin_row_sets[j] & pin_row_sets[c]
                           for c in comp):
                        comp.add(j)
                        remaining.discard(j)
                        grew = True
            if len(comp) > 1:
                families.append((node, sorted(comp)))
    if families:
        def _fam_desc(node, fam):
            broadest = max(fam, key=lambda i: len(pin_row_sets[i]))
            lines = []
            for i in fam:
                rel = ("" if i == broadest else
                       "  (subset)" if pin_row_sets[i]
                       < pin_row_sets[broadest] else
                       "  (identical)" if pin_row_sets[i]
                       == pin_row_sets[broadest] else
                       "  (partial overlap)")
                lines.append(
                    f"    Pin(total={pins[i].total:>13,.2f}, "
                    f"scope={pins[i].scope or {}}) covers "
                    f"{len(pin_row_sets[i])} row(s){rel}")
            return f"  pin conflict on '{node}':\n" + "\n".join(lines)

        msg = ("apply_pins: overlapping pins constrain the same rows "
               "— the result would satisfy only the last-applied "
               "pin:\n"
               + "\n".join(_fam_desc(n, f) for n, f in families)
               + "\n  -> use on_conflict='narrower_wins' (scoped pins "
                 "stand, the broader total constrains the remainder) "
                 "| 'warn' | 'allow' to proceed.")
        if on_conflict == "error":
            raise ValueError(msg)
        if on_conflict == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
        if on_conflict == "narrower_wins":
            for node, fam in families:
                broad = max(fam, key=lambda i: len(pin_row_sets[i]))
                rest = [i for i in fam if i != broad]
                bset = pin_row_sets[broad]
                if not all(pin_row_sets[i] < bset for i in rest):
                    raise ValueError(
                        f"apply_pins(on_conflict='narrower_wins'): "
                        f"pins on '{node}' overlap PARTIALLY (or are "
                        f"identical) — narrower_wins needs every "
                        f"narrower pin to be a strict subset of one "
                        f"broadest pin.\n{_fam_desc(node, fam)}")
                for a in range(len(rest)):
                    for b in range(a + 1, len(rest)):
                        if pin_row_sets[rest[a]] & pin_row_sets[rest[b]]:
                            raise ValueError(
                                f"apply_pins(on_conflict="
                                f"'narrower_wins'): the narrower pins "
                                f"on '{node}' overlap EACH OTHER — "
                                f"they must be disjoint.\n"
                                f"{_fam_desc(node, fam)}")
                if len({pins[i].basis for i in fam}) > 1:
                    raise ValueError(
                        f"apply_pins(on_conflict='narrower_wins'): "
                        f"pins on '{node}' mix basis='base' and "
                        f"basis='cascaded' — the remainder total is "
                        f"not well-defined across layers.")
                covered = set().union(*(pin_row_sets[i] for i in rest))
                rem_rows = sorted(bset - covered)
                rem_total = (pins[broad].total
                             - sum(pins[i].total for i in rest))
                if rem_total < -0.005:
                    raise ValueError(
                        f"apply_pins(on_conflict='narrower_wins'): on "
                        f"'{node}' the narrower pins sum to "
                        f"{sum(pins[i].total for i in rest):,.2f}, "
                        f"EXCEEDING the broader total "
                        f"{pins[broad].total:,.2f} — infeasible.")
                if not rem_rows and abs(rem_total) > 0.005:
                    raise ValueError(
                        f"apply_pins(on_conflict='narrower_wins'): "
                        f"the narrower pins on '{node}' cover EVERY "
                        f"row of the broader pin but leave "
                        f"{rem_total:,.2f} of its total unplaced.")
                eff_rows[broad] = rem_rows
                eff_total[broad] = rem_total
                adjusted[broad] = rem_total
        mode_note = {"warn": "warned", "allow": "allowed",
                     "narrower_wins": "narrower_wins"}[on_conflict]
        for node, fam in families:
            for i in fam:
                conflict_note[i] = (
                    f"overlaps pins {[j for j in fam if j != i]} on "
                    f"'{node}' ({mode_note})")

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
    def _pin_depth(i: int) -> float:
        rs = pin_row_sets.get(i)
        if not rs:
            return float("inf")
        return float(min(df.at[idx, "depth"] for idx in rs))

    application_order = sorted(
        (i for i in range(len(pins)) if i not in skipped_reason),
        key=lambda i: (_pin_depth(i), i))
    report_rows: List[Optional[dict]] = [None] * len(pins)
    for i, reason in skipped_reason.items():
        report_rows[i] = {
            "pin_node": pins[i].node, "pin_type": None,
            "basis": pins[i].basis,
            "requested_total": round(pins[i].total, 2),
            "baseline_total": 0.0, "achieved_total": 0.0,
            "rows_affected": 0, "absorbed": 0.0, "unabsorbed": 0.0,
            "subtree_shortfall": 0.0, "feasible": False,
            "unabsorbed_reason": None, "intentional": False,
            "skipped": True, "reason": reason,
        }

    for pin_i in application_order:
        pin = pins[pin_i]
        basis_col = "base_quota" if pin.basis == "base" else "cascaded_quota"
        node_idx = eff_rows[pin_i]
        tot = eff_total[pin_i]
        if not node_idx:
            # narrower_wins: the broader pin was fully expressed by its
            # subset pins (remainder $0 over zero rows) — nothing to do.
            report_rows[pin_i] = {
                "pin_node": pin.node, "pin_type": None,
                "basis": pin.basis,
                "requested_total": round(pin.total, 2),
                "baseline_total": 0.0, "achieved_total": 0.0,
                "rows_affected": 0, "absorbed": 0.0, "unabsorbed": 0.0,
                "subtree_shortfall": 0.0, "feasible": True,
                "unabsorbed_reason": None, "intentional": False,
                "skipped": False, "reason": None,
            }
            continue
        is_subtree = any(_descendants(key_of.at[i], pin.node)
                         for i in node_idx)
        pin_type = "subtree" if is_subtree else "leaf"

        baseline_total = float(df.loc[node_idx, basis_col].sum())
        # Per-row allocation: proportional to baseline mix, equal if flat 0
        if baseline_total > 0:
            alloc = {i: tot * float(df.at[i, basis_col]) / baseline_total
                     for i in node_idx}
        else:
            warnings.warn(
                f"Pin '{pin.node}': baseline total is 0 across matched rows "
                f"— splitting {tot:,.2f} equally across "
                f"{len(node_idx)} row(s).",
                UserWarning, stacklevel=2,
            )
            alloc = {i: tot / len(node_idx) for i in node_idx}

        # Protection set (issue #39): nodes pinned by ANY pin, frozen
        # nodes, and this pin's exclude list keep their current values
        # everywhere — including INSIDE a pinned manager's subtree and
        # inside absorbing siblings' subtrees. This makes pin order
        # irrelevant: a later manager pin rescales AROUND an earlier
        # descendant pin instead of trampling it.
        protected = (all_pinned | frozen | set(pin.exclude)) - {pin.node}

        absorbed_sum, unabsorbed_sum, shortfall_sum = 0.0, 0.0, 0.0
        unabsorbed_reason = None   # issue #45

        def _note_reason(r):
            nonlocal unabsorbed_reason
            if (unabsorbed_reason is None
                    or _REASON_RANK[r] > _REASON_RANK[unabsorbed_reason]):
                unabsorbed_reason = r
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
            sib_all = [s for s in child_ix.get((k, df.at[idx, "parent"]),
                                                [])
                       if df.at[s, "node_id"] != pin.node]
            sibs = [s for s in sib_all
                    if df.at[s, "node_id"] not in protected]
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
                if delta - absorb > 0.005:
                    # Why could it not fit? (issue #45)
                    _note_reason("floors_at_zero" if pool > 0
                                 else ("all_blocked" if sib_all
                                       else "no_siblings"))
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
                        _note_reason("all_blocked" if sib_all
                                     else "no_siblings")

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
            "unabsorbed_reason": (unabsorbed_reason
                                  if unabsorbed_sum > 0.01 else None),
            "intentional": bool(
                unabsorbed_sum > 0.01
                and unabsorbed_reason in ("no_siblings", "all_blocked")
                and shortfall_sum <= 0.01),
            "skipped": False, "reason": None,
        })  # slot pin_i: report emitted in INPUT pin order (issue #41)
        if (unabsorbed_sum > 0.01
                and unabsorbed_reason == "floors_at_zero"):
            # Intentional cases (#45) — root pins (no_siblings) and
            # caller-specified partitions (all_blocked) — land in the
            # report (unabsorbed_reason / intentional) WITHOUT a
            # warning: data, not noise. Only genuine floors warn.
            warnings.warn(
                f"Pin '{pin.node}': {unabsorbed_sum:,.2f} could not be "
                f"absorbed — free siblings floored at $0. Parents will "
                f"not fully conserve; see the feasibility report.",
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

    # Final-frame audit (issue #63): with overlapping pins allowed, a
    # later pin can overwrite an earlier one's rows — recompute every
    # pin's achieved_total over its ORIGINAL row set on the FINAL
    # frame, so a defeated pin can never report feasible=True.
    for i, rs in pin_row_sets.items():
        row = report_rows[i]
        if row is None:
            continue
        bcol = ("base_quota" if pins[i].basis == "base"
                else "cascaded_quota")
        ach = float(df.loc[sorted(rs), bcol].sum())
        row["achieved_total"] = round(ach, 2)
        if abs(ach - pins[i].total) > 0.01:
            row["feasible"] = False
    for i, row in enumerate(report_rows):
        if row is not None:
            row["conflict"] = conflict_note.get(i)
            row["adjusted_total"] = (round(adjusted[i], 2)
                                     if i in adjusted else None)

    _warn_derived_used(derived_used, df, "apply_pins")

    # ---- Cross-pin envelope check (issue #55) -----------------------
    # Individually-valid pins can collectively break a parent-child
    # identity in some cascade slice. Detect per (cascade, parent);
    # never silent.
    overshoot_records: List[Dict[str, Any]] = []
    for idx in df.index:
        node = df.at[idx, "node_id"]
        k = key_of.at[idx]
        kids = child_ix.get((k, node), [])
        if not kids:
            continue
        gap = (sum(float(df.at[c, "base_quota"]) for c in kids)
               - float(df.at[idx, "base_quota"]))
        if abs(gap) > 0.05:
            overshoot_records.append(
                {**(dict(zip(keys, k)) if keys else {}),
                 "node_id": node, "gap": round(gap, 2)})
    report_df = pd.DataFrame(report_rows)
    if overshoot_records:
        if on_overshoot == "error":
            raise ValueError(
                f"apply_pins: pins collectively break "
                f"{len(overshoot_records)} parent-child identit(ies): "
                f"{overshoot_records[:5]}... Use "
                f"on_overshoot='scale_pins' to fit pins "
                f"proportionally, or 'allow' to keep the gaps "
                f"(attrs['overshoot_report']).")
        if on_overshoot in ("scale_pins", "rebalance"):
            with warnings.catch_warnings(record=True) as _w:
                warnings.simplefilter("always")
                df, _enf = enforce_identities(
                    df, on_overshoot=on_overshoot, row_keys=keys)
            for w_ in _w:
                warnings.warn(w_.message, w_.category, stacklevel=2)
            # Recompute per-pin achievement after scaling
            report_df["overshoot_scaled"] = False
            for i_, pin in enumerate(pins):
                if report_df.at[i_, "skipped"]:
                    continue
                basis_col = ("base_quota" if pin.basis == "base"
                             else "cascaded_quota")
                mask = df["node_id"] == pin.node
                for col, val in pin.scope.items():
                    mask &= df[col] == val
                achieved = float(df.loc[mask, basis_col].sum())
                if abs(achieved
                       - report_df.at[i_, "achieved_total"]) > 0.05:
                    report_df.at[i_, "overshoot_scaled"] = True
                    report_df.at[i_, "feasible"] = False
                report_df.at[i_, "achieved_total"] = round(achieved, 2)
        else:                                            # 'allow'
            warnings.warn(
                f"apply_pins: pins collectively break "
                f"{len(overshoot_records)} parent-child identit(ies) — "
                f"see edited.attrs['overshoot_report'], or rerun with "
                f"on_overshoot='scale_pins' / use enforce_identities().",
                UserWarning, stacklevel=2)
    df.attrs["overshoot_report"] = overshoot_records
    return df, report_df


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


_ON_OVERSHOOT = ("scale_pins", "error", "allow", "rebalance")


def enforce_identities(
    quotas_long: pd.DataFrame,
    on_overshoot: str = "scale_pins",
    tolerance: float = 0.05,
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
    anchor: str = "root",
    subset: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    reconcile() that FIXES instead of reporting (issues #54/#58/#59):
    force every parent-child identity per cascade slice.

    subset= (issue #64): pass seed node ids to confine the run to
    those nodes' full subtrees — the per-region pattern that turns a
    whole-frame pass into a slice pass. The library does the
    slice-and-stitch itself: original index and row order preserved,
    attrs carried through (a hand-rolled pd.concat stitch silently
    DROPS attrs['cascade_row_keys'], changing key resolution — the
    real footgun behind "row order changed my results"). Identities
    of parents OUTSIDE the subset are not enforced — run reconcile()
    on the full frame afterwards if you need that proof.

    Anchors (issue #58)
    -------------------
    anchor='root' (default) — parents are budgets, children conform:
      pinned children hold, free children scale to fill, and on
      overshoot the `on_overshoot` policy decides (below).
    anchor='leaves' — the DUAL: children stand, every parent is
      DERIVED as the exact sum of its children, bottom-up, all the way
      to the root (which floats). Pins are never touched; conservation
      holds by construction. `on_overshoot` is ignored.

    on_overshoot (anchor='root')
    ----------------------------
    'scale_pins' (default) — per (cascade, parent) slice: free rows
      floor first, then pinned children scale down proportionally to
      fit (factors recorded per row and named in the warning).
      CAUTION (#59): this is per-combo and one-directional; when a
      node's AGGREGATE is correct but combos deliberately over/under-
      shoot (concentrate / resplit_by_metric composed with aggregate
      pins), it removes the overshoot without restoring the
      undershoot — use 'rebalance'.
    'rebalance' (#59) — processed bottom-up ACROSS combos: when a
      node's aggregate matches its children's aggregate (within
      `tolerance`), the node's per-combo values FLOAT to the child
      sums — the aggregate is conserved, so AGGREGATE PINS on that
      node stay exact (a Pin is an aggregate total by definition).
      Only a genuinely-off aggregate falls back to per-combo
      scale-down (subtrees rescaled proportionally, factors recorded).
    'error' — raise naming (parent, cascade, gap).
    'allow' — leave gaps in place (recorded).

    Everything else is unchanged from v0.35.0: pinned = is_pinned
    provenance + freeze_nodes; pins are NEVER scaled up implicitly;
    cascaded re-derives from each row's own ratio (#21) so hedged
    identities restore automatically (pinned by reconcile() in tests);
    share_of_parent recomputed; clean frames return bit-identical.

    Returns (edited_df, report): one row per adjusted / broken
    (cascade, parent): cascade keys, node_id, gap_before, action
    ('scaled_free' | 'scaled_pins' | 'rebalanced' | 'floated' |
    'left'), factor (pin scale factor where applicable), pins_scaled,
    resolved.
    """
    from b2b_revenue_forecasting.batch import _cascade_key_context

    if on_overshoot not in _ON_OVERSHOOT:
        raise ValueError(f"on_overshoot must be one of {_ON_OVERSHOOT}, "
                         f"got '{on_overshoot}'.")
    if anchor not in ("root", "leaves"):
        raise ValueError(f"anchor must be 'root' or 'leaves', "
                         f"got '{anchor}'.")
    required = {"node_id", "parent", "base_quota", "cascaded_quota"}
    missing = required - set(quotas_long.columns)
    if missing:
        raise ValueError(f"quotas_long is missing required columns "
                         f"{sorted(missing)}.")

    if subset is not None:
        expanded = _expand_subset(quotas_long, list(subset),
                                  caller="enforce_identities")
        sub = quotas_long[quotas_long["node_id"].isin(expanded)]
        sub.attrs = dict(quotas_long.attrs)
        edited, report = enforce_identities(
            sub, on_overshoot=on_overshoot, tolerance=tolerance,
            freeze_nodes=freeze_nodes, row_keys=row_keys, anchor=anchor)
        out = quotas_long.copy()
        out.loc[edited.index, list(edited.columns)] = edited
        out.attrs = dict(quotas_long.attrs)
        return out, report

    df = quotas_long.copy()
    keys, key_of, row_ix, child_ix = _cascade_key_context(
        df, row_keys, exclude_cols=[], caller="enforce_identities")

    frozen = set(freeze_nodes or [])
    has_pin_col = "is_pinned" in df.columns

    def _pinned(idx) -> bool:
        node = df.at[idx, "node_id"]
        if node in frozen:
            return True
        return bool(has_pin_col and df.at[idx, "is_pinned"] == True)  # noqa: E712

    ratio, ratio_derived = _row_ratios(df, key_of, child_ix)
    derived_used: Dict[Any, str] = {}

    def _set_base(idx, new_base: float) -> None:
        if abs(new_base - float(df.at[idx, "base_quota"])) < 0.005:
            return
        if idx in ratio_derived and abs(new_base) > 0.005:
            derived_used[idx] = ratio_derived[idx]
        df.at[idx, "base_quota"] = round(new_base, 2)
        df.at[idx, "cascaded_quota"] = round(new_base * ratio[idx], 2)

    def _scale_subtree(k, idx, factor: float) -> None:
        for c in child_ix.get((k, df.at[idx, "node_id"]), []):
            _set_base(c, float(df.at[c, "base_quota"]) * factor)
            _scale_subtree(k, c, factor)

    report_rows: List[Dict[str, Any]] = []
    scaled_pin_notes: List[str] = []

    groups: Dict[Any, List[Any]] = {}
    for idx in df.index:
        groups.setdefault(key_of.at[idx], []).append(idx)

    def _bfs_order(k, idxs):
        in_group = set(df.loc[idxs, "node_id"])
        queue = [idx for idx in idxs
                 if pd.isna(df.at[idx, "parent"])
                 or df.at[idx, "parent"] not in in_group]
        order, seen = [], set()
        while queue:
            idx = queue.pop(0)
            if idx in seen:
                continue
            seen.add(idx)
            order.append(idx)
            queue.extend(child_ix.get((k, df.at[idx, "node_id"]), []))
        return order

    def _fix_children(k, idx, kd, propagate: bool) -> None:
        """anchor='root' per-(cascade, parent) fix: pinned hold, free
        fill, on_overshoot policy. With propagate=True, adjusted
        children's subtrees rescale proportionally (used by the
        rebalance fallback, which runs bottom-up)."""
        node = df.at[idx, "node_id"]
        kids = child_ix.get((k, node), [])
        if not kids:
            return
        target = float(df.at[idx, "base_quota"])
        pin_k = [c for c in kids if _pinned(c)]
        free_k = [c for c in kids if not _pinned(c)]
        pin_sum = sum(float(df.at[c, "base_quota"]) for c in pin_k)
        free_sum = sum(float(df.at[c, "base_quota"]) for c in free_k)
        gap = (pin_sum + free_sum) - target
        if abs(gap) <= tolerance:
            return
        old_vals = {c: float(df.at[c, "base_quota"]) for c in kids}

        def _apply(c, new):
            _set_base(c, new)
            if propagate and old_vals[c] > 0:
                f = new / old_vals[c]
                if abs(f - 1.0) > 1e-9:
                    _scale_subtree(k, c, f)

        remainder = target - pin_sum
        factor = None
        if remainder >= -tolerance and free_k:
            remainder = max(remainder, 0.0)
            for c in free_k:
                cb = old_vals[c]
                share = (remainder * cb / free_sum if free_sum > 0
                         else remainder / len(free_k))
                _apply(c, share)
            action, resolved, scaled = "scaled_free", True, []
        elif remainder < -tolerance:
            if on_overshoot == "error":
                raise ValueError(
                    f"enforce_identities: pinned children of '{node}' "
                    f"in cascade {kd or '<single>'} sum to "
                    f"{pin_sum:,.2f} against a parent budget of "
                    f"{target:,.2f} (overshoot {pin_sum - target:,.2f})."
                    f" Use on_overshoot='scale_pins'/'rebalance', or "
                    f"'allow' to keep the gap.")
            for c in free_k:
                _apply(c, 0.0)
            if (on_overshoot in ("scale_pins", "rebalance")
                    and pin_sum > 0):
                factor = max(target, 0.0) / pin_sum
                scaled = []
                for c in pin_k:
                    _apply(c, old_vals[c] * factor)
                    scaled.append(df.at[c, "node_id"])
                    scaled_pin_notes.append(
                        f"{df.at[c, 'node_id']}@{kd or '<single>'}: "
                        f"x{factor:.4f}")
                action, resolved = "scaled_pins", True
            else:
                action, resolved, scaled = "left", False, []
        else:
            action, resolved, scaled = "left", False, []
        report_rows.append({**kd, "node_id": node,
                            "gap_before": round(gap, 2),
                            "action": action,
                            "factor": (round(factor, 6)
                                       if factor is not None else None),
                            "pins_scaled": scaled,
                            "resolved": resolved})

    if anchor == "leaves":
        # Bottom-up rebuild (#58): parents derived as exact child sums,
        # root floats. Pins never touched; conservation by construction.
        for k, idxs in groups.items():
            kd = dict(zip(keys, k)) if keys else {}
            for idx in reversed(_bfs_order(k, idxs)):
                node = df.at[idx, "node_id"]
                kids = child_ix.get((k, node), [])
                if not kids:
                    continue
                new = sum(float(df.at[c, "base_quota"]) for c in kids)
                gap = new - float(df.at[idx, "base_quota"])
                if abs(gap) > tolerance:
                    report_rows.append({**kd, "node_id": node,
                                        "gap_before": round(gap, 2),
                                        "action": "floated",
                                        "factor": None,
                                        "pins_scaled": [],
                                        "resolved": True})
                _set_base(idx, new)
    elif on_overshoot == "rebalance":
        # Bottom-up across combos (#59): float a node's per-combo
        # values to its child sums whenever the node's AGGREGATE is
        # conserved; only genuinely-off aggregates fall back to
        # per-combo scaling (subtrees propagated).
        node_rows: Dict[str, List[Tuple[Any, Any]]] = {}
        node_depth: Dict[str, int] = {}
        for k, idxs in groups.items():
            depth_map = {}
            for pos, idx in enumerate(_bfs_order(k, idxs)):
                node = df.at[idx, "node_id"]
                p_ = df.at[idx, "parent"]
                depth_map[node] = (depth_map.get(p_, -1) + 1
                                   if pd.notna(p_) else 0)
                node_rows.setdefault(node, []).append((k, idx))
                node_depth[node] = max(node_depth.get(node, 0),
                                       depth_map[node])
        parents = [n for n in node_rows
                   if any(child_ix.get((k, n)) for k, _ in node_rows[n])]
        for node in sorted(parents, key=lambda n: -node_depth[n]):
            entries = node_rows[node]
            s = {k: sum(float(df.at[c, "base_quota"])
                        for c in child_ix.get((k, node), []))
                 for k, _ in entries}
            p = {k: float(df.at[idx, "base_quota"])
                 for k, idx in entries}
            if abs(sum(s.values()) - sum(p.values())) <= tolerance:
                for k, idx in entries:
                    if abs(s[k] - p[k]) > tolerance:
                        report_rows.append(
                            {**(dict(zip(keys, k)) if keys else {}),
                             "node_id": node,
                             "gap_before": round(s[k] - p[k], 2),
                             "action": "rebalanced", "factor": None,
                             "pins_scaled": [], "resolved": True})
                    _set_base(idx, s[k])
            else:
                for k, idx in entries:
                    _fix_children(k, idx,
                                  dict(zip(keys, k)) if keys else {},
                                  propagate=True)
    else:
        # anchor='root', top-down (v0.35.0 behavior, + factor audit)
        for k, idxs in groups.items():
            kd = dict(zip(keys, k)) if keys else {}
            for idx in _bfs_order(k, idxs):
                _fix_children(k, idx, kd, propagate=False)

    if "share_of_parent" in df.columns:
        for idx in df.index:
            k = key_of.at[idx]
            p_ = df.at[idx, "parent"]
            if pd.isna(p_) or (k, p_) not in row_ix:
                df.at[idx, "share_of_parent"] = 1.0
                continue
            pb = float(df.at[row_ix[(k, p_)], "base_quota"])
            nb = float(df.at[idx, "base_quota"])
            df.at[idx, "share_of_parent"] = (round(nb / pb, 6)
                                             if pb != 0
                                             else float("nan"))

    report = pd.DataFrame(report_rows)
    if scaled_pin_notes:
        shown = scaled_pin_notes[:6]
        more = ("" if len(scaled_pin_notes) <= 6
                else f" (+{len(scaled_pin_notes) - 6} more)")
        warnings.warn(
            f"enforce_identities: scaled pinned node(s) down to fit — "
            f"{'; '.join(shown)}{more}. Their pinned totals are no "
            f"longer exact; see the report's factor column.",
            UserWarning, stacklevel=2)
    if len(report) and not report["resolved"].all():
        left = report[~report.resolved]["node_id"].tolist()
        warnings.warn(
            f"enforce_identities: {len(left)} identity gap(s) left "
            f"unresolved ({left}) — no free children to adjust (or "
            f"on_overshoot='allow').",
            UserWarning, stacklevel=2)
    _warn_derived_used(derived_used, df, "enforce_identities")
    return df, report


def reallocate(
    quotas_long: pd.DataFrame,
    sources: Union[str, List[str]],
    recipients: Optional[List[str]] = None,
    fraction: float = 1.0,
    weights: Union[str, Dict[str, float]] = "proportional",
    scope: Optional[Dict[str, Any]] = None,
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generalized sibling move (issue #57): take a FRACTION of one or
    more sources' (scoped) quota and hand it to named recipients under
    an explicit split — "cut 75% of these reps' Cloud quota and move
    it 60/40 to those two reps" is one call::

        edited, report = reallocate(
            quotas_long, sources=['r1', 'r2'], fraction=0.75,
            weights={'r3': 0.6, 'r4': 0.4},
            scope={'base_product_r4f': 'Cloud'})

    `redistribute(x, ...)` is exactly `reallocate([x], fraction=1.0,
    ...)` (pinned by test). Thin sugar over apply_pins like its
    siblings: each source is pinned to (1 - fraction) x its baseline,
    each recipient to baseline + share x moved total — so the whole
    pin contract applies (per-depth subtree reshaping, parent
    conservation, per-row hedge re-derivation, scope isolation,
    freeze, $0 floors), and unlisted siblings are VERIFIED to stay at
    baseline.

    Parameters mirror redistribute; `fraction` must be in (0, 1];
    dict `weights` keys are the recipients (normalized); sources and
    recipients must all be siblings (same parent) and disjoint.

    Returns (edited_df, report): node, role ('source' / 'destination'
    / 'bystander'), baseline_total, target_total, achieved_total,
    exact — source targets are (1 - fraction) x baseline.
    """
    df = quotas_long
    if isinstance(sources, str):
        sources = [sources]
    if not sources or not all(isinstance(s, str) and s for s in sources):
        raise ValueError("reallocate: sources must be one or more node "
                         "id strings.")
    if len(set(sources)) != len(sources):
        raise ValueError("reallocate: duplicate nodes in sources.")
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError(f"reallocate: fraction must be in (0, 1], "
                         f"got {fraction!r}.")
    if isinstance(weights, str):
        if weights not in ("proportional", "equal"):
            raise ValueError("weights must be 'proportional', 'equal', "
                             f"or a dict, got '{weights}'.")
    elif not isinstance(weights, dict):
        raise ValueError("weights must be 'proportional', 'equal', or a "
                         f"dict of node->share, got {type(weights)}.")

    mask = _scope_mask(df, scope)
    frozen = set(freeze_nodes or [])
    src_parents: Optional[set] = None
    for s_ in sources:
        rows_ = df[(df["node_id"] == s_) & mask]
        if rows_.empty:
            raise ValueError(f"reallocate: source '{s_}' matches no "
                             f"rows (after scope {scope or {}}).")
        if s_ in frozen:
            raise ValueError(f"reallocate: source '{s_}' is frozen.")
        parents_ = {p for p in rows_["parent"] if pd.notna(p)}
        if not parents_:
            raise ValueError(f"reallocate: '{s_}' is a root — no "
                             f"sibling group to move within.")
        if src_parents is None:
            src_parents = parents_
        elif parents_ != src_parents:
            raise ValueError(
                f"reallocate: sources are not siblings of each other "
                f"('{s_}' has parents {sorted(parents_)}). For "
                f"cross-parent moves use route_targets.")

    sib_rows = df[mask & df["parent"].isin(src_parents)
                  & ~df["node_id"].isin(sources)]
    all_sibs = [n for n in sib_rows["node_id"].unique()
                if n not in frozen]

    if isinstance(weights, dict):
        if recipients is not None and set(recipients) != set(weights):
            raise ValueError("recipients and dict weights disagree — "
                             "pass one or the other (the dict keys are "
                             "the recipients).")
        dests = list(weights)
        raw = {d: float(weights[d]) for d in dests}
        if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError("dict weights must be non-negative and sum "
                             "to a positive number.")
    else:
        dests = list(recipients) if recipients is not None else all_sibs
        raw = None
    if not dests:
        raise ValueError("reallocate: no eligible recipients (all "
                         "siblings frozen or listed as sources?).")
    bad = [d for d in dests
           if d in sources or d in frozen
           or df[(df["node_id"] == d) & mask].empty
           or set(df.loc[(df["node_id"] == d) & mask, "parent"].dropna())
           != src_parents]
    if bad:
        raise ValueError(
            f"reallocate: {bad} are not eligible recipients — each must "
            f"be an unfrozen SIBLING of the sources (same parent, in "
            f"the scoped rows, not itself a source). For cross-parent "
            f"moves use route_targets.")

    def _base(node):
        return float(df.loc[(df["node_id"] == node) & mask,
                            "base_quota"].sum())

    src_base = {s_: _base(s_) for s_ in sources}
    moved = fraction * sum(src_base.values())
    dest_base = {d: _base(d) for d in dests}
    if raw is not None:
        tot = sum(raw.values())
        share = {d: raw[d] / tot for d in dests}
    elif weights == "equal":
        share = {d: 1.0 / len(dests) for d in dests}
    else:
        pool = sum(dest_base.values())
        if pool > 0:
            share = {d: dest_base[d] / pool for d in dests}
        else:
            warnings.warn("reallocate: recipients have an all-zero "
                          "baseline — splitting equally.",
                          UserWarning, stacklevel=2)
            share = {d: 1.0 / len(dests) for d in dests}

    pins = [Pin(s_, (1.0 - fraction) * src_base[s_], scope=scope)
            for s_ in sources]
    pins += [Pin(d, dest_base[d] + share[d] * moved, scope=scope)
             for d in dests]
    edited = _run_pins_quietly(df, pins, freeze_nodes, row_keys)

    emask = _scope_mask(edited, scope)

    def _after(node):
        return float(edited.loc[(edited["node_id"] == node) & emask,
                                "base_quota"].sum())

    rows = []
    for s_ in sources:
        rows.append({"node": s_, "role": "source",
                     "baseline_total": round(src_base[s_], 2),
                     "target_total": round((1 - fraction)
                                           * src_base[s_], 2),
                     "achieved_total": round(_after(s_), 2)})
    for d in dests:
        rows.append({"node": d, "role": "destination",
                     "baseline_total": round(dest_base[d], 2),
                     "target_total": round(dest_base[d]
                                           + share[d] * moved, 2),
                     "achieved_total": round(_after(d), 2)})
    for b_ in all_sibs:
        if b_ in dests:
            continue
        b0 = _base(b_)
        rows.append({"node": b_, "role": "bystander",
                     "baseline_total": round(b0, 2),
                     "target_total": round(b0, 2),
                     "achieved_total": round(_after(b_), 2)})
    report = pd.DataFrame(rows)
    report["exact"] = (report["achieved_total"]
                       - report["target_total"]).abs() <= 0.05
    if not report["exact"].all():
        off = report.loc[~report["exact"], "node"].tolist()
        warnings.warn(f"reallocate: {off} did not land exactly on "
                      f"target — see the returned report.",
                      UserWarning, stacklevel=2)
    return edited, report


def resplit_by_metric(
    quotas_long: pd.DataFrame,
    node: str,
    metric: str,
    scope: Optional[Dict[str, Any]] = None,
    freeze_nodes: Optional[List[str]] = None,
    row_keys: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Re-split `node`'s CHILDREN proportional to a carried metric column
    (issue #57): "re-split this team's Migration by dc_seats" is::

        edited, report = resplit_by_metric(
            quotas_long, 'CENTRAL6-MIGRATION', 'dc_seats',
            scope={'st1_sales_type': 'Migration'})

    Per cascade slice: each child's weight is the SUM of `metric` over
    its descendant leaves (the metric must be ON the frame — carry it
    with cascade_many(metadata_cols=[...], issue #16); the rollup is
    computed internally, same semantics as rollup_metrics). Frozen
    children hold their values; free children split the remaining
    parent budget by metric shares (equal split if the free metric
    pool is 0, with a warning); child subtrees rescale proportionally;
    hedged values re-derive from each row's own ratio (#21) and
    share_of_parent is recomputed — reconcile() stays clean.

    NOTE: this is a DELIBERATE overwrite of the node's internal
    allocation — is_pinned provenance on children is NOT honored here
    (that's the point of a re-split); use freeze_nodes to protect
    specific children.

    Returns (edited_df, report): one row per (cascade, child) —
    cascade keys, node_id, frozen, metric_sum, metric_share, old_base,
    new_base, exact.
    """
    from b2b_revenue_forecasting.batch import _cascade_key_context

    df = quotas_long.copy()
    if metric not in df.columns:
        raise ValueError(
            f"resplit_by_metric: '{metric}' not in quotas_long. Carry "
            f"metric columns onto leaf rows with cascade_many("
            f"metadata_cols=[...]) (v0.8.0; it carries metric values "
            f"too — issue #16).")
    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"resplit_by_metric: '{metric}' is not "
                         f"numeric.")
    keys, key_of, row_ix, child_ix = _cascade_key_context(
        df, row_keys, exclude_cols=[metric],
        caller="resplit_by_metric")
    frozen = set(freeze_nodes or [])

    ratio, ratio_derived = _row_ratios(df, key_of, child_ix)
    derived_used: Dict[Any, str] = {}

    def _set_base(idx, new_base):
        if abs(new_base - float(df.at[idx, "base_quota"])) < 0.005:
            return
        if idx in ratio_derived and abs(new_base) > 0.005:
            derived_used[idx] = ratio_derived[idx]
        df.at[idx, "base_quota"] = round(new_base, 2)
        df.at[idx, "cascaded_quota"] = round(new_base * ratio[idx], 2)

    def _metric_sum(k, idx):
        kids = child_ix.get((k, df.at[idx, "node_id"]), [])
        if not kids:
            v = df.at[idx, metric]
            return float(v) if pd.notna(v) else 0.0
        return sum(_metric_sum(k, c) for c in kids)

    def _scale_subtree(k, idx, new_total):
        old = float(df.at[idx, "base_quota"])
        kids = child_ix.get((k, df.at[idx, "node_id"]), [])
        _set_base(idx, new_total)
        if not kids:
            return
        if old > 0:
            for c in kids:
                _scale_subtree(k, c,
                               float(df.at[c, "base_quota"])
                               * new_total / old)
        else:
            for c in kids:
                _scale_subtree(k, c, new_total / len(kids))

    smask = _scope_mask(df, scope)
    node_idxs = [i for i in df.index
                 if df.at[i, "node_id"] == node and smask.at[i]]
    if not node_idxs:
        raise ValueError(f"resplit_by_metric: '{node}' matches no rows "
                         f"(after scope {scope or {}}).")
    report_rows: List[Dict[str, Any]] = []
    for idx in node_idxs:
        k = key_of.at[idx]
        kd = dict(zip(keys, k)) if keys else {}
        kids = child_ix.get((k, node), [])
        if not kids:
            raise ValueError(f"resplit_by_metric: '{node}' has no "
                             f"children in cascade "
                             f"{kd or '<single>'} — nothing to "
                             f"re-split.")
        budget = float(df.at[idx, "base_quota"])
        froz = [c for c in kids if df.at[c, "node_id"] in frozen]
        free = [c for c in kids if df.at[c, "node_id"] not in frozen]
        froz_sum = sum(float(df.at[c, "base_quota"]) for c in froz)
        remaining = max(budget - froz_sum, 0.0)
        msums = {c: _metric_sum(k, c) for c in free}
        pool = sum(msums.values())
        if pool <= 0 and free:
            warnings.warn(
                f"resplit_by_metric: free children of '{node}' in "
                f"cascade {kd or '<single>'} have a zero '{metric}' "
                f"pool — splitting equally.",
                UserWarning, stacklevel=2)
        for c in kids:
            cid = df.at[c, "node_id"]
            old = float(df.at[c, "base_quota"])
            if c in froz:
                new = old
                mshare = None
            else:
                mshare = (msums[c] / pool if pool > 0
                          else 1.0 / len(free))
                new = remaining * mshare
                _scale_subtree(k, c, new)
            report_rows.append({**kd, "node_id": cid,
                                "frozen": c in froz,
                                "metric_sum": (round(msums.get(c, 0.0), 4)
                                               if c in free else None),
                                "metric_share": (round(mshare, 6)
                                                 if mshare is not None
                                                 else None),
                                "old_base": round(old, 2),
                                "new_base": round(new, 2)})
        if froz_sum - budget > 0.05:
            warnings.warn(
                f"resplit_by_metric: frozen children of '{node}' in "
                f"cascade {kd or '<single>'} hold "
                f"{froz_sum - budget:,.2f} more than the parent budget "
                f"— free children floored at $0.",
                UserWarning, stacklevel=2)

    if "share_of_parent" in df.columns:
        for idx in df.index:
            k = key_of.at[idx]
            p_ = df.at[idx, "parent"]
            if pd.isna(p_) or (k, p_) not in row_ix:
                df.at[idx, "share_of_parent"] = 1.0
                continue
            pb = float(df.at[row_ix[(k, p_)], "base_quota"])
            nb = float(df.at[idx, "base_quota"])
            df.at[idx, "share_of_parent"] = (round(nb / pb, 6)
                                             if pb != 0
                                             else float("nan"))
    report = pd.DataFrame(report_rows)
    if len(report):
        achieved = []
        for _, r_ in report.iterrows():
            m2 = pd.Series(True, index=df.index)
            for c_, v_ in (scope or {}).items():
                m2 &= df[c_] == v_
            for c_ in keys:
                if c_ in r_ and c_ in df.columns:
                    m2 &= df[c_] == r_[c_]
            m2 &= df["node_id"] == r_["node_id"]
            achieved.append(round(float(df.loc[m2, "base_quota"].sum()),
                                  2))
        report["achieved_base"] = achieved
        report["exact"] = (report["achieved_base"]
                           - report["new_base"]).abs() <= 0.05
    _warn_derived_used(derived_used, df, "resplit_by_metric")
    return df, report
