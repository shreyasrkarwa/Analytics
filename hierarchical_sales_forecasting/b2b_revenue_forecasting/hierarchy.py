import warnings

import networkx as nx
import pandas as pd
from typing import List, Dict, Any


# Valid values for from_dataframe(on_collision=...)
_COLLISION_POLICIES = ("suffix", "skip", "error")

# String cell values treated as missing, in addition to real NaN/None.
# Guards against CSVs read with keep_default_na=False, where blanks and
# literal "nan"/"none" strings would otherwise become real node ids and
# collapse unrelated branches into one shared node (issue #1).
# NOTE: "na" is deliberately NOT in this list — 'NA' is a common real
# region name (North America); the README's keep_default_na=False advice
# exists precisely to keep it as data.
_MISSING_STRINGS = ("", "nan", "none", "null")


def _is_missing_level(value: Any) -> bool:
    """True if a hierarchy cell should be treated as absent (jagged row)."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS:
        return True
    return False


_TRUTHY_STRINGS = ("true", "yes", "y", "t")
_FALSY_STRINGS = ("false", "no", "n", "f")

# Sentinel: distinguishes "could not coerce" from a legitimate None.
_UNCOERCIBLE = object()


def coerce_metric_value(value: Any):
    """
    Coerce one metric/gate cell into a clean Python numeric (issue #3).

    Returns
    -------
    bool | int | float — the coerced value, or the _UNCOERCIBLE sentinel
    when the input has no numeric interpretation.

    Rules (in order):
      - bool                        -> kept as bool (the cascader's
                                       boolean auto-detection relies on it)
      - int / float                 -> unchanged
      - numpy scalars (np.int64, np.float32, np.bool_, ...) -> unboxed via
                                       .item(); np.bool_ becomes bool
      - strings: "true"/"yes"/...   -> True, "false"/"no"/... -> False
                 numeric text       -> float, tolerating thousands commas,
                                       leading $/€/£ and trailing %
      - anything else               -> _UNCOERCIBLE

    Silent-zero behavior is what this replaces: pre-v0.6.1, a value like
    numpy.bool_(True) or "true" was stored raw and aggregated as 0 by the
    cascader — gating whole slices to $0 with no indication why.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    # numpy (and similar) scalars expose .item() -> native Python type
    if hasattr(value, "item"):
        try:
            item = value.item()
        except (ValueError, TypeError):
            return _UNCOERCIBLE
        if isinstance(item, bool) or isinstance(item, (int, float)):
            return item
        value = item  # e.g. np.str_ -> str; fall through to string parsing
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUTHY_STRINGS:
            return True
        if v in _FALSY_STRINGS:
            return False
        cleaned = v.replace(",", "").lstrip("$€£").rstrip("%").strip()
        try:
            return float(cleaned)
        except ValueError:
            return _UNCOERCIBLE
    return _UNCOERCIBLE


class HierarchyValidationError(ValueError):
    """
    Raised when the built hierarchy is not a DAG (contains a cycle or
    self-loop), or when from_dataframe(on_collision='error') encounters a
    row whose adjacent levels hold the same value.
    """
    pass


def _coerce_brand_new_flag(value: Any) -> bool:
    """
    Coerce a CSV cell into a brand-new boolean.

    Truthy: True, 1, "true", "yes", "y", "t" (case-insensitive)
    Falsy:  False, 0, "false", "no", "n", "f", "" (case-insensitive)
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "y", "t", "1"):
            return True
        if v in ("false", "no", "n", "f", "0", ""):
            return False
    # Unknown -> treat as not brand-new (safer default)
    return False


class SalesHierarchy:
    """
    Models a B2B Enterprise Org Chart as a Directed Acyclic Graph (DAG).
    Allows for flexible node depths (e.g., Global -> Region -> L2 Manager -> L1 Manager -> IC).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        # sanitized_id -> original value, populated when the on_collision
        # policy renames a duplicate-level node (issue #7). Lets outputs
        # carry an `original_id` column so analysis joins don't break.
        self.id_map: Dict[str, str] = {}
        
    def add_node(self, node_id: str, attributes: Dict[str, Any] = None):
        """Adds an entity (e.g., IC, Manager, or Region) to the reporting DAG."""
        if attributes is None:
            attributes = {}
        # Allows for updating metrics natively (like past 4 quarter performance)
        if node_id not in self.graph:
            self.graph.add_node(node_id, **attributes)
        else:
            nx.set_node_attributes(self.graph, {node_id: attributes})
            
    def add_edge(self, parent_id: str, child_id: str):
        """Creates a direct reporting relationship."""
        self.graph.add_edge(parent_id, child_id)
        
    def from_dataframe(self, df: pd.DataFrame, path_cols: List[str],
                       metrics_cols: List[str] = None,
                       brand_new_col: str = None,
                       on_collision: str = "suffix",
                       metadata_cols: List[str] = None):
        """
        Builds the hierarchy flexibly from a flattened organizational DataFrame.
        `path_cols` should outline the hierarchy from root to IC, e.g.,
        ['Global', 'Region', 'Second Level Manager', 'First Level Manager', 'IC'].
        Because the algorithm loops sequentially, it naturally supports 3 nodes or 10 nodes deep.

        Parameters
        ----------
        df : pd.DataFrame
            One row per IC.
        path_cols : List[str]
            Hierarchy columns from root to leaf.
        metrics_cols : List[str], optional
            Historical-performance columns to attach as attributes to the
            deepest node in each row. Accepts ANY metric names and ANY
            numeric data types — including booleans (True/False are stored
            as-is and aggregated as 1/0 by the cascader).
        brand_new_col : str, optional
            Name of a boolean / 0-1 column on the IC row marking that IC as
            a brand-new hire. Stored on the leaf node as a special attribute
            ('_is_brand_new'). QuotaCascader reads this when you call
            cascade_quota(new_ic_attr='_is_brand_new'), letting analysts
            keep all configuration in the same CSV instead of passing a
            separate Python list. Cells parsed as truthy
            (True / 1 / "true" / "yes") flag the IC as brand-new.
        metadata_cols : List[str], optional
            Descriptive (non-metric) columns — segment, geo, rep name,
            employee id, etc. — attached to the deepest node of each row
            AS-IS (no numeric coercion, never aggregated, never read by
            MetricSpecs). Retrieve them in outputs via
            quotas_to_dataframe(metadata_cols=[...]) (issue #7).
        on_collision : str
            (Suffix format note, issue #18: renamed ids take the STABLE
            form '<id>__<level_column>', e.g. 'T1__rep'. Never parse
            it back — outputs carry original_id / original_parent and
            cascade_many stamps attrs['id_map'].)
            What to do when a row repeats the same value at two levels
            (e.g., team 'T1' AND rep 'T1'). Pre-v0.6.0 this silently
            created a self-loop / cycle and later crashed cascade_quota
            with a RecursionError (issue #1).

            "suffix" (default)
                The deeper duplicate is renamed to
                '<value>__<level_column>' (e.g., 'T1__node_5_rep_no'),
                the edge is kept, and a single summary warning is
                emitted listing examples. Deterministic, so the same
                value renames identically across rows. Non-colliding
                node ids are unchanged.
            "skip"
                The deeper duplicate level is dropped from that row's
                path (treated like a jagged hierarchy). Note: if the
                dropped level was the row's deepest, metrics attach to
                the surviving (shallower) node.
            "error"
                Raise HierarchyValidationError naming the row and the
                colliding value. Strictest — forces the analyst to
                clean the CSV.

        Missing levels
        --------------
        Real NaN/None cells were always skipped (jagged hierarchies).
        Since v0.6.0, blank strings and literal "nan"/"none"/"null"/
        "na"/"n/a" strings (common with keep_default_na=False) are also
        treated as missing — previously they became real nodes named
        e.g. "nan" shared across unrelated branches, corrupting the
        graph.

        Validation
        ----------
        After building, the graph is checked with
        nx.is_directed_acyclic_graph. If a cycle survives (e.g., a name
        reused across two DIFFERENT levels in different rows), a
        HierarchyValidationError is raised naming the cycle — instead
        of a RecursionError from deep inside networkx at cascade time.
        """
        if on_collision not in _COLLISION_POLICIES:
            raise ValueError(
                f"on_collision must be one of {_COLLISION_POLICIES}, "
                f"got '{on_collision}'."
            )

        collision_examples = []    # (row_index, value, level_col) for the warning
        non_numeric_examples = []  # (row_index, metric_col, raw_value)

        for idx, row in df.iterrows():
            # --- Resolve this row's path: stringify levels, drop missing ones
            levels = []  # list of (node_id, level_col)
            for col in path_cols:
                val = row[col]
                if _is_missing_level(val):
                    continue  # jagged hierarchy — skip absent levels
                levels.append((str(val), col))

            if not levels:
                continue  # nothing usable on this row

            # --- Apply the collision policy along the row's path
            resolved = []          # list of (node_id, level_col)
            seen_ids = set()
            for node_id, col in levels:
                if node_id in seen_ids:
                    if on_collision == "error":
                        raise HierarchyValidationError(
                            f"Row {idx}: value '{node_id}' appears at more "
                            f"than one level (again at '{col}'). This would "
                            f"create a self-loop/cycle. Clean the data or "
                            f"use on_collision='suffix' / 'skip'."
                        )
                    if on_collision == "skip":
                        collision_examples.append((idx, node_id, col))
                        continue  # drop the duplicate level from this path
                    # "suffix": deterministic rename by level column
                    new_id = f"{node_id}__{col}"
                    while new_id in seen_ids:  # pathological repeats
                        new_id += "_"
                    collision_examples.append((idx, node_id, col))
                    self.id_map[new_id] = node_id  # sanitized -> original
                    node_id = new_id
                seen_ids.add(node_id)
                resolved.append((node_id, col))

            # --- Add nodes and edges; deepest resolved node gets the metrics
            for i, (node_id, _col) in enumerate(resolved):
                is_deepest = i == len(resolved) - 1
                if is_deepest:
                    attributes = {}
                    if metrics_cols:
                        for c in metrics_cols:
                            raw = row[c]
                            if pd.isna(raw):
                                continue
                            coerced = coerce_metric_value(raw)
                            if coerced is _UNCOERCIBLE:
                                # Warn + treat as missing (issue #3) — never
                                # store a value the cascader would read as 0.
                                non_numeric_examples.append((idx, c, raw))
                                continue
                            attributes[c] = coerced
                    if metadata_cols:
                        for c in metadata_cols:
                            if c in row.index and pd.notna(row[c]):
                                # Stored raw — descriptive data, not signal
                                attributes[c] = row[c]
                    if (brand_new_col and brand_new_col in row.index
                            and pd.notna(row[brand_new_col])):
                        attributes['_is_brand_new'] = _coerce_brand_new_flag(
                            row[brand_new_col])
                    self.add_node(node_id, attributes=attributes or None)
                else:
                    self.add_node(node_id)
                if i > 0:
                    self.add_edge(resolved[i - 1][0], node_id)

        if collision_examples:
            sample = "; ".join(
                f"row {i}: '{v}' at '{c}'" for i, v, c in collision_examples[:5]
            )
            action = ("renamed to '<value>__<level>'"
                      if on_collision == "suffix" else "dropped from the path")
            warnings.warn(
                f"{len(collision_examples)} duplicate-level value(s) detected "
                f"and {action} (on_collision='{on_collision}'). "
                f"Examples: {sample}",
                UserWarning,
                stacklevel=2,
            )

        if non_numeric_examples:
            sample = "; ".join(
                f"row {i}, column '{c}': {v!r}"
                for i, c, v in non_numeric_examples[:5]
            )
            warnings.warn(
                f"{len(non_numeric_examples)} metric value(s) could not be "
                f"coerced to a number and were treated as MISSING (they will "
                f"not contribute to cascades or gates). Examples: {sample}. "
                f"Clean these cells if they should carry signal.",
                UserWarning,
                stacklevel=2,
            )

        # --- Final safety net: catch cross-row cycles with a clear error
        self.validate()

    def validate(self) -> "SalesHierarchy":
        """
        Assert the graph is a DAG. Raises HierarchyValidationError naming
        the offending cycle (or self-loop) if not; returns self so it can
        be chained. Called automatically at the end of from_dataframe;
        call it manually after building a hierarchy via add_edge().
        """
        loops = list(nx.nodes_with_selfloops(self.graph))
        if loops:
            raise HierarchyValidationError(
                f"Hierarchy contains self-loop(s) at: {sorted(loops)[:10]}. "
                f"A node cannot report to itself."
            )
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            path = " -> ".join([cycle[0][0]] + [e[1] for e in cycle])
            raise HierarchyValidationError(
                f"Hierarchy is not a DAG — cycle found: {path}. This usually "
                f"means a name is reused at two different levels across rows."
            )
        return self

    # ------------------------------------------------------------------
    # Read-only accessors (issue #5) — `.graph` is the canonical name for
    # the underlying nx.DiGraph across the whole package; the helpers
    # below cover the common questions so consumers rarely need to touch
    # the raw graph at all.
    # ------------------------------------------------------------------
    @property
    def hierarchy(self):
        """
        Alias for `.graph` (issue #5). Some consumers naturally reach for
        `hierarchy.hierarchy` because QuotaCascader historically used that
        name internally — both now work, `.graph` is canonical.
        """
        return self.graph

    def roots(self) -> List[str]:
        """Nodes with no parent (usually exactly one: the global root)."""
        return [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]

    def leaves(self, root: str = None) -> List[str]:
        """
        All leaf (IC) nodes. With `root`, only leaves under that node
        (same as get_leaves); without, every leaf in the hierarchy.
        """
        if root is not None:
            return self.get_leaves(root)
        return [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]

    def managers(self, root: str = None) -> List[str]:
        """
        All non-leaf nodes (anyone with at least one direct report).
        With `root`, only managers strictly under that node.
        """
        if root is not None:
            return [n for n in nx.descendants(self.graph, root)
                    if self.graph.out_degree(n) > 0]
        return [n for n in self.graph.nodes if self.graph.out_degree(n) > 0]

    def node_depths(self) -> Dict[str, int]:
        """
        Depth of every node from the root(s); roots are depth 0. Nodes
        reachable from multiple roots keep their shallowest depth.
        Handy for per-level hedging or custom per-depth reports.
        """
        depths: Dict[str, int] = {}
        for root in self.roots():
            for node, d in nx.shortest_path_length(self.graph,
                                                   source=root).items():
                if node not in depths or d < depths[node]:
                    depths[node] = d
        return depths

    def get_children(self, node_id: str) -> List[str]:
        """Returns direct reports of a given node."""
        return list(self.graph.successors(node_id))

    def get_leaves(self, node_id: str) -> List[str]:
        """Returns all ICs (leaf nodes) reporting up under this node."""
        return [n for n in nx.descendants(self.graph, node_id) if self.graph.out_degree(n) == 0]
