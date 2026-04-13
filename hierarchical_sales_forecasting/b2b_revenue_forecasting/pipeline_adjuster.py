import networkx as nx
import pandas as pd
from typing import Dict, Set, Union, Any

# Default coverage thresholds used when no per-node overrides are provided
_DEFAULT_THRESHOLDS = {'healthy': 3.0, 'at_risk': 1.5}


class PipelineAdjuster:
    """
    Post-cascade pipeline health analyzer and quota adjuster.
    
    After QuotaCascader distributes targets top-down, the PipelineAdjuster evaluates
    pipeline coverage at each node and optionally redistributes quota among ICs within
    the same manager based on pipeline health.
    
    Key design constraints:
      - Only IC-level quotas are adjusted; manager and above quotas are NEVER changed.
      - Redistribution is zero-sum within each manager's team.
      - Coverage thresholds are fully configurable per-node with ancestor inheritance.
      - Locked nodes (CRO-mandated quotas) are excluded from adjustment.
    """
    
    def __init__(self, hierarchy, cascaded_quotas: Dict[str, float],
                 pipeline_attr: str = 'Current_Pipeline'):
        """
        Args:
            hierarchy:        SalesHierarchy object (with pipeline metrics on IC nodes).
            cascaded_quotas:  Dict from QuotaCascader.cascade_quota().
            pipeline_attr:    Name of the pipeline attribute on IC nodes.
                              Defaults to 'Current_Pipeline'.
        """
        self.graph = hierarchy.graph
        self.cascaded_quotas = cascaded_quotas.copy()
        self.pipeline_attr = pipeline_attr
    
    def _get_node_pipeline(self, node_id: str) -> float:
        """
        Recursively calculates the total pipeline for a node by summing the
        pipeline attribute of all leaf nodes (ICs) underneath it.
        """
        if self.graph.out_degree(node_id) == 0:
            return self.graph.nodes[node_id].get(self.pipeline_attr, 0.0)
        
        total = 0.0
        for child in self.graph.successors(node_id):
            total += self._get_node_pipeline(child)
        return total
    
    def _resolve_threshold(self, node_id: str, coverage_thresholds: Dict) -> Dict[str, float]:
        """
        Resolves the coverage threshold for a node by walking up the hierarchy
        until a matching ancestor is found in coverage_thresholds, or falls back
        to '_default'.
        
        Inheritance order: node itself → parent → grandparent → ... → '_default'
        
        Args:
            node_id:             The node to resolve thresholds for.
            coverage_thresholds: Dict mapping node IDs to {'healthy': float, 'at_risk': float}.
                                 Must include a '_default' key as fallback.
        
        Returns:
            Dict with 'healthy' and 'at_risk' keys.
        """
        # Check the node itself
        if node_id in coverage_thresholds:
            return coverage_thresholds[node_id]
        
        # Walk up ancestors
        predecessors = list(self.graph.predecessors(node_id))
        while predecessors:
            parent = predecessors[0]
            if parent in coverage_thresholds:
                return coverage_thresholds[parent]
            predecessors = list(self.graph.predecessors(parent))
        
        # Fallback to default
        return coverage_thresholds.get('_default', _DEFAULT_THRESHOLDS)
    
    def _classify_risk(self, coverage_ratio: float, threshold: Dict[str, float]) -> str:
        """Classifies a node's risk status based on coverage ratio vs threshold."""
        if coverage_ratio >= threshold['healthy']:
            return 'Healthy'
        elif coverage_ratio >= threshold['at_risk']:
            return 'Moderate'
        elif coverage_ratio >= 1.0:
            return 'At Risk'
        else:
            return 'Critical'
    
    def _get_node_level(self, node_id: str) -> int:
        """Returns the depth of a node from the root (root = 0)."""
        # Find the shortest path length from any root to this node
        roots = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        for root in roots:
            try:
                return nx.shortest_path_length(self.graph, root, node_id)
            except nx.NetworkXNoPath:
                continue
        return -1
    
    def diagnose(self, coverage_thresholds: Dict = None) -> pd.DataFrame:
        """
        Returns a diagnostic DataFrame showing pipeline coverage and risk status
        for every node in the hierarchy.
        
        Args:
            coverage_thresholds: Dict mapping node IDs (or level names) to their
                threshold config. Each value is a dict with 'healthy' and 'at_risk' keys.
                Must include a '_default' key as fallback.
                
                Example:
                    {
                        'NA':       {'healthy': 1.5, 'at_risk': 0.8},
                        'EMEA':     {'healthy': 2.5, 'at_risk': 1.2},
                        'APAC':     {'healthy': 3.0, 'at_risk': 1.5},
                        '_default': {'healthy': 2.0, 'at_risk': 1.0}
                    }
                
                Nodes inherit their ancestor's threshold if not explicitly set.
        
        Returns:
            pd.DataFrame with columns:
            [Node, Level, Cascaded_Quota, Pipeline, Coverage_Ratio, Risk_Status]
        """
        if coverage_thresholds is None:
            coverage_thresholds = {'_default': _DEFAULT_THRESHOLDS.copy()}
        elif '_default' not in coverage_thresholds:
            coverage_thresholds['_default'] = _DEFAULT_THRESHOLDS.copy()
        
        rows = []
        for node in self.cascaded_quotas:
            quota = self.cascaded_quotas[node]
            pipeline = self._get_node_pipeline(node)
            coverage = pipeline / quota if quota > 0 else 0.0
            threshold = self._resolve_threshold(node, coverage_thresholds)
            risk = self._classify_risk(coverage, threshold)
            level = self._get_node_level(node)
            
            rows.append({
                'Node': node,
                'Level': level,
                'Cascaded_Quota': round(quota, 2),
                'Pipeline': round(pipeline, 2),
                'Coverage_Ratio': round(coverage, 4),
                'Healthy_Threshold': threshold['healthy'],
                'At_Risk_Threshold': threshold['at_risk'],
                'Risk_Status': risk
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values(['Level', 'Node']).reset_index(drop=True)
    
    def adjust(self, mode: str = 'flag_only', coverage_thresholds: Dict = None,
               max_adjustment_pct: float = 0.20,
               locked_nodes: Union[Dict[str, float], Set[str]] = None) -> Dict[str, float]:
        """
        Adjusts cascaded quotas based on pipeline health.
        
        Args:
            mode: 
                'flag_only'    — Returns original quotas unchanged; use diagnose() for the report.
                'redistribute' — Redistributes quota among ICs within the same manager
                                 based on pipeline coverage. Zero-sum within each manager.
            
            coverage_thresholds: Same per-node config as diagnose().
                Example:
                    {
                        'NA':       {'healthy': 1.5, 'at_risk': 0.8},
                        'APAC':     {'healthy': 3.0, 'at_risk': 1.5},
                        '_default': {'healthy': 2.0, 'at_risk': 1.0}
                    }
            
            max_adjustment_pct: Maximum percentage any IC's quota can change (default 0.20 = 20%).
                                Acts as a safety rail to prevent aggressive swings.
            
            locked_nodes: ICs whose quotas cannot be adjusted.
                - If a Dict[str, float], keys are IC IDs and values are locked quota amounts.
                - If a Set[str], those IC IDs keep their cascaded quota as-is.
                These ICs are excluded from both donor and receiver pools.
        
        Returns:
            Dict[str, float] — Adjusted quotas (same shape as cascade_quota output).
            In 'flag_only' mode, returns original quotas unchanged.
        """
        if mode not in ('flag_only', 'redistribute'):
            raise ValueError(f"mode must be 'flag_only' or 'redistribute', got '{mode}'")
        
        if coverage_thresholds is None:
            coverage_thresholds = {'_default': _DEFAULT_THRESHOLDS.copy()}
        elif '_default' not in coverage_thresholds:
            coverage_thresholds['_default'] = _DEFAULT_THRESHOLDS.copy()
        
        # Normalize locked_nodes
        if locked_nodes is None:
            locked_set = set()
            locked_quotas = {}
        elif isinstance(locked_nodes, dict):
            locked_set = set(locked_nodes.keys())
            locked_quotas = locked_nodes.copy()
        else:
            locked_set = set(locked_nodes)
            locked_quotas = {}
        
        # Start with a copy of cascaded quotas
        adjusted = self.cascaded_quotas.copy()
        
        # Apply locked quota overrides
        for node, fixed_quota in locked_quotas.items():
            if node in adjusted:
                adjusted[node] = fixed_quota
        
        if mode == 'flag_only':
            return adjusted
        
        # --- Redistribute mode ---
        # Only adjust at IC level, within each manager's team
        
        # Find all manager nodes (nodes whose children are all ICs / leaf nodes)
        manager_nodes = []
        for node in self.graph.nodes:
            children = list(self.graph.successors(node))
            if children and all(self.graph.out_degree(c) == 0 for c in children):
                manager_nodes.append(node)
        
        for manager in manager_nodes:
            ics = list(self.graph.successors(manager))
            
            # Separate locked vs adjustable ICs
            adjustable_ics = [ic for ic in ics if ic not in locked_set]
            
            if len(adjustable_ics) < 2:
                continue  # Need at least 2 adjustable ICs to redistribute
            
            # Calculate coverage for each adjustable IC
            ic_data = {}
            for ic in adjustable_ics:
                quota = adjusted.get(ic, 0.0)
                pipeline = self._get_node_pipeline(ic)
                coverage = pipeline / quota if quota > 0 else 0.0
                threshold = self._resolve_threshold(ic, coverage_thresholds)
                ic_data[ic] = {
                    'quota': quota,
                    'pipeline': pipeline,
                    'coverage': coverage,
                    'threshold': threshold
                }
            
            # Identify donors (coverage >= healthy) and receivers (coverage < at_risk)
            donors = {ic: data for ic, data in ic_data.items()
                      if data['coverage'] >= data['threshold']['healthy']}
            receivers = {ic: data for ic, data in ic_data.items()
                         if data['coverage'] < data['threshold']['at_risk']}
            
            if not donors or not receivers:
                continue  # Nothing to redistribute in this team
            
            # Calculate surplus from donors (capped at max_adjustment_pct)
            total_surplus = 0.0
            donor_contributions = {}
            for ic, data in donors.items():
                max_give = data['quota'] * max_adjustment_pct
                # Proportional contribution: how far above healthy they are
                excess_ratio = (data['coverage'] - data['threshold']['healthy']) / data['coverage']
                contribution = min(max_give, data['quota'] * excess_ratio)
                contribution = max(0.0, contribution)  # Safety: never negative
                donor_contributions[ic] = contribution
                total_surplus += contribution
            
            if total_surplus <= 0:
                continue
            
            # Calculate receiver needs (how far below at_risk they are)
            receiver_needs = {}
            total_need = 0.0
            for ic, data in receivers.items():
                # Need is proportional to gap below at_risk threshold
                if data['coverage'] > 0:
                    gap = (data['threshold']['at_risk'] - data['coverage']) / data['threshold']['at_risk']
                else:
                    gap = 1.0  # Maximum need for ICs with zero coverage
                need = max(0.0, gap)
                receiver_needs[ic] = need
                total_need += need
            
            if total_need <= 0:
                continue
            
            # Distribute surplus to receivers proportionally, respecting max_adjustment_pct
            for ic in receivers:
                weight = receiver_needs[ic] / total_need
                raw_increase = total_surplus * weight
                max_receive = ic_data[ic]['quota'] * max_adjustment_pct
                actual_increase = min(raw_increase, max_receive)
                adjusted[ic] = ic_data[ic]['quota'] + actual_increase
            
            # Deduct from donors — proportionally to their contribution
            actual_distributed = sum(adjusted[ic] - ic_data[ic]['quota'] for ic in receivers)
            if actual_distributed > 0 and total_surplus > 0:
                scale = actual_distributed / total_surplus
                for ic in donors:
                    deduction = donor_contributions[ic] * scale
                    adjusted[ic] = ic_data[ic]['quota'] - deduction
        
        return adjusted
