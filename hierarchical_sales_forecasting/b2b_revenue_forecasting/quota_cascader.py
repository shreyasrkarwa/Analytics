import networkx as nx
from typing import Dict, Any

class QuotaCascader:
    def __init__(self, hierarchy):
        """
        Initializes the cascader with a SalesHierarchy object.
        """
        # Exposes the underlying nx.DiGraph
        self.hierarchy = hierarchy.graph
        
    def _calculate_node_historical_capacity(self, node_id: str) -> float:
        """
        Recursively calculates the historical capacity of a node by summing up 
        all '_Attainment' metrics of leaf nodes (ICs) underneath it.
        
        Supports any number of historical quarters (4, 8, 12, etc.) by dynamically
        discovering all attributes containing '_Attainment' on each IC node.
        
        For ICs with partial history (e.g., hired recently with some zero quarters),
        zero-valued quarters are imputed with the average of that IC's own non-zero
        quarters. This prevents underweighting reps who haven't been employed for
        the full lookback period.
        
        Returns 0.0 for brand-new ICs with no historical data at all — these are
        handled separately in cascade_quota() via equal-share carve-out.
        """
        # If it's a leaf node (IC), return its historical capacity
        if self.hierarchy.out_degree(node_id) == 0:
            attrs = self.hierarchy.nodes[node_id]
            # Dynamically collect ALL attainment values (supports any number of quarters)
            attainments = [v for k, v in attrs.items()
                          if '_Attainment' in k and isinstance(v, (int, float))]
            
            if not attainments:
                return 0.0
            
            non_zero = [v for v in attainments if v > 0]
            
            if not non_zero:
                return 0.0  # Brand-new hire — handled by equal-share in cascade_quota
            
            # Impute zero quarters with the IC's own non-zero average
            # e.g., IC hired 1 quarter ago: Q1=0, Q2=0, Q3=0, Q4=150K
            #   → avg_non_zero = 150K → imputed = [150K, 150K, 150K, 150K] → total = 600K
            avg_non_zero = sum(non_zero) / len(non_zero)
            imputed = [v if v > 0 else avg_non_zero for v in attainments]
            return sum(imputed)
            
        # Otherwise, aggregate the mathematical capacity of its children
        total_capacity = 0.0
        for child in self.hierarchy.successors(node_id):
            total_capacity += self._calculate_node_historical_capacity(child)
            
        return total_capacity

    def cascade_quota(self, root_node: str, macro_target: float, hedge_multiplier=1.0,
                      new_ic_overrides: Dict[str, float] = None) -> Dict[str, float]:
        """
        Distributes the macro_target from the root_node down to all descendants.
        Uses historical capacity to proportionally weight distribution at each level.
        
        Supports any number of historical quarters — the algorithm dynamically discovers
        all '_Attainment' attributes on IC nodes.
        
        hedge_multiplier: Can be a single float (e.g., 1.05 to apply a 5% buffer at EVERY level)
                          OR a dictionary of specific nodes to their explicit hedge 
                          (e.g., {'RVP_NA_1': 1.10, 'Dir_RVP_NA_1_1': 1.05}). 
                          Defaults to 1.0 (no hedge).
        
        new_ic_overrides: Optional Dict[str, float] mapping IC node IDs to fixed quota amounts.
                          Use for CRO-mandated quotas that override the algorithm.
                          e.g., {'IC_Strategic_Hire': 500000.0}
                          These ICs are excluded from proportional distribution.
                          
        New IC handling (automatic):
          - Brand-new ICs (all zeros): Receive equal share (target / num_children)
            before proportional distribution of the remainder among experienced ICs.
          - Partial-history ICs: Zero quarters are imputed with the IC's own non-zero
            average, so they compete fairly in proportional distribution.
        """
        if new_ic_overrides is None:
            new_ic_overrides = {}
        
        # Dictionary to store dynamically calculated quotas
        quotas = {root_node: macro_target}
        
        # Traverse top-down through the organization
        for node in nx.topological_sort(self.hierarchy):
            if node not in quotas:
                continue
                
            current_target = quotas[node]
            children = list(self.hierarchy.successors(node))
            
            if not children:
                continue # Reached an IC (leaf node)
                
            # Determine the specific hedge for this manager node
            if isinstance(hedge_multiplier, dict):
                current_hedge = hedge_multiplier.get(node, 1.0)
            else:
                current_hedge = hedge_multiplier
                
            # Apply the hedge/overassignment buffer for this layer of management
            target_to_distribute = current_target * current_hedge
            
            # Check if we're at the leaf level (all children are ICs)
            at_leaf_level = all(self.hierarchy.out_degree(c) == 0 for c in children)
            
            if at_leaf_level:
                # Handle CRO overrides — carve out fixed quotas first
                override_ics = [c for c in children if c in new_ic_overrides]
                override_total = sum(new_ic_overrides[c] for c in override_ics)
                
                # Identify brand-new ICs (all zeros, no override) — give them equal share
                remaining_children = [c for c in children if c not in override_ics]
                new_ics = [c for c in remaining_children
                           if self._calculate_node_historical_capacity(c) == 0.0]
                experienced_ics = [c for c in remaining_children if c not in new_ics]
                
                # Assign override quotas
                for ic in override_ics:
                    quotas[ic] = new_ic_overrides[ic]
                
                if new_ics:
                    # Equal share for brand-new ICs (from total pool, not remainder)
                    equal_share = target_to_distribute / len(children)
                    for ic in new_ics:
                        quotas[ic] = equal_share
                    
                    # Distribute remainder proportionally among experienced ICs
                    remaining = target_to_distribute - override_total - (equal_share * len(new_ics))
                    
                    exp_capacities = {c: self._calculate_node_historical_capacity(c) 
                                      for c in experienced_ics}
                    total_exp_capacity = sum(exp_capacities.values())
                    
                    for ic in experienced_ics:
                        if total_exp_capacity > 0:
                            weight = exp_capacities[ic] / total_exp_capacity
                        else:
                            weight = 1.0 / len(experienced_ics) if experienced_ics else 0
                        quotas[ic] = remaining * weight
                else:
                    # No new ICs — standard proportional distribution (minus overrides)
                    remaining = target_to_distribute - override_total
                    
                    child_capacities = {c: self._calculate_node_historical_capacity(c)
                                        for c in experienced_ics}
                    total_child_capacity = sum(child_capacities.values())
                    
                    for child in experienced_ics:
                        if total_child_capacity > 0:
                            weight = child_capacities[child] / total_child_capacity
                        else:
                            weight = 1.0 / len(experienced_ics) if experienced_ics else 0
                        quotas[child] = remaining * weight
            else:
                # Non-leaf level — standard proportional distribution
                child_capacities = {child: self._calculate_node_historical_capacity(child) 
                                    for child in children}
                total_child_capacity = sum(child_capacities.values())
                
                # Distribute proportionally based on historical capacity performance
                for child in children:
                    if total_child_capacity > 0:
                        weight = child_capacities[child] / total_child_capacity
                    else:
                        # Fallback to pure even mathematical split if no historical data exists
                        weight = 1.0 / len(children)
                        
                    quotas[child] = target_to_distribute * weight
                
        return quotas
