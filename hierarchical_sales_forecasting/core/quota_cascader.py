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
        the Q1-Q4 attainment of all leaf nodes (ICs) underneath it.
        """
        # If it's a leaf node (IC), return its direct historical sum of the past 4 quarters
        if self.hierarchy.out_degree(node_id) == 0:
            attrs = self.hierarchy.nodes[node_id]
            return sum([attrs.get(f'Q{i}_Attainment', 0.0) for i in range(1, 5)])
            
        # Otherwise, aggregate the mathematical capacity of its children
        total_capacity = 0.0
        for child in self.hierarchy.successors(node_id):
            total_capacity += self._calculate_node_historical_capacity(child)
            
        return total_capacity

    def cascade_quota(self, root_node: str, macro_target: float, hedge_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Distributes the macro_target from the root_node down to all descendants.
        Uses the historical 4-quarter capacity linearly to strictly weight distribution.
        
        hedge_multiplier: The percentage over-assignment applied at EVERY step down the tree to create commit safety.
                          e.g., 1.05 means managers assign 105% of their own target downwards to their direct reports.
        """
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
                
            # Apply the hedge/overassignment buffer for this layer of management
            target_to_distribute = current_target * hedge_multiplier
            
            # Calculate historical 4-quarter capacities for all direct reports
            child_capacities = {child: self._calculate_node_historical_capacity(child) for child in children}
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
