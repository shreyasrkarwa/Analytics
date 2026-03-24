import networkx as nx
import pandas as pd
from typing import List, Dict, Any

class SalesHierarchy:
    """
    Models a B2B Enterprise Org Chart as a Directed Acyclic Graph (DAG).
    Allows for flexible node depths (e.g., Global -> Region -> L2 Manager -> L1 Manager -> IC).
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        
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
        
    def from_dataframe(self, df: pd.DataFrame, path_cols: List[str], metrics_cols: List[str] = None):
        """
        Builds the hierarchy flexibly from a flattened organizational DataFrame.
        `path_cols` should outline the hierarchy from root to IC, e.g., 
        ['Global', 'Region', 'Second Level Manager', 'First Level Manager', 'IC'].
        Because the algorithm loops sequentially, it naturally supports 3 nodes or 10 nodes deep.
        """
        for _, row in df.iterrows():
            # Dynamically add nodes and edges down the structural path
            for i in range(len(path_cols) - 1):
                parent = row[path_cols[i]]
                child = row[path_cols[i+1]]
                
                # Check for NaNs handles jagged hierarchies (e.g., an IC reporting directly to a VP)
                if pd.notna(parent) and pd.notna(child):
                    self.add_node(str(parent))
                    
                    # If this is the deepest defined node for this row, attach its historical metrics
                    if i == len(path_cols) - 2 and metrics_cols:
                        metrics = {col: row[col] for col in metrics_cols if pd.notna(row[col])}
                        self.add_node(str(child), attributes=metrics)
                    else:
                        self.add_node(str(child))
                    
                    self.add_edge(str(parent), str(child))

    def get_children(self, node_id: str) -> List[str]:
        """Returns direct reports of a given node."""
        return list(self.graph.successors(node_id))
        
    def get_leaves(self, node_id: str) -> List[str]:
        """Returns all ICs (leaf nodes) reporting up under this node."""
        return [n for n in nx.descendants(self.graph, node_id) if self.graph.out_degree(n) == 0]
