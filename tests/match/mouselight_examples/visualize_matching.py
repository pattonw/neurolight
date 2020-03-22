from neurolight.visualizations.neuroglancer_trees import visualize_trees
from neurolight.match import mouselight_preprocessing, get_costs
import networkx as nx

import sys
from pathlib import Path
import copy

from funlib.match import match

if __name__ == "__main__":

    # filename = Path("mouselight_matchings", "valid", "000")
    filename = Path(sys.argv[1])

    graph = nx.read_gpickle(filename / "graph.obj")
    tree = nx.read_gpickle(filename / "tree.obj")

    temp = copy.deepcopy(graph)
    mouselight_preprocessing(temp, max_dist=48, expected_edge_len=10)
    node_costs, edge_costs = get_costs(
        temp,
        tree,
        expected_edge_len=10,
        node_match_threshold=76,
        edge_match_threshold=76,
        location_attr="location",
        penalty_attr="penalty",
    )

    matched = match(temp, tree, node_costs, edge_costs, use_gurobi=False)

    visualize_trees(
        {
            "graph": graph,
            "tree": tree,
            # "preprocessed": temp,
            "matched": matched,
        }
    )
