from neurolight.visualizations.neuroglancer_trees import visualize_trees
from neurolight.match.costs import get_costs
from neurolight.match.preprocess import mouselight_preprocessing

from funlib.match.helper_functions import match
import networkx as nx

import sys
from pathlib import Path
import copy
import pickle

filename = Path(sys.argv[1])

penalty_attr = "penalty"
location_attr = "location"

graph = pickle.load((filename / "graph.obj").open("rb"))
tree = pickle.load((filename / "tree.obj").open("rb"))

temp = copy.deepcopy(graph)

max_edge_len = 48

mouselight_preprocessing(
    temp,
    max_dist=max_edge_len,
    voxel_size=[10, 3, 3],
    penalty_attr=penalty_attr,
    location_attr=location_attr,
)

node_match_threshold = 76
edge_match_threshold = 76
node_balance = 10

node_costs, edge_costs = get_costs(
    temp,
    tree,
    location_attr=location_attr,
    penalty_attr=penalty_attr,
    node_match_threshold=node_match_threshold,
    edge_match_threshold=node_match_threshold,
    node_balance=node_balance,
)

try:
    matched = match(temp, tree, node_costs, edge_costs)
except RuntimeError:
    matched = nx.DiGraph()
except ValueError:
    matched = nx.DiGraph()

visualize_trees(
    {"graph": graph, "tree": tree, "preprocessed": temp, "matched": matched}
)
