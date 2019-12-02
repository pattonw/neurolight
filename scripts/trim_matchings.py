from neurolight.match.costs import get_costs
from neurolight.match.preprocess import mouselight_preprocessing
from funlib.match.helper_functions import match

import networkx as nx

from obj_to_npz import parse_npy_graph
import pickle

from pathlib import Path
import copy


penalty_attr = "penalty"
location_attr = "location"

trimmed = Path.cwd() / "invalid_trimmed"
trimmed.mkdir(parents=True, exist_ok=True)

for i, valid in enumerate(Path("invalid").iterdir()):
    print(valid.name)
    graph = pickle.load((valid / "graph.obj").open("rb"))
    tree = pickle.load((valid / "tree.obj").open("rb"))

    temp = copy.deepcopy(graph)

    max_edge_len = 48
    mouselight_preprocessing(
        temp,
        max_dist=150,
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
        node_match_threshold=100,
        edge_match_threshold=100,
        node_balance=node_balance,
    )

    try:
        matched = match(temp, tree, node_costs, edge_costs)
    except:
        print("failed to match!")
        matched = graph

    untouched_ccs = list(set(x) for x in nx.connected_components(graph))

    for node in matched.nodes():
        for cc in untouched_ccs:
            if node in cc:
                untouched_ccs.remove(cc)

    (trimmed / f"{i:03}").mkdir(exist_ok=True, parents=True)
    for untouched_cc in untouched_ccs:
        graph.remove_nodes_from(untouched_cc)
    pickle.dump(graph, open(f"invalid_trimmed/{i:03}/graph.obj", "wb"))
    pickle.dump(tree, open(f"invalid_trimmed/{i:03}/tree.obj", "wb"))
