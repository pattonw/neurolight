import numpy as np
import networkx as nx


def parse_npy_graph(filename):
    a = np.load(filename)
    edge_list = a["edge_list"]
    node_ids = a["node_ids"]
    locations = a["locations"]
    point_type = a.get("point_type")
    radius = a.get("radius")
    neuron_part = a.get("neuron_part")
    human_placed = a.get("human_placed")

    keys, data = [
        list(x)
        for x in zip(
            *[
                (key, array)
                for key, array in [
                    ("node_ids", node_ids),
                    ("locations", locations),
                    ("point_type", point_type),
                    ("radius", radius),
                    ("neuron_part", neuron_part),
                    ("human_placed", human_placed),
                ]
                if array is not None
            ]
        )
    ]

    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    graph.add_nodes_from(
        [(row[0], {k: v for k, v in zip(keys, row[1:])}) for row in zip(*data)]
    )
    return graph
