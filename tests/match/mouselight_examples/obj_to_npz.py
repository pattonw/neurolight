import numpy as np
import networkx as nx

from pathlib import Path

consensus_pickles = Path(".")


def parse_npy_graph(filename, location_attr: str = "location", directed: bool = True):
    a = np.load(filename)
    edge_list = a["edge_list"]
    node_ids = a["node_ids"]
    locations = a["locations"]
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()
    graph.add_edges_from(edge_list)
    graph.add_nodes_from(
        [(nid, {location_attr: l}) for nid, l in zip(node_ids, locations)]
    )
    return graph


def graph_to_npz(graph: nx.Graph, name: str):

    g_edge_list = np.array(graph.edges, dtype=int)
    g_node_ids = np.array(graph.nodes, dtype=int)
    g_locations = np.array([graph.nodes[nid]["location"] for nid in g_node_ids])

    np.savez(name, edge_list=g_edge_list, node_ids=g_node_ids, locations=g_locations)

