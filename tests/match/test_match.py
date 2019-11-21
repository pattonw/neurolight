import networkx as nx
import numpy as np

from neurolight.match.costs import get_costs


def test_get_machings():
    """
    A-----B-----C
    a-b-c-d-e-f-g
    """
    consensus_nodes = [
        ("A", {"location": np.array([1, 1, 1])}),
        ("B", {"location": np.array([1, 1, 4])}),
        ("C", {"location": np.array([1, 1, 7])}),
    ]
    consensus_edges = [("A", "B"), ("B", "C")]

    skeleton_nodes = [
        ("a", {"location": np.array([1, 1, 1])}),
        ("b", {"location": np.array([1, 1, 2])}),
        ("c", {"location": np.array([1, 1, 3])}),
        ("d", {"location": np.array([1, 1, 4])}),
        ("e", {"location": np.array([1, 1, 5])}),
        ("f", {"location": np.array([1, 1, 6])}),
        ("g", {"location": np.array([1, 1, 7])}),
    ]
    skeleton_edges = [
        ("a", "b"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("e", "f"),
        ("f", "g"),
    ]

    consensus = nx.DiGraph()
    consensus.add_nodes_from(consensus_nodes)
    consensus.add_edges_from(consensus_edges)

    skeleton = nx.Graph()
    skeleton.add_nodes_from(skeleton_nodes)
    skeleton.add_edges_from(skeleton_edges)

    node_match_threshold = 2
    edge_match_threshold = 2
    node_balance = 2

    node_matchings, edge_matchings = get_costs(
        skeleton,
        consensus,
        "location",
        "penalty",
        node_match_threshold,
        edge_match_threshold,
        node_balance,
    )

    expected_node_distances = {
        "a": {"A": 0},
        "b": {"A": 1, "B": 2},
        "c": {"A": 2, "B": 1},
        "d": {"B": 0},
        "e": {"B": 1, "C": 2},
        "f": {"B": 2, "C": 1},
        "g": {"C": 0},
    }

    expected_edge_distances = {
        ("a", "b"): {("A", "B"): 0},
        ("b", "c"): {("A", "B"): 0, ("B", "C"): 3 / 2},
        ("c", "d"): {("A", "B"): 0, ("B", "C"): 1 / 2},
        ("d", "e"): {("A", "B"): 1 / 2, ("B", "C"): 0},
        ("e", "f"): {("A", "B"): 3 / 2, ("B", "C"): 0},
        ("f", "g"): {("B", "C"): 0},
    }

    for skeleton_node, tree_node, cost in node_matchings:
        assert cost == expected_node_distances[skeleton_node][tree_node] * node_balance

    for skeleton_edge, tree_edge, cost in edge_matchings:
        assert cost == expected_edge_distances[skeleton_edge][tree_edge]

