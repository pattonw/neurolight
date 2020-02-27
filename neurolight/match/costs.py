import numpy as np
from scipy.spatial import cKDTree
import rtree
import networkx as nx

import itertools
from typing import List, Hashable, Tuple, Optional

Node = Hashable
Edge = Tuple[Node, Node]


def get_costs(
    graph: nx.Graph,
    tree: nx.DiGraph,
    location_attr: str,
    penalty_attr: str,
    node_match_threshold: float,
    edge_match_threshold: float,
    expected_edge_len: float,
    node_balance: Optional[float] = None,
) -> Tuple[List[Tuple[Node, Node, float]], List[Tuple[Edge, Edge, float]]]:

    node_balance = (
        node_balance
        if node_balance is not None
        else get_node_balance(tree, location_attr, expected_edge_len)
    )

    graph_kd, graph_kd_ids, tree_kd, tree_kd_ids = initialize_kdtrees(
        graph, tree, location_attr
    )
    tree_rtree = initialize_rtree(tree, location_attr)

    node_matchings = []

    close_enough = graph_kd.query_ball_tree(tree_kd, node_match_threshold)
    for i, close_nodes in enumerate(close_enough):
        graph_n = graph_kd_ids[i]
        for j in close_nodes:
            tree_n = tree_kd_ids[j]
            distance = np.linalg.norm(
                np.array(graph.nodes[graph_n][location_attr])
                - np.array(tree.nodes[tree_n][location_attr])
            )
            penalty = graph.nodes[graph_n].get(penalty_attr, 0) + tree.nodes[
                tree_n
            ].get(penalty_attr, 0)
            node_matchings.append(
                (
                    graph_n,
                    tree_n,
                    node_cost(distance, node_match_threshold, penalty, node_balance),
                )
            )
        node_matchings.append(
            (
                graph_n,
                None,
                node_cost(
                    0,
                    node_match_threshold,
                    graph.nodes[graph_n].get(penalty_attr, 0),
                    node_balance,
                ),
            )
        )

    edge_matchings = []
    num_filtered = 0
    total = 0
    for graph_e in graph.edges():

        possible_tree_edges = list(
            tree_edge_query(
                np.array(graph.nodes[graph_e[0]][location_attr]),
                np.array(graph.nodes[graph_e[1]][location_attr]),
                edge_match_threshold,
                tree_rtree,
                tree,
                location_attr,
            )
        )

        num_filtered += len([x for x in possible_tree_edges if x is None])
        total += len(possible_tree_edges)
        possible_tree_edges = [x for x in possible_tree_edges if x is not None]

        for tree_e in possible_tree_edges:
            distance = edge_dist(
                np.array(graph.nodes[graph_e[0]][location_attr]),
                np.array(graph.nodes[graph_e[1]][location_attr]),
                np.array(tree.nodes[tree_e[0]][location_attr]),
                np.array(tree.nodes[tree_e[1]][location_attr]),
            )
            length = np.linalg.norm(
                np.array(graph.nodes[graph_e[0]][location_attr])
                - np.array(graph.nodes[graph_e[1]][location_attr])
            )
            penalty = graph.edges[graph_e].get(penalty_attr, 0) + tree.edges[
                tree_e
            ].get(penalty_attr, 0)
            match_cost = edge_cost(
                distance, edge_match_threshold, length, expected_edge_len, penalty
            )
            edge_matchings.append((graph_e, tree_e, match_cost))

            if not isinstance(graph, nx.DiGraph):
                graph_e_inv = (graph_e[1], graph_e[0])
                edge_matchings.append((graph_e_inv, tree_e, match_cost))

    print(f"filtered {num_filtered} of {total} edges out")

    return node_matchings, edge_matchings


def get_node_balance(tree: nx.DiGraph, pos_attr: str, expected_edge_len: float):
    num_nodes = len(list(tree.nodes))
    tree_cable_length = 0.0
    for u, v in tree.edges():
        u_loc = np.array(tree.nodes[u][pos_attr])
        v_loc = np.array(tree.nodes[v][pos_attr])

        tree_cable_length += np.linalg.norm(u_loc - v_loc)

    return (tree_cable_length // expected_edge_len) / num_nodes


def initialize_rtree(tree, location_attr):
    p = rtree.index.Property()
    p.dimension = 3
    tree_rtree = rtree.index.Index(properties=p)
    for i, (u, v) in enumerate(tree.edges()):
        u_loc = np.array(tree.nodes[u][location_attr])
        v_loc = np.array(tree.nodes[v][location_attr])
        mins = np.min(np.array([u_loc, v_loc]), axis=0)
        maxs = np.max(np.array([u_loc, v_loc]), axis=0)
        box = tuple(x for x in itertools.chain(mins.tolist(), maxs.tolist()))
        tree_rtree.insert(i, box, obj=(u, v))

    return tree_rtree


def initialize_kdtrees(graph: nx.Graph, tree: nx.DiGraph, location_attr: str):
    tree_kd_ids, tree_node_attrs = [list(x) for x in zip(*tree.nodes.items())]
    tree_kd = cKDTree([attrs[location_attr] for attrs in tree_node_attrs])

    graph_kd_ids, graph_node_attrs = [list(x) for x in zip(*graph.nodes.items())]
    graph_kd = cKDTree([attrs[location_attr] for attrs in graph_node_attrs])

    return graph_kd, graph_kd_ids, tree_kd, tree_kd_ids


def node_cost(distance: float, max_dist, penalty: float, node_balance: float) -> float:
    return node_balance * (distance / max_dist + penalty)


def tree_edge_query(
    u_loc: np.ndarray, v_loc: np.ndarray, radius: float, tree_rtree, tree, location_attr
):
    # r tree only stores bounding boxes of lines, so we have to retrieve all
    # potential edges, and then filter them based on a line specific distance
    # calculation
    rect = tuple(
        x
        for x in itertools.chain(
            (np.min(np.array([u_loc, v_loc]), axis=0) - radius).tolist(),
            (np.max(np.array([u_loc, v_loc]), axis=0) + radius).tolist(),
        )
    )
    possible_tree_edges = [
        x.object for x in tree_rtree.intersection(rect, objects=True)
    ]
    # line distances
    for x, y in possible_tree_edges:
        dist = edge_dist(
            u_loc,
            v_loc,
            np.array(tree.nodes[x][location_attr]),
            np.array(tree.nodes[y][location_attr]),
        )
        if dist < radius:
            yield (x, y)
        else:
            yield None


def edge_dist(
    u_loc: np.ndarray, v_loc: np.ndarray, x_loc: np.ndarray, y_loc: np.ndarray
) -> float:
    """
    psuedo avg distance of a line to another line.
    calculated as the average distance of the end points (u, v) to the line (x, y).
    """
    distance = (
        point_to_edge_dist(u_loc, x_loc, y_loc)
        + point_to_edge_dist(v_loc, x_loc, y_loc)
    ) / 2
    return distance


def point_to_edge_dist(
    center: np.ndarray, u_loc: np.ndarray, v_loc: np.ndarray
) -> float:
    slope = v_loc - u_loc
    edge_mag = np.linalg.norm(slope)
    if np.isclose(edge_mag, 0):
        return np.linalg.norm(u_loc - center)
    frac = np.clip(np.dot(center - u_loc, slope) / np.dot(slope, slope), 0, 1)
    min_dist = np.linalg.norm(frac * slope + u_loc - center)
    return min_dist


def edge_cost(dist, max_dist, length, exp_length, penalty) -> float:
    """
    The cost of selecting an edge in the matching.
    
    Normalize dist and length to keep the costs within an expected range.
    This allows easier balancing between edge costs and node costs.
    """
    return (length / exp_length) * (dist / max_dist) + penalty
