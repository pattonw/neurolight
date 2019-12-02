import numpy as np
from scipy.spatial import cKDTree
import rtree
import networkx as nx

import itertools
from typing import List, Hashable, Tuple

Node = Hashable
Edge = Tuple[Node, Node]


def get_costs(
    graph: nx.Graph,
    tree: nx.DiGraph,
    location_attr: str,
    penalty_attr: str,
    node_match_threshold: float,
    edge_match_threshold: float,
    node_balance: float,
) -> Tuple[List[Tuple[Node, Node, float]], List[Tuple[Edge, Edge, float]]]:
    graph_kd, graph_kd_ids, tree_kd, tree_kd_ids = initialize_kdtrees(
        graph, tree, location_attr
    )
    tree_rtree = initialize_rtree(tree, location_attr)

    node_matchings = []

    close_enough = graph_kd.query_ball_tree(tree_kd, node_match_threshold)
    for i, close_nodes in enumerate(close_enough):
        for j in close_nodes:
            graph_n = graph_kd_ids[i]
            tree_n = tree_kd_ids[j]
            distance = np.linalg.norm(
                graph.nodes[graph_n][location_attr] - tree.nodes[tree_n][location_attr]
            )
            penalty = graph.nodes[graph_n].get(penalty_attr, 0) + tree.nodes[
                tree_n
            ].get(penalty_attr, 0)
            node_matchings.append(
                (graph_n, tree_n, node_cost(distance, penalty, node_balance))
            )

    edge_matchings = []
    for graph_e in graph.edges():

        possible_tree_edges = tree_edge_query(
            graph.nodes[graph_e[0]][location_attr],
            graph.nodes[graph_e[1]][location_attr],
            edge_match_threshold,
            tree_rtree,
            tree,
            location_attr,
        )

        for tree_e in possible_tree_edges:
            distance = edge_dist(
                graph.nodes[graph_e[0]][location_attr],
                graph.nodes[graph_e[1]][location_attr],
                tree.nodes[tree_e[0]][location_attr],
                tree.nodes[tree_e[1]][location_attr],
            )
            penalty = graph.edges[graph_e].get(penalty_attr, 0) + tree.edges[
                tree_e
            ].get(penalty_attr, 0)
            match_cost = edge_cost(distance, penalty)
            edge_matchings.append((graph_e, tree_e, match_cost))

            if not isinstance(graph, nx.DiGraph):
                graph_e_inv = (graph_e[1], graph_e[0])
                edge_matchings.append((graph_e_inv, tree_e, match_cost))

    return node_matchings, edge_matchings


def initialize_rtree(tree, location_attr):
    p = rtree.index.Property()
    p.dimension = 3
    tree_rtree = rtree.index.Index(properties=p)
    for i, (u, v) in enumerate(tree.edges()):
        u_loc = tree.nodes[u][location_attr]
        v_loc = tree.nodes[v][location_attr]
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


def node_cost(distance: float, penalty: float, node_balance: float) -> float:
    return node_balance * (distance + penalty)


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
            (np.min(np.array([u_loc, v_loc]), axis=0) + radius).tolist(),
        )
    )
    possible_tree_edges = [
        x.object for x in tree_rtree.intersection(rect, objects=True)
    ]
    # line distances
    for x, y in possible_tree_edges:
        dist = edge_dist(
            u_loc, v_loc, tree.nodes[x][location_attr], tree.nodes[y][location_attr]
        )
        if dist < radius:
            yield (x, y)


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


def edge_cost(distance, penalty) -> float:
    """
    Distance is normalized by line length
        This means it is probably much smaller for large lines than small lines
        Especially since added lines are often aligned with the consensus
    Penalty is the line length normalized by min(voxel size) to get
        an overestimate of how many skeletonized edges it would replace

    formula:
        distance + (distance + penalty) * (penalty)

    Motivation: 
        distance: base cost for all edges with or without penalty
        (penalty + distance) * penalty:  
    """
    return distance * (penalty ** 3 + 1)
