import numpy as np
from scipy.spatial import cKDTree
import rtree
import networkx as nx

import itertools
from typing import List, Hashable, Tuple
import logging

Node = Hashable
Edge = Tuple[Node, Node]

logger = logging.getLogger(__file__)


def get_costs(
    graph: nx.Graph,
    tree: nx.DiGraph,
    location_attr: str,
    penalty_attr: str,
    node_match_threshold: float,
    edge_match_threshold: float,
    node_balance: float,
) -> Tuple[List[Tuple[Node, Node, float]], List[Tuple[Edge, Edge, float]]]:

    graph_nodes = np.array(graph.nodes)
    tree_nodes = list(tree.nodes)

    # map from node to index. Necessary to vectorize edge operations
    graph_index_map = {u: i for i, u in enumerate(graph_nodes)}
    tree_index_map = {u: i for i, u in enumerate(tree_nodes + [None])}

    # loop over graph nodes to extract locations
    graph_locations = np.array([graph.nodes[x][location_attr] for x in graph_nodes])

    # loop over graph nodes to extract penalties
    graph_penalties = np.array(
        [graph.nodes[x].get(penalty_attr, 0) for x in graph_nodes]
    )

    # loop over tree nodes to extract locations
    # "None" node must have a placeholder location
    tree_locations = np.array(
        [tree.nodes[x][location_attr] for x in tree_nodes] + [(0, 0, 0)]
    )

    # loop over tree nodes to extract penalties
    # "None" node must have a placeholder penalty
    tree_penalties = np.array(
        [tree.nodes[x].get(penalty_attr, 0) for x in tree_nodes] + [0]
    )

    # create array of tree_nodes for np.find calls
    tree_nodes = np.array(tree_nodes + [None])

    # initialize kdtrees
    graph_kd = cKDTree(graph_locations)
    tree_kd = cKDTree(tree_locations)

    # get (u, v) index pairs from tree_kd and graph_kd
    # u in tree and v in graph
    close_enough = tree_kd.query_ball_tree(graph_kd, node_match_threshold)
    # loop over query_ball_tree result
    index_pairs = np.array(
        [(i, x) for i, y in enumerate(close_enough) for x in y]
        + [(-1, i) for i in range(len(graph_nodes))]  # -1 is index of None tree node
    )

    pairs_t = np.take(tree_nodes, index_pairs[:, 0])
    pairs_g = np.take(graph_nodes, index_pairs[:, 1])
    pairs_t_locs = np.take(tree_locations, index_pairs[:, 0], axis=0)
    pairs_g_locs = np.take(graph_locations, index_pairs[:, 1], axis=0)
    pairs_t_pens = np.take(tree_penalties, index_pairs[:, 0], axis=0)
    pairs_g_pens = np.take(graph_penalties, index_pairs[:, 1], axis=0)

    pairs_ds = np.linalg.norm(pairs_g_locs - pairs_t_locs, axis=1) * np.not_equal(
        pairs_t, None
    )
    node_costs = node_cost(pairs_ds, pairs_t_pens + pairs_g_pens, node_balance)
    node_matchings = np.stack([pairs_g, pairs_t, node_costs], axis=1)

    # Edge matchings

    # loop over edges to get array of u, v. map u, v to index of u, v in above vectors
    tree_edges = np.array(
        [(tree_index_map[u], tree_index_map[v]) for u, v in tree.edges], dtype=int
    )
    graph_edges = np.array(
        [(graph_index_map[u], graph_index_map[v]) for u, v in graph.edges], dtype=int
    )

    # might be faster if we flip tree_e and graph_e
    tree_rtree = initialize_rtree(tree_edges, tree_locations)

    possible_edge_matchings = query_rtree(
        tree_rtree, graph_edges, graph_locations, edge_match_threshold
    )
    tree_e_indices = np.take(tree_edges, possible_edge_matchings[:, 1], axis=0)
    t_e_locs = np.take(tree_locations, tree_e_indices, axis=0)
    graph_e_indices = np.take(graph_edges, possible_edge_matchings[:, 0], axis=0)
    g_e_locs = np.take(graph_locations, graph_e_indices, axis=0)

    distances = edge_dist(g_e_locs, t_e_locs)

    print(
        f"V2 Filtering out {sum(distances >= edge_match_threshold)} of {len(distances)} edges"
    )

    filtered_matchings = possible_edge_matchings[distances < edge_match_threshold]

    filtered_tree_es = np.take(tree_edges, filtered_matchings[:, 1], axis=0)
    tree_us = np.take(tree_nodes, filtered_tree_es[:, 0])
    tree_vs = np.take(tree_nodes, filtered_tree_es[:, 1])
    tree_as = np.concatenate([tree_us, tree_us])
    tree_bs = np.concatenate([tree_vs, tree_vs])
    filtered_graph_es = np.take(graph_edges, filtered_matchings[:, 0], axis=0)
    graph_us = np.take(graph_nodes, filtered_graph_es[:, 0])
    graph_vs = np.take(graph_nodes, filtered_graph_es[:, 1])
    graph_as = np.concatenate([graph_us, graph_vs])
    graph_bs = np.concatenate([graph_vs, graph_us])
    filtered_distances = distances[distances < edge_match_threshold]
    ds = np.concatenate([filtered_distances, filtered_distances])

    edge_matchings = np.stack([graph_as, graph_bs, tree_as, tree_bs, ds], axis=1)

    """
    for graph_a, graph_b, tree_a, tree_b, d in edge_matchings:
        print(f"ga: {graph_a}, gb: {graph_b}, ta: {tree_a}, tb: {tree_b}")
        ga_loc = np.take(graph_locations, graph_index_map[graph_a], axis=0)
        gb_loc = np.take(graph_locations, graph_index_map[graph_b], axis=0)
        ta_loc = np.take(tree_locations, tree_index_map[tree_a], axis=0)
        tb_loc = np.take(tree_locations, tree_index_map[tree_b], axis=0)

        print(f"ga_loc: {ga_loc}, gb_loc: {gb_loc}, ta_loc: {ta_loc}, tb_loc: {tb_loc}")

        d_v1 = edge_dist_v1(np.stack([ga_loc, gb_loc]), np.stack([ta_loc, tb_loc]))
        d_v2 = edge_dist(
            np.stack([ga_loc, gb_loc]).reshape([1, 2, 3]),
            np.stack([ta_loc, tb_loc]).reshape([1, 2, 3]),
        )
        assert np.isclose(d_v2, d_v1) and np.isclose(
            d_v2, d
        ), f"{d_v2} vs {d_v1} vs {d}"
    """

    return node_matchings, edge_matchings


def initialize_rtree(edges, locs):
    p = rtree.index.Property()
    p.dimension = 3
    tree_rtree = rtree.index.Index(properties=p)
    for i, (u, v) in enumerate(edges):
        u_loc = locs[u]
        v_loc = locs[v]
        mins = np.min(np.array([u_loc, v_loc]), axis=0)
        maxs = np.max(np.array([u_loc, v_loc]), axis=0)
        box = tuple(x for x in itertools.chain(mins.tolist(), maxs.tolist()))
        tree_rtree.insert(i, box, obj=i)

    return tree_rtree


def query_rtree(rtree, edges, locs, radius):

    rects = []
    for u, v in edges:
        s = np.stack([locs[u, :], locs[v, :]])
        lower = np.min(s, axis=0) - radius
        upper = np.max(s, axis=0) + radius
        rects.append(tuple(np.concatenate([lower, upper])))
    possible_tree_edges = [
        [x.object for x in rtree.intersection(rect, objects=True)] for rect in rects
    ]
    # line distances
    possible_matchings = np.array(
        [(i, j) for i, js in enumerate(possible_tree_edges) for j in js]
    )
    return possible_matchings


def initialize_kdtrees(graph: nx.Graph, tree: nx.DiGraph, location_attr: str):
    tree_kd_ids, tree_node_attrs = [list(x) for x in zip(*tree.nodes.items())]
    tree_kd = cKDTree([attrs[location_attr] for attrs in tree_node_attrs])

    graph_kd_ids, graph_node_attrs = [list(x) for x in zip(*graph.nodes.items())]
    graph_kd = cKDTree([attrs[location_attr] for attrs in graph_node_attrs])

    return graph_kd, graph_kd_ids, tree_kd, tree_kd_ids


def node_cost(distance: float, penalty: float, node_balance: float) -> float:
    return node_balance * (distance ** 2 + penalty)


def edge_dist(a_locs: np.ndarray, b_locs: np.ndarray) -> float:
    """
    psuedo avg distance of a line to another line.
    calculated as the average distance of the end points of a to the line b.
    """
    distance = (
        point_to_edge_dist(a_locs[:, 0], b_locs[:, 0], b_locs[:, 1])
        + point_to_edge_dist(a_locs[:, 1], b_locs[:, 0], b_locs[:, 1])
    ) / 2
    return distance


def point_to_edge_dist(
    centers: np.ndarray, u_locs: np.ndarray, v_locs: np.ndarray
) -> float:
    print(f"v2: center: {centers[0, :]}, u_loc: {u_locs[0, :]}, v_loc: {v_locs[0, :]}")
    slope = v_locs - u_locs
    edge_mag = np.linalg.norm(slope, axis=1)
    zero_mag = np.isclose(edge_mag, 0)
    frac = np.clip(
        np.sum((centers - u_locs) * slope, axis=1) / np.sum(slope * slope, axis=1), 0, 1
    )
    frac = np.where(zero_mag, 0, frac)
    min_dist = np.linalg.norm((frac * slope.T).T + u_locs - centers, axis=1)
    return min_dist


def edge_dist_v1(a_locs: np.ndarray, b_locs: np.ndarray) -> float:
    """
    psuedo avg distance of a line to another line.
    calculated as the average distance of the end points of a to the line b.
    """
    distance = (
        point_to_edge_dist_v1(a_locs[0, :], b_locs[0, :], b_locs[1, :])
        + point_to_edge_dist_v1(a_locs[1, :], b_locs[0, :], b_locs[1, :])
    ) / 2
    return distance


def point_to_edge_dist_v1(
    center: np.ndarray, u_loc: np.ndarray, v_loc: np.ndarray
) -> float:
    print(f"v1: center: {center}, u_loc: {u_loc}, v_loc: {v_loc}")
    slope = v_loc - u_loc
    edge_mag = np.linalg.norm(slope)
    if np.isclose(edge_mag, 0):
        return np.linalg.norm(u_loc - center)
    frac = np.clip(np.dot(center - u_loc, slope) / np.dot(slope, slope), 0, 1)
    min_dist = np.linalg.norm(frac * slope + u_loc - center)
    return min_dist


def edge_cost(distance, length, penalty) -> float:
    return length * (distance + penalty)
