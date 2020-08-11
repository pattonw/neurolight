import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from collections import Counter
import itertools
from typing import Dict, List
import logging


def assignments_from_matched(pre_solved: nx.DiGraph, target_attr: str):
    """
    Get the target onto which each node in presolved matched
    """
    node_assignments = []
    edge_assignments = []
    for node, attrs in pre_solved.nodes.items():
        consensus_target = attrs[target_attr]
        node_assignments.append((node, consensus_target))
    for edge, attrs in pre_solved.edges.items():
        consensus_target = attrs[target_attr]
        if consensus_target is not None:
            consensus_target = tuple(consensus_target)
        edge_assignments.append((edge, consensus_target))
    return node_assignments, edge_assignments


def add_precomputed_edges(skeletonization, pre_solved):
    """
    add edge (u, v) to skeletonization if (u, v) in pre_solved and
    both u, and v are in skeletonization.
    """
    for u, v in pre_solved.edges():
        if (
            u in skeletonization.nodes()
            and v in skeletonization.nodes()
            and (u, v) not in skeletonization.edges()
        ):
            skeletonization.add_edge(u, v)


def mouselight_preprocessing(
    graph: nx.DiGraph(),
    max_dist: float,
    expected_edge_len: float,
    penalty_attr: str = "penalty",
    location_attr: str = "location",
):
    """
    Preprocessing for mouselight skeletonizations s.t. they can be matched
    to mouselight consensus neurons.

    Common problems seen in skeletonizations are:
    
    Gaps:
    A------------------------------B
    a---b---c---d--e      f--g--h--i

    Crossovers:
            A
            |                       a
            |                       |
    B-------C-------D   vs      b---c-d----e
            |                         |
            |                         f
            E

    Solution:
    Remove nodes of degree > 1, and then for all pairwise connected components,
    create edges between all nodes in component a to all nodes in component b
    if the edge is below `min_dist`.
    """

    temp = graph.to_undirected()

    temp.remove_nodes_from(filter(lambda x: temp.degree(x) > 2, list(temp.nodes)))

    wccs = list(list(x) for x in nx.connected_components(temp))
    if len(wccs) < 2:
        return

    spatial_wccs = [
        cKDTree([np.array(temp.nodes[x][location_attr]) for x in wcc]) for wcc in wccs
    ]
    for (ind_a, spatial_a), (ind_b, spatial_b) in itertools.combinations(
        enumerate(spatial_wccs), 2
    ):
        for node_a_index, closest_nodes in itertools.chain(
            *zip(enumerate(spatial_a.query_ball_tree(spatial_b, max_dist)))
        ):
            for node_b_index in closest_nodes:
                node_a = wccs[ind_a][node_a_index]
                node_b = wccs[ind_b][node_b_index]
                edge_len = np.linalg.norm(
                    np.array(temp.nodes[node_a][location_attr])
                    - np.array(temp.nodes[node_b][location_attr])
                )
                if edge_len < max_dist:
                    # Add extra penalties to added edges to minimize cable length
                    # assigned to ambiguous ground truth.
                    graph.add_edge(
                        node_a,
                        node_b,
                        **{
                            penalty_attr: preprocessed_edge_penalty(
                                edge_len, expected_edge_len
                            )
                        }
                    )


def add_fallback(
    graph: nx.DiGraph(),
    fallback: nx.DiGraph(),
    node_offset: int,
    max_new_edge: float,
    expected_edge_len: float,
    penalty_attr: str = "penalty",
    location_attr: str = "location",
):
    """
    In the case you are matching a graph G to a tree T, it
    may be the case that G does not contain a subgraph isomorphic
    to G. If you want to prevent failure, you can augment G with T,
    so that there is always a solution, just matching T to T.
    However with sufficient penalties assigned to this matching,
    we can make matching T to T, a last resort, that will only be
    used if matching G to T is impossible.

    T's node id's will be shifted up by node_offset to avoid id conflicts
    """

    for node, node_attrs in fallback.nodes.items():
        u = int(node + node_offset)
        graph.add_node(u, **node_attrs)
        graph.nodes[u][penalty_attr] = fallback_node_penalty()

    for u, v in fallback.edges:
        u = int(u + node_offset)
        v = int(v + node_offset)
        length = np.linalg.norm(
            np.array(graph.nodes[u][location_attr])
            - np.array(graph.nodes[v][location_attr])
        )

        graph.add_edge(
            u, v, **{penalty_attr: fallback_edge_penalty(length, expected_edge_len)}
        )

    wccs = list(list(x) for x in nx.connected_components(graph))
    if len(wccs) < 2:
        return

    spatial_wccs = [
        cKDTree([np.array(graph.nodes[x][location_attr]) for x in wcc]) for wcc in wccs
    ]
    for (ind_a, spatial_a), (ind_b, spatial_b) in itertools.combinations(
        enumerate(spatial_wccs), 2
    ):
        for node_a_index, closest_nodes in itertools.chain(
            *zip(enumerate(spatial_a.query_ball_tree(spatial_b, max_new_edge)))
        ):
            node_a = wccs[ind_a][node_a_index]
            closest_node = None
            shortest_edge = None
            for node_b_index in closest_nodes:
                node_b = wccs[ind_b][node_b_index]
                edge_len = np.linalg.norm(
                    np.array(graph.nodes[node_a][location_attr])
                    - np.array(graph.nodes[node_b][location_attr])
                )
                if edge_len < max_new_edge:
                    # Add extra penalties to added edges to minimize cable length
                    # assigned to ambiguous ground truth.
                    if closest_node is None:
                        closest_node = node_b
                        shortest_edge = edge_len
                    elif edge_len < shortest_edge:
                        closest_node = node_b
                        shortest_edge = edge_len
            if shortest_edge is not None:
                graph.add_edge(
                    node_a,
                    closest_node,
                    **{
                        penalty_attr: preprocessed_edge_penalty(
                            shortest_edge, expected_edge_len
                        )
                    }
                )

    return graph


def preprocessed_edge_penalty(edge_len, expected_edge_len):
    """
    edge cost function: len / exp_len * dist + penalty
    distance for a preprocessed edge is likely to be 0

    Square the len penalty since we really don't want long edges.
    dist in cost is normalized to 1, so here we can multiply by
    2 to guarantee this edge has much higher cost.
    """
    return (edge_len / expected_edge_len) ** 2 * 2


def fallback_node_penalty():
    """
    node cost function: node_balance * (dist + penalty)
    distance will be 0 for a fallback node.
    
    Using a fall back node should be an order of magnitude worse
    than using a regular node
    """
    return 10


def fallback_edge_penalty(edge_len, expected_edge_len):
    """
    edge cost function: len / exp_len * dist + penalty
    distance for a fallback edge will be 0

    Cube the len penalty penalize fallback edges more than
    preprocessed edges.
    multiply by 10 to make the fallback edges an order
    of magnitude worse than a regular edge
    """
    return (edge_len / expected_edge_len) ** 3 * 10
