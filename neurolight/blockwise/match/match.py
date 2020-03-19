from neurolight.match.preprocess import (
    mouselight_preprocessing,
    add_fallback,
    add_precomputed_edges,
    assignments_from_matched,
)
from neurolight.match.costs_vectorized import wrapped_costs as get_costs

from funlib.match import GraphToTreeMatcher

import networkx as nx

import time
import logging

logger = logging.getLogger(__file__)


def match(
    consensus,
    skeletonization,
    pre_solved,
    node_offset,
    max_new_edge,
    penalty_attr,
    location_attr,
    target_attr,
    expected_edge_len,
    max_match_dist,
):
    skeletonization = skeletonization.to_undirected()
    start_time = time.time()

    if len(consensus.nodes) == 0:
        logger.info(f"No consensus nodes. Skipping")
        return ([], [], skeletonization)

    logger.info("PreProcess...")
    start_time = time.time()

    logger.info(
        f"Skeletoniation has {len(skeletonization.nodes)} nodes and "
        f"{len(skeletonization.edges)} edges before preprocessing"
    )

    mouselight_preprocessing(
        skeletonization,
        max_dist=max_new_edge,
        penalty_attr=penalty_attr,
        location_attr=location_attr,
        expected_edge_len=expected_edge_len,
    )

    logger.info(f"Preprocessing took {time.time() - start_time} seconds")
    logger.info(
        f"Skeletoniation has {len(skeletonization.nodes)} nodes and "
        f"{len(skeletonization.edges)} edges after preprocessing"
    )

    start_time = time.time()
    add_fallback(
        skeletonization,
        consensus,
        node_offset=node_offset,
        max_new_edge=max_match_dist,
        expected_edge_len=expected_edge_len,
        penalty_attr=penalty_attr,
        location_attr=location_attr,
    )

    logger.info(f"Adding fallback took {time.time() - start_time} seconds")
    logger.info(
        f"Skeletoniation has {len(skeletonization.nodes)} nodes and "
        f"{len(skeletonization.edges)} edges after adding fallback"
    )
    start_time = time.time()

    add_precomputed_edges(skeletonization, pre_solved)

    logger.info(f"Adding precomputed edges took {time.time() - start_time} seconds")

    start_time = time.time()

    node_costs, edge_costs = get_costs(
        skeletonization,
        consensus,
        location_attr,
        penalty_attr,
        max_match_dist,
        max_match_dist,
        expected_edge_len,
    )

    # edge_costs = [((a, b), (c, d), e) for a, b, c, d, e in edge_costs]

    logger.info(f"Getting costs took {time.time() - start_time} seconds")

    node_assignments, edge_assignments = assignments_from_matched(
        pre_solved, target_attr
    )
    logger.info(
        f"Input contains {len(node_assignments)} node assignments "
        + f"and {len(edge_assignments)} edge assignments"
    )
    filtered_node_assignments, filtered_edge_assignments = [], []
    missing_target_nodes = []
    for source, target in node_assignments:
        assert (
            source in skeletonization.nodes
        ), f"prematched source {source} not in skeletonization"
        if target is not None:
            # skeletonization nodes may match to a target node which is outside
            # the current context. These will/can not be enforced
            if target not in consensus.nodes:
                missing_target_nodes.append(target)
            else:
                filtered_node_assignments.append((source, target))
        else:
            filtered_node_assignments.append((source, target))
    missing_target_edges = []
    for source, target in edge_assignments:
        assert (
            source in skeletonization.edges
        ), f"prematched source {source} not in skeletonization"
        if target is not None:
            u, v = target
            # skeletonization edges may match to a target edge, one node
            # of which is outside the current context. These can/will not be enforced
            if (u - node_offset, v - node_offset) not in consensus.nodes:
                missing_target_edges.append(target)
            else:
                filtered_edge_assignments.append((source, target))
        else:
            filtered_node_assignments.append((source, target))

    logger.info(f"{len(missing_target_nodes)} node assignments could not be enforced")
    logger.info(f"{len(missing_target_edges)} edge assignments could not be enforced")

    start_time = time.time()

    matcher = GraphToTreeMatcher(
        skeletonization, consensus, node_costs, edge_costs, timeout=120
    )
    matcher.enforce_expected_assignments(filtered_node_assignments)
    matcher.enforce_expected_assignments(filtered_edge_assignments)
    logger.info(f"Initializing took {time.time() - start_time} seconds")

    try:
        start_time = time.time()
        node_matchings, edge_matchings, _ = matcher.match()

        logger.info("Solving took %s seconds" % (time.time() - start_time))
    except ValueError as e:
        logger.warning(e)
        return skeletonization

    return node_matchings, edge_matchings, skeletonization
