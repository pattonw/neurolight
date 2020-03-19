import daisy
from daisy.persistence import MongoDbGraphProvider
import networkx as nx
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from scipy.spatial import cKDTree

import json
import logging
import numpy as np
import os
import sys
import time
import pickle

from daisy_check_functions import check_function, write_done
import configparser
from config_parser import (
    read_solve_config,
    read_data_config,
    read_worker_config,
    read_graph_config,
)
from funlib.match import GraphToTreeMatcher, build_matched
from neurolight.match import get_costs, mouselight_preprocessing, add_fallback

logger = logging.getLogger(__name__)


def solve_in_block(
    db_host,
    skeletonization_db,
    subsampled_skeletonization_db,
    time_limit,
    solve_number,
    graph_number,
    location_attr,
    u_name,
    v_name,
    **kwargs,
):
    logger.info("Solve in block")

    subsampled_provider = MongoDbGraphProvider(
        subsampled_skeletonization_db, db_host, mode="r+", directed=True
    )

    skeletonization_provider = MongoDbGraphProvider(
        skeletonization_db, db_host, mode="r+"
    )

    client = daisy.Client()

    while True:
        logger.info("Acquire block")
        block = client.acquire_block()

        if not block:
            return 0

        logger.debug("Solving in block %s", block)

        if check_function(
            block,
            "solve_s{}".format(solve_number),
            subsampled_skeletonization_db,
            db_host,
        ):
            client.release_block(block, 0)
            continue

        start_time = time.time()
        skeletonization = skeletonization_provider.get_graph(
            block.read_roi, node_inclusion="dangling", edge_inclusion="both"
        )
        # anything in matched was solved previously and must be maintained.
        pre_solved = subsampled_provider.get_graph(
            block.read_roi, node_inclusion="dangling", edge_inclusion="both"
        )

        # if len(skeletonization.nodes()) > 10_000:
        #     to_remove = set(skeletonization.nodes()) - set(pre_solved.nodes())
        #     skeletonization.remove_nodes_from(to_remove)
        #     logger.info(f"Solving for {len(skeletonization.nodes())} would take too long")
        #     logger.info(f"Ignoring {len(to_remove)} nodes and skipping this block!")

        logger.info(
            f"Reading skeletonization with {len(skeletonization.nodes)} nodes and "
            + f"{len(skeletonization.edges)} edges took {time.time() - start_time} seconds"
        )

        if len(skeletonization.nodes) == 0:
            logger.info(f"No consensus nodes in roi {block.read_roi}. Skipping")
            write_done(
                block, f"solve_s{solve_number}", subsampled_skeletonization_db, db_host
            )
            client.release_block(block, 0)
            continue

        logger.info("PreProcess...")
        start_time = time.time()

        logger.info(
            f"Skeletoniation has {len(skeletonization.nodes)} nodes "
            f"and {len(skeletonization.edges)} edges before subsampling"
        )

        num_removed = remove_line_nodes(skeletonization, location_attr)
        logger.info(f"Removed {num_removed} nodes from skeletonization!")

        num_nodes, num_edges = write_matched(
            db_host,
            subsampled_skeletonization_db,
            block,
            skeletonization,
            pre_solved,
            location_attr,
            u_name,
            v_name,
        )

        logger.info(
            f"Writing matched graph with {num_nodes} nodes and {num_edges} edges "
            f"took {time.time()-start_time} seconds"
        )

        logger.info("Write done")
        write_done(
            block,
            "solve_s{}".format(solve_number),
            subsampled_skeletonization_db,
            db_host,
        )

        logger.info("Release block")
        client.release_block(block, 0)

    return 0


def remove_line_nodes(skeletonization, pos_attr):
    to_remove = []
    for node in skeletonization:
        if skeletonization.degree(node) == 2:
            loc = np.array(skeletonization.nodes[node][pos_attr])
            neighbor_locations = [
                np.array(skeletonization.nodes[n][pos_attr])
                for n in skeletonization.neighbors(node)
            ]
            slopes = [loc - n_loc for n_loc in neighbor_locations]
            slopes = [
                sign * (s / np.linalg.norm(s)) for sign, s in zip([1, -1], slopes)
            ]
            if all(np.isclose(*slopes)):
                to_remove.append(node)
    for node in to_remove:
        u, v = list(skeletonization.neighbors(node))
        skeletonization.add_edge(u, v)
        skeletonization.remove_node(node)


def write_matched(
    host,
    db_name,
    roi,
    skeletonization,
    pre_solved,
    location_attr,
    u_name="u",
    v_name="v",
    nodes_collection_name="nodes",
    edges_collection_name="edges",
):

    client = MongoClient(host)
    db = client[db_name]
    nodes = db[nodes_collection_name]
    edges = db[edges_collection_name]

    edge_doc_list = []
    all_matched_nodes = set([])

    for skel_e in skeletonization.edges:
        u, v = skel_e
        if (
            roi.write_roi.contains(skeletonization.nodes[u][location_attr])
            or (u in pre_solved.nodes and v not in pre_solved.nodes)
        ) and (
            roi.write_roi.contains(skeletonization.nodes[v][location_attr])
            or (v in pre_solved.nodes and u not in pre_solved.nodes)
        ):
            all_matched_nodes.add(u)
            all_matched_nodes.add(v)
            doc = {u_name: int(u), v_name: int(v)}
            edge_doc_list.append(doc)

    node_doc_list = []
    for skel_n in skeletonization.nodes:
        if skel_n not in all_matched_nodes:
            continue
        if roi.write_roi.contains(skeletonization.nodes[skel_n][location_attr]):
            node_doc_list.append({"id": int(skel_n), **skeletonization.nodes[skel_n]})

    ids = [x["id"] for x in node_doc_list]
    uids = set(ids)
    assert len(ids) == len(uids)

    try:
        if len(node_doc_list) > 0:
            nodes.insert_many(node_doc_list)
    except BulkWriteError as bwe:
        logger.info(bwe.details)
        raise bwe
    try:
        if len(edge_doc_list) > 0:
            edges.insert_many(edge_doc_list)
    except BulkWriteError as bwe:
        # remove nodes that were
        logger.info(bwe.details)
        raise bwe

    return len(node_doc_list), len(edge_doc_list)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    out_hdlr = logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    out_hdlr.setLevel(logging.INFO)
    logger.addHandler(out_hdlr)
    logger.setLevel(logging.INFO)

    predict_config = sys.argv[1]
    worker_config = sys.argv[2]
    data_config = sys.argv[3]
    graph_config = sys.argv[4]
    solve_config = sys.argv[5]

    worker_config_dict = read_worker_config(worker_config)
    data_config_dict = read_data_config(data_config)
    graph_config_dict = read_graph_config(graph_config)
    solve_config_dict = read_solve_config(solve_config)

    full_config = worker_config_dict
    full_config.update(data_config_dict)
    full_config.update(graph_config_dict)
    full_config.update(solve_config_dict)

    start_time = time.time()
    solve_in_block(**full_config)
    logger.info("Solving took {} seconds".format(time.time() - start_time))

