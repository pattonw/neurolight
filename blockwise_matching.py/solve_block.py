import daisy
from daisy.persistence import MongoDbGraphProvider
import json
import logging
import numpy as np
import os
import sys
import time

from daisy_check_functions import check_function, write_done
import configparser
from config_parser import (
    read_solve_config,
    read_data_config,
    read_worker_config,
    read_graph_config,
)
from funlib.match import match
from neurolight.match import get_costs, mouselight_preprocessing

logger = logging.getLogger(__name__)


def solve_in_block(
    db_host,
    consensus_db,
    skeletonization_db,
    matched_db,
    time_limit,
    solve_number,
    graph_number,
    selected_attr="selected",
    solved_attr="solved",
    **kwargs,
):

    logger.info("Solve in block")

    consensus_provider = MongoDbGraphProvider(consensus_db, db_host, mode="r+")

    skeletonization_provider = MongoDbGraphProvider(
        skeletonization_db, db_host, mode="r+"
    )

    matched_provider = MongoDbGraphProvider(matched_db, db_host, mode="r+")

    client = daisy.Client()

    while True:
        logger.info("Acquire block")
        block = client.acquire_block()

        if not block:
            return 0

        logger.debug("Solving in block %s", block)

        if check_function(block, "solve_s{}".format(solve_number), matched_db, db_host):
            client.release_block(block, 0)
            continue

        start_time = time.time()
        consensus = consensus_provider.get_graph(block.read_roi)
        skeletonization = skeletonization_provider.get_graph(block.read_roi)
        # anything in matched was solved previously and must be maintained.
        pre_solved = matched_provider.get_graph(block.read_roi)

        logger.info(
            f"Reading consensus with {len(consensus.nodes)} nodes and "
            + f"{len(consensus.edges)} edges took {time.time() - start_time} seconds"
        )
        logger.info(
            f"Reading skeletonization with {len(skeletonization.nodes)} nodes and "
            + f"{len(skeletonization.edges)} edges took {time.time() - start_time} seconds"
        )

        if len(consensus.nodes) == 0:
            logger.info(f"No consensus nodes in roi {block.read_roi}. Skipping")
            write_done(block, f"solve_s{solve_number}", matched_db, db_host)
            client.release_block(block, 0)
            continue

        logger.info("Solve...")
        start_time = time.time()
        overcomplete = mouselight_preprocessing(
            skeletonization,
            max_dist=max_dist,
            voxel_size=voxel_size,
            penalty_attr=penalty_attr,
            location_attr=location_attr,
        )
        node_costs, edge_costs = get_costs(
            skeletonization,
            consensus,
            location_attr=location_attr,
            penalty_attr=penalty_attr,
            node_match_threshold=node_match_threshold,
            edge_match_threshold=edge_match_threshold,
            node_balance=node_balance,
        )

        logger.info(
            f"Preprocessing and initializing costs took {time.time() - start_time} seconds"
        )

        matched = match(overcomplete, consensus, node_costs, edge_costs)

        logger.info("Solving took %s seconds" % (time.time() - start_time))

        start_time = time.time()
        write_matched(matched, matched_provider)

        logger.info(
            "Writing matched graph with {len(matched.nodes)} nodes took {time.time()-start_time} seconds"
        )

        logger.info("Write done")
        write_done(block, "solve_s{}".format(solve_number), db_name, db_host)

        logger.info("Release block")
        client.release_block(block, 0)

    return 0


def write_matched(matched, matched_provider):
    raise NotImplementedError()


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
    print("Solving took {} seconds".format(time.time() - start_time))
