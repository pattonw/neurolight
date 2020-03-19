import configparser
import os
import numpy as np
import json


def read_graph_config(graph_config):
    config = configparser.ConfigParser()
    config.read(graph_config)

    cfg_dict = {}
    cfg_dict["graph_number"] = int(config.get("Graph", "graph_number"))
    cfg_dict["block_size"] = tuple(
        [
            int(v)
            for v in np.array(config.get("Graph", "block_size").split(", "), dtype=int)
        ]
    )
    cfg_dict["build_graph"] = config.get("Graph", "build_graph")

    return cfg_dict


def read_solve_config(solve_config):
    config = configparser.ConfigParser()
    config.read(solve_config)

    cfg_dict = {}
    # Solve
    cfg_dict["max_new_edge"] = float(config.get("Solve", "max_new_edge"))
    cfg_dict["max_match_dist"] = float(config.get("Solve", "max_match_dist"))
    cfg_dict["expected_edge_len"] = float(config.get("Solve", "expected_edge_len"))
    cfg_dict["node_weight"] = float(config.get("Solve", "node_weight"))

    cfg_dict["context"] = tuple(
        np.array(config.get("Solve", "context").split(", "), dtype=int)
    )
    cfg_dict["daisy_solve"] = config.get("Solve", "daisy_solve")
    cfg_dict["solve_block"] = config.get("Solve", "solve_block")
    cfg_dict["solve_number"] = int(config.get("Solve", "solve_number"))
    time_limit = config.get("Solve", "time_limit")
    if time_limit == "None":
        cfg_dict["time_limit"] = None
    else:
        cfg_dict["time_limit"] = int(time_limit)

    return cfg_dict


def read_worker_config(worker_config):
    config = configparser.ConfigParser()
    config.read(worker_config)

    cfg_dict = {}

    # Worker
    cfg_dict["singularity_container"] = config.get("Worker", "singularity_container")
    cfg_dict["num_cpus"] = int(config.getint("Worker", "num_cpus"))
    cfg_dict["num_block_workers"] = int(config.getint("Worker", "num_block_workers"))
    cfg_dict["num_cache_workers"] = int(config.getint("Worker", "num_cache_workers"))
    cfg_dict["queue"] = config.get("Worker", "queue")
    cfg_dict["mount_dirs"] = tuple(
        [v for v in config.get("Worker", "mount_dirs").split(", ")]
    )

    return cfg_dict


def read_data_config(data_config):
    config = configparser.ConfigParser()
    config.read(data_config)

    cfg_dict = {}

    # Data
    cfg_dict["sample"] = config.get("Data", "sample")
    cfg_dict["roi_offset"] = tuple(
        int(x) for x in config.get("Data", "roi_offset").split(", ")
    )
    cfg_dict["roi_size"] = tuple(
        int(x) for x in config.get("Data", "roi_size").split(", ")
    )
    cfg_dict["node_balance"] = float(config.get("Data", "node_balance"))
    cfg_dict["location_attr"] = config.get("Data", "location_attr")
    cfg_dict["penalty_attr"] = config.get("Data", "penalty_attr")
    cfg_dict["target_attr"] = config.get("Data", "target_attr")

    # Database
    cfg_dict[
        "consensus_db"
    ] = f"mouselight-{cfg_dict['sample']}-{config.get('Database', 'consensus_db')}"
    cfg_dict[
        "skeletonization_db"
    ] = f"mouselight-{cfg_dict['sample']}-{config.get('Database', 'skeletonization_db')}"
    cfg_dict[
        "matched_db"
    ] = f"mouselight-{cfg_dict['sample']}-{config.get('Database', 'matched_db')}"
    cfg_dict["consensus"] = config.get("Database", "consensus_db")
    cfg_dict["skeletonization"] = config.get("Database", "skeletonization_db")
    cfg_dict["matched"] = config.get("Database", "matched_db")
    cfg_dict["db_host"] = config.get("Database", "db_host")
    cfg_dict["u_name"] = config.get("Database", "u_name")
    cfg_dict["v_name"] = config.get("Database", "v_name")

    # Create json container spec for in_data:
    in_container_spec = {
        # "container": cfg_dict["in_container"],
        "offset": cfg_dict["roi_offset"],
        "size": cfg_dict["roi_size"],
    }

    in_container_spec_file = os.path.join(
        os.path.dirname(data_config), "in_container_spec.json"
    )

    with open(in_container_spec_file, "w+") as f:
        json.dump(in_container_spec, f)

    cfg_dict["in_container_spec"] = os.path.abspath(in_container_spec_file)

    return cfg_dict
