import configparser

import daisy


def read_data_config(data_config):
    config = configparser.ConfigParser()
    config.read(data_config)

    cfg_dict = {}

    # Data
    cfg_dict["sample"] = config.get("Data", "sample")
    # Don't think I need these
    offset = config.get("Data", "roi_offset")
    size = config.get("Data", "roi_size")
    cfg_dict["roi_offset"] = daisy.Coordinate(
        tuple(int(x) for x in offset.split(", ")) if not offset == "None" else [None] * 3
    )
    cfg_dict["roi_size"] = daisy.Coordinate(
        tuple(int(x) for x in size.split(", ")) if not size == "None" else [None] * 3
    )
    cfg_dict["roi"] = daisy.Roi(cfg_dict["roi_offset"], cfg_dict["roi_size"])
    cfg_dict["location_attr"] = config.get("Data", "location_attr")
    cfg_dict["penalty_attr"] = config.get("Data", "penalty_attr")
    cfg_dict["target_edge_len"] = int(config.get("Data", "target_edge_len"))

    # Database
    cfg_dict[
        "consensus_db"
    ] = f"mouselight-{cfg_dict['sample']}-{config.get('Data', 'consensus_db')}"
    cfg_dict[
        "subdivided_db"
    ] = f"mouselight-{cfg_dict['sample']}-{config.get('Data', 'subdivided_db')}"
    cfg_dict["db_host"] = config.get("Data", "db_host")

    return cfg_dict
