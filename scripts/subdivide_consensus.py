from daisy.persistence import MongoDbGraphProvider
import numpy as np
import networkx as nx

import itertools
from pathlib import Path
import logging
from tqdm import tqdm

from config_parser import read_data_config

logging.basicConfig(level=logging.INFO)

config = read_data_config(Path("config.ini"))

print(config["sample"])

consensus_provider = MongoDbGraphProvider(
    config["consensus_db"], config["db_host"], mode="r+", directed=True
)

subdivided_provider = MongoDbGraphProvider(
    config["subdivided_db"], config["db_host"], mode="w", directed=True
)

consensus = consensus_provider.get_graph(
    config["roi"], node_inclusion="dangling", edge_inclusion="either"
)

print(f"Read consensus with {len(consensus.nodes)} nodes")

subdivided = subdivided_provider.get_graph(
    config["roi"], node_inclusion="dangling", edge_inclusion="either"
)
print(f"Read subdivided with {len(subdivided.nodes)} nodes")

target_edge_len = config["target_edge_len"]
pos_attr = config["location_attr"]

assert len(subdivided.nodes) == 0

node_id = itertools.count()

id_map = {}

for node, attrs in tqdm(list(consensus.nodes.items())):
    current_id = id_map.setdefault(node, next(node_id))
    subdivided.add_node(
        int(np.int64(current_id)),
        parent_u=int(np.int64(node)),
        parent_v=int(np.int64(node)),
        **attrs,
    )

    for neighbor in consensus.neighbors(node):
        u_loc = np.array(consensus.nodes[node][pos_attr])
        v_loc = np.array(consensus.nodes[neighbor][pos_attr])
        edge_len = np.linalg.norm(u_loc - v_loc)
        if np.isclose(edge_len, 0):
            neighbor_id = id_map.setdefault(neighbor, next(node_id))
            subdivided.add_edge(int(np.int64(current_id)), int(np.int64(neighbor_id)))
            continue
        diff = v_loc - u_loc
        slope = diff / edge_len
        step = target_edge_len * slope

        # last_subdivided = None
        last_subdivided = current_id

        for i in range(1, int((edge_len - target_edge_len / 2) // target_edge_len)):
            subdivided_id = next(node_id)
            subdivided_loc = u_loc + i * step
            subdivided.add_node(
                int(np.int64(subdivided_id)),
                parent_u=int(np.int64(node)),
                parent_v=int(np.int64(neighbor)),
                **{pos_attr: tuple(int(np.int64(x)) for x in subdivided_loc)},
            )
            subdivided.add_edge(
                int(np.int64(last_subdivided)), int(np.int64(subdivided_id))
            )
            last_subdivided = subdivided_id

        if last_subdivided != current_id:
            neighbor_id = id_map.setdefault(neighbor, next(node_id))
            subdivided.add_edge(last_subdivided, neighbor_id)
        else:
            neighbor_id = id_map.setdefault(neighbor, next(node_id))
            subdivided.add_edge(int(np.int64(current_id)), int(np.int64(neighbor_id)))


consensus_ccs = len(list(nx.weakly_connected_components(consensus)))
subdivided_ccs = len(list(nx.weakly_connected_components(subdivided)))
assert (
    consensus_ccs == subdivided_ccs
), f"consensus_ccs {consensus_ccs} != subdivided_ccs: {subdivided_ccs}"

for i in range(10):
    if i == 0 or i == 2:
        continue
    else:
        consensus_nodes = [n for n in consensus.nodes if consensus.degree(n) == i]
        subdivided_nodes = [n for n in subdivided.nodes if subdivided.degree(n) == i]
        assert len(consensus_nodes) == len(subdivided_nodes)

print(f"subdivided now has {len(subdivided.nodes)} nodes")

subdivided.write_nodes()
subdivided.write_edges()
