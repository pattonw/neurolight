import numpy as np
import networkx as nx

from tqdm import tqdm

from pathlib import Path
import time

consensus_pickles = Path("/nrs/funke/mouselight-v2/2018-07-02/consensus_tracings")


def parse_npy_graph(filename):
    t1 = time.time()
    a = np.load(filename)
    edge_list = a["edge_list"]
    node_ids = a["node_ids"]
    locations = a["locations"]
    point_type = a["point_type"]
    radius = a["radius"]
    neuron_part = a["neuron_part"]
    human_placed = a["human_placed"]
    t2 = time.time()
    graph = nx.DiGraph()
    graph.add_edges_from(edge_list)
    graph.add_nodes_from(
        [
            (
                node_ids[i],
                {
                    "location": locations[i],
                    "radius": radius[i],
                    "point_type": point_type[i],
                    "neuron_part": neuron_part[i],
                    "human_placed": human_placed[i],
                },
            )
            for i in range(len(node_ids))
        ]
    )
    return graph, t2 - t1


pickle_load_time = []
pickle_load_count = []
npy_load_time = []
npy_load_count = []
npy_process_time = []
for pickled_consensus in tqdm(consensus_pickles.iterdir()):
    t1 = time.time()
    graph_pickle = nx.read_gpickle(pickled_consensus)
    t2 = time.time()
    pickle_load_count.append(len(graph_pickle.nodes))
    pickle_load_time.append((t2 - t1) / len(graph_pickle.nodes))

    edge_list = np.array(graph_pickle.edges, dtype=int)
    node_ids = np.array(graph_pickle.nodes, dtype=int)
    locations = np.array([graph_pickle.nodes[nid]["location"] for nid in node_ids])
    radius = np.array(
        [graph_pickle.nodes[nid]["radius"] for nid in node_ids], dtype=int
    )
    point_type = np.array(
        [graph_pickle.nodes[nid]["point_type"] for nid in node_ids], dtype=int
    )
    neuron_part = np.array(
        [graph_pickle.nodes[nid]["neuron_part"] for nid in node_ids], dtype=int
    )
    human_placed = np.array(
        [graph_pickle.nodes[nid]["human_placed"] for nid in node_ids], dtype=bool
    )

    np.savez(
        "test",
        edge_list=edge_list,
        node_ids=node_ids,
        locations=locations,
        radius=radius,
        point_type=point_type,
        neuron_part=neuron_part,
        human_placed=human_placed,
    )

    t1 = time.time()
    graph_npy, load_time = parse_npy_graph("test.npz")
    t2 = time.time()
    npy_load_count.append(len(graph_npy.nodes))
    npy_load_time.append(load_time / len(graph_npy.nodes))
    npy_process_time.append((t2 - t1) / len(graph_npy.nodes))

print(
    f"Pickle loading time is about {sum(pickle_load_time) / sum(pickle_load_count)} vs numpy {sum(npy_load_time)/sum(npy_load_count)}"
)

