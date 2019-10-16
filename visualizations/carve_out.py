import networkx as nx
from scipy.spatial import cKDTree
from tqdm import tqdm
import numpy as np

from neurolight.transforms.txt_to_graph import parse_txt
from neurolight.transforms.swc_to_graph import parse_consensus

from pathlib import Path

axon_path = Path(
    "/groups/mousebrainmicro/mousebrainmicro/tracing_complete/2018-07-02/G-002/consensus.swc"
)
dendrite_path = Path(
    "/groups/mousebrainmicro/mousebrainmicro/tracing_complete/2018-07-02/G-002/dendrite.swc"
)
transform = Path("/nrs/mouselight/SAMPLES/2018-07-02/transform.txt")


skeletonization_path = Path(
    "/groups/mousebrainmicro/mousebrainmicro/cluster/Reconstructions/2018-07-02/skeleton-graph.txt"
)

if Path("skeletonization.obj").exists():
    skeletonization = nx.read_gpickle("skeletonization.obj")
else:
    skeletonization = parse_txt(skeletonization_path, transform)
    nx.write_gpickle(skeletonization, "skeletonization.obj")

if Path("consensus-002.obj").exists():
    consensus = nx.read_gpickle("consensus-002.obj")
else:
    consensus = parse_consensus(axon_path, dendrite_path, transform)
    nx.write_gpickle(consensus, "consensus-002.obj")

ids, attrs = (list(x) for x in zip(*skeletonization.nodes.items()))
skeletonization_points = cKDTree(
    [node_attrs["location"] for node_attrs in attrs]
)

point_counts = {}
carved_points = []
radius = 100  # microns
for node, node_attrs in tqdm(consensus.nodes.items(), "carving points"):
    location = node_attrs["location"]
    node_points = skeletonization_points.query_ball_point(
        location, radius, p=float("inf")
    )
    point_counts[node] = len(node_points)
    carved_points += node_points
carved_points = set(carved_points)

carved_subgraph = nx.DiGraph()
carved_subgraph.add_nodes_from(
    (ids[n], skeletonization.nodes[ids[n]])
    for n in tqdm(carved_points, "Adding carved nodes to subgraph")
)
carved_subgraph.add_edges_from(
    (ids[n], nbr, d)
    for n in tqdm(carved_points, "Adding edges to subgraph")
    for nbr, d in skeletonization.adj[ids[n]].items()
    if nbr in carved_points
)

carved_subgraph.graph["origin"] = skeletonization.graph["origin"]
carved_subgraph.graph["spacing"] = skeletonization.graph["spacing"]

nx.write_gpickle(carved_subgraph, f"skeletonization_carved-002-{radius}.obj")

node_ids, counts = zip(*point_counts.items())
print(
    f"count percentiles: {np.percentile(counts, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])}"
)

