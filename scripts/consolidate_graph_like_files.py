from tqdm import tqdm
import daisy
import numpy as np
import networkx as nx

from neurolight.transforms.txt_to_graph import txt_to_pickle, node_gen
from neurolight.transforms.swc_to_graph import swc_to_pickle, parse_consensus

from pathlib import Path
import shutil
import logging

from mongo_bulk import bulk_write_edges, bulk_write_nodes, write_metadata

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# data sources
tracing_source = Path("/groups/mousebrainmicro/mousebrainmicro/tracing_complete")
axon_source_template = "{node_id}/consensus.swc"
dendrite_source_template = "{node_id}/dendrite.swc"
skeletonization_source_template = "/groups/mousebrainmicro/mousebrainmicro/cluster/Reconstructions/{sample}/skeleton-graph.txt"
transform_source_template = "/nrs/mouselight/SAMPLES/{sample}/transform.txt"

# data targets
target = Path("/nrs/funke/mouselight-v2")
consensus_target_template = (
    "/nrs/funke/mouselight-v2/{sample}/consensus_tracings/{node_id}.obj"
)

# samples of interest
samples = (
    # "2017-09-25",
    # "2017-11-17",
    # "2018-03-09",
    # "2018-04-25",
    # "2018-05-23",
    # "2018-06-14",
    # "2018-07-02",
    # "2018-08-01",
    "2018-10-01",
    "2018-12-01",
)


def ingest_skeletonization(
    sample: str, skeletonization_file: Path, transform_file: Path
):
    """
    Storing data in MongoDB using psuedo world coords (1,.3,.3) microns rather than the
    slightly off floats found in the transform.txt file.
    """
    # checkout robomongo
    url = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea"

    mongo_graph_provider = daisy.persistence.MongoDbGraphProvider(
        f"mouselight-{sample}-skeletonization", url, directed=False, mode="w"
    )
    # graph = mongo_graph_provider.get_graph(
    #     daisy.Roi(
    #         daisy.Coordinate([None, None, None]), daisy.Coordinate([None, None, None])
    #     )
    # )

    count = 0
    node_ids, positions = [], []
    for node_id, x, y, z, neighbors in node_gen(
        skeletonization_file,
        transform_file,
        origin=np.zeros(3),
        spacing=np.array([300, 300, 1000]),
    ):
        node_id = int(np.int64(node_id))
        count += 1
        node_ids.append(node_id)
        positions.append([z, y, x])
        if count > 10_000_000:
            count = 0
            bulk_write_nodes(
                url,
                f"mouselight-{sample}-skeletonization",
                "nodes",
                {"id": node_ids, "position": positions},
            )
            node_ids = []
            positions = []
    bulk_write_nodes(
        url,
        f"mouselight-{sample}-skeletonization",
        "nodes",
        {"id": node_ids, "position": positions},
    )

    count = 0
    edges = []
    for node_id, x, y, z, neighbors in node_gen(
        skeletonization_file,
        transform_file,
        origin=np.zeros(3),
        spacing=np.array([300, 300, 1000]),
    ):
        node_id = int(np.int64(node_id))
        for neighbor in neighbors:
            neighbor = int(np.int64(neighbor))
            if neighbor > node_id:
                count += 1
                edges.append((node_id, neighbor))
        if count > 10_000_000:
            count = 0
            bulk_write_edges(
                url, f"mouselight-{sample}-skeletonization", "edges", ("u", "v"), edges, False
            )
            edges = []
    bulk_write_edges(
        url, f"mouselight-{sample}-skeletonization", "edges", ("u", "v"), edges, False
    )


def ingest_consensus(sample: str, consensus_dir: Path, transform_file: Path):
    """
    Storing data in MongoDB using psuedo world coords (1,.3,.3) microns rather than the
    slightly off floats found in the transform.txt file.
    """
    # checkout robomongo
    url = "mongodb://linajeaAdmin:FeOOHnH2O@funke-mongodb4/admin?replicaSet=rsLinajea"

    mongo_graph_provider = daisy.persistence.MongoDbGraphProvider(
        f"mouselight-{sample}-consensus", url, directed=True, mode="w"
    )
    graph = mongo_graph_provider.get_graph(
        daisy.Roi(
            daisy.Coordinate([None, None, None]), daisy.Coordinate([None, None, None])
        )
    )
    consensus_graphs = []
    for consensus_neuron in tqdm(consensus_dir.iterdir(), "Consensus neurons: "):
        if (
            not consensus_neuron.is_dir()
            or not (consensus_neuron / "consensus.swc").exists()
        ):
            continue
        consensus_graph = parse_consensus(
            consensus_neuron / "consensus.swc",
            consensus_neuron / "dendrite.swc",
            transform,
            offset=np.array([0, 0, 0]),
            resolution=np.array([300, 300, 1000]),
            transpose=[2, 1, 0],
        )
        for node in consensus_graph.nodes:
            consensus_graph.nodes[node]["position"] = consensus_graph.nodes[node][
                "location"
            ].tolist()
            del consensus_graph.nodes[node]["location"]
        consensus_graphs.append(consensus_graph)
    logger.info("Consolidating consensus graphs!")
    consensus_graph = nx.disjoint_union_all(consensus_graphs)

    data = {}
    for node_id, attrs in consensus_graph.nodes.items():
        node_id = int(np.int64(node_id))
        node_ids = data.setdefault("id", [])
        node_ids.append(node_id)
        for key, value in attrs.items():
            dlist = data.setdefault(key, [])
            dlist.append(value)

    logger.info(
        f"Writing {len(consensus_graph.nodes)} nodes and {len(consensus_graph.edges)} edges!"
    )

    bulk_write_nodes(url, f"mouselight-{sample}-consensus", "nodes", data)
    bulk_write_edges(
        url,
        f"mouselight-{sample}-consensus",
        "edges",
        ("u", "v"),
        list(consensus_graph.edges),
        True,
    )


for sample in tqdm(samples, "Samples: "):
    sample_tracings = tracing_source / sample
    assert sample_tracings.is_dir(), f"{sample_tracings} is not a directory!"

    transform = Path(transform_source_template.format(sample=sample))
    assert transform.exists(), f"Sample {sample} has no transform.txt at {transform}!"

    # read skeletonization and move it
    skeletonization = Path(skeletonization_source_template.format(sample=sample))
    assert (
        skeletonization.exists()
    ), f"Sample {sample} has no skeleton-graph.txt at {skeletonization}"
    
    if sample == "2018-07-02":
        ingest_consensus(sample, sample_tracings, transform)
    else:
        continue
    logger.info(f"Starting sample {sample}!")
    continue

    # ingest_consensus(sample, sample_tracings, transform)
    # skeletonizations for samples 2018-10-01 and 2018-12-01 need to be reuploaded
    # due to pixel size issues.
    ingest_skeletonization(sample, skeletonization, transform)
