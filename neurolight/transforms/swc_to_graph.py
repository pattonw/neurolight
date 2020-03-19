import networkx as nx
import numpy as np
from tqdm import tqdm

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# This module provides functions for transforming between swc's to networkx graphs.
# For simplicity, networkx graphs are then pickled an stored as .obj files.
# Mouselight skeletons are provided in swc's with x,y,z coordinates stored in microns.
# Converting from microns to voxel coordinates is done using the transform.txt file.
# This script stores locations in integer voxel coordinates.
# Origin and spacing are stored as graph attributes so that micron coordinates can be computed.
# Node indices start ar 1. -1 as parent index means no parent, i.e. root (soma).
# Each swc contains a single connected component.
# point_type of 42 means machine generated point, 43 means human generated.

# "Consensus" neuron files contain only axon, dendrites are stored in a seperate swc.
# Here they are combined so that we have 1 file per consensus neuron.

# Since we know the part of the neuron, we may as well keep that data.
# This script adds a "neuron_part" attribute that is 0 for soma, 1 for axon, 2 for dendrite


def swc_to_pickle(
    axon: Path,
    dendrite: Path,
    transform: Path,
    output_file: Path = None,
    offset=None,
    resolution=None,
    transpose=[0, 1, 2],
):
    name_parts = axon.name.split(".")
    file_ext = name_parts[-1]
    assert (
        file_ext == "swc"
    ), f"This function is inteded to work on swc files, not {file_ext} files"

    if offset is None and resolution is None:
        offset, resolution = load_transform(transform)

    consensus_graph = parse_consensus(
        axon, dendrite, transform, offset, resolution, transpose
    )
    nx.write_gpickle(consensus_graph, output_file)


def parse_consensus(
    axon: Path,
    dendrite: Path,
    transform: Path,
    offset=np.array([0, 0, 0]),
    resolution=np.array([1, 1, 1]),
    transpose=[0, 1, 2],
):

    axon_graph = parse_swc(axon, transform, offset, resolution, transpose)
    assert nx.is_arborescence(axon_graph), "Axon graph is not an arborescence!"
    dendrite_graph = parse_swc(dendrite, transform, offset, resolution, transpose)
    assert nx.is_arborescence(dendrite_graph), "Dendrite graph is not an arborescence!"

    consensus_graph = merge_graphs(axon_graph, dendrite_graph)
    consensus_graph.graph["spacing"] = resolution
    consensus_graph.graph["origin"] = offset
    return consensus_graph


def merge_graphs(axon: nx.DiGraph, dendrite: nx.DiGraph) -> nx.DiGraph:
    axon = nx.convert_node_labels_to_integers(axon, ordering="sorted")
    for node, node_attrs in axon.nodes.items():
        node_attrs["neuron_part"] = 1
    next_node_id = len(axon.nodes)
    dendrite = nx.convert_node_labels_to_integers(
        dendrite, ordering="sorted", first_label=next_node_id
    )
    nx.relabel.relabel_nodes(dendrite, {next_node_id: 0}, copy=False)
    for node, node_attrs in dendrite.nodes.items():
        node_attrs["neuron_part"] = 0 if node == 0 else 2

    consensus_graph = nx.compose(axon, dendrite)

    for node, node_attrs in consensus_graph.nodes.items():
        assert "neuron_part" in node_attrs

    assert nx.is_arborescence(
        consensus_graph
    ), "Consensus graph is not an arborescence!"

    assert all(
        np.isclose(axon.nodes[0]["location"], consensus_graph.nodes[0]["location"])
    )
    return consensus_graph


def load_transform(transform_path: Path):
    text = transform_path.open("r").read()
    lines = text.split("\n")
    constants = {}
    for line in lines:
        if len(line) > 0:
            variable, value = line.split(":")
            constants[variable] = float(value)
    spacing = (
        np.array([constants["sx"], constants["sy"], constants["sz"]])
        / 2 ** (constants["nl"] - 1)
        / 1000
    )
    origin = spacing * (
        (np.array([constants["ox"], constants["oy"], constants["oz"]]) // spacing)
        / 1000
    )
    return origin, spacing


def micron_to_voxel_coords(
    coord: np.ndarray, origin: np.ndarray, spacing: np.ndarray
) -> np.ndarray:
    """
    Assuming origin [0,0,0] and spacing [1,1,1], rounding means that voxel [0,0,0]
    contains all points from [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]. This means that
    voxel [0,0,0] has an integer valued center.
    """
    return np.round((coord - origin) / spacing).astype(int)


def parse_swc(
    filename: Path,
    transform: Path,
    offset=np.array([0, 0, 0]),
    resolution=np.array([1, 1, 1]),
    transpose=[0, 1, 2],
) -> nx.DiGraph:
    # swc's are directed
    graph = nx.DiGraph()

    origin, spacing = load_transform(transform)

    # parse file
    for node_id, vx, vy, vz, r, pt, hp, parent_id in node_gen(
        filename, origin, spacing
    ):
        location = ((np.array([vx, vy, vz]) + offset) * resolution).take(transpose)
        graph.add_node(
            node_id, location=location, radius=r, point_type=pt, human_placed=hp
        )

        if parent_id >= 0:
            graph.add_edge(parent_id, node_id)
    return graph


def parse_swc_no_transform(
    filename: Path,
    offset=np.array([0, 0, 0]),
    resolution=np.array([1, 1, 1]),
    transpose=[0, 1, 2],
) -> nx.DiGraph:
    # swc's are directed
    graph = nx.DiGraph()

    # parse file
    for node_id, vx, vy, vz, r, pt, hp, parent_id in node_gen(filename):
        location = ((np.array([vx, vy, vz]) + offset) * resolution).take(transpose)
        graph.add_node(
            node_id, location=location, radius=r, point_type=pt, human_placed=hp
        )

        if parent_id >= 0:
            graph.add_edge(parent_id, node_id)
    return graph


def node_gen(filename: Path, origin=np.array([0, 0, 0]), spacing=np.array([1, 1, 1])):
    # initialize file specific variables
    header = True

    offset = np.array([0, 0, 0])
    resolution = np.array([1, 1, 1])

    # parse file
    with filename.open() as o_f:
        for line in tqdm(o_f.read().splitlines(), f"Parsing {filename.name}"):
            if header and line.startswith("#"):
                # Search header comments for variables
                offset = _search_swc_header(line, "offset", offset)
                resolution = _search_swc_header(line, "resolution", resolution)
                continue
            elif line.startswith("#"):
                # comments not in header get skipped
                continue
            elif header:
                # first line without a comment marks end of header
                header = False

            row = line.strip().split()
            if len(row) != 7:
                raise ValueError("SWC has a malformed line: {}".format(line))

            node_id, node_type, x, y, z, radius, parent_id = row
            location = (np.array([float(c) for c in (x, y, z)]) + offset) * resolution
            voxel_location = micron_to_voxel_coords(location, origin, spacing)

            x, y, z = voxel_location.tolist()
            node_id, parent_id = int(row[0]), int(row[6])

            r = float(row[5])
            pt = int(row[1])
            hp = int(row[1]) == 43

            yield (node_id, x, y, z, r, pt, hp, parent_id)


def _search_swc_header(line: str, key: str, default: np.ndarray) -> np.ndarray:
    # assumes header variables are seperated by spaces
    if key in line.lower():
        try:
            value = np.array([float(x) for x in line.lower().split(key)[1].split()])
        except Exception as e:
            logger.debug("Invalid line: {}".format(line))
            raise e
        return value
    else:
        return default
