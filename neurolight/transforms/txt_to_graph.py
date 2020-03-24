import networkx as nx
import numpy as np

from pathlib import Path
import logging

from tqdm import tqdm

from typing import Tuple, Generator, List

logger = logging.getLogger(__name__)

# read in a text file provided by Adam.
# Note: coordinates (x, y, z) are provided in voxel coordinates
# Note: coordinates are 1 based. I subtract 1 during parsing to make them 0 based


def txt_to_pickle(filename: Path, transform_path: Path, output_location: Path = None):
    name_parts = filename.name.split(".")
    name = ".".join(name_parts[:-1])
    file_ext = name_parts[-1]
    assert file_ext == "txt", "This function is intended for .txt files"

    graph = parse_txt(filename, transform_path)

    nx.write_gpickle(graph, output_location)


def load_transform(transform_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
    return np.round((coord - origin) / spacing).astype(int)


def voxel_to_micron_coords(
    coord: np.ndarray, origin: np.ndarray, spacing: np.ndarray
) -> np.ndarray:
    """
    Assuming voxel coordinate is 0 based.

    """
    return coord * spacing + origin


def parse_txt(filename: Path, transform_path: Path) -> nx.Graph:
    """
    original skeletonizations are stored in .txt files of the form:
    node-index, x, y, z, neighbor-1-node_index, neighbor-2-node-index, ...
    """
    # swc's are directed
    graph = nx.DiGraph()

    origin, spacing = load_transform(transform_path)
    graph.graph["origin"] = origin
    graph.graph["spacing"] = spacing

    # parse file
    for node_id, x, y, z, neighbors in node_gen(filename, transform_path):
        location = voxel_to_micron_coords(np.array(x, y, z))
        graph.add_node(int(node_id), location=location)
        for v in neighbors:
            graph.add_edge(int(node_id), int(v))

    return graph


def node_gen(filename: Path, transform_path: Path, origin=None, spacing=None):
    """
    original skeletonizations are stored in .txt files of the form:
    node-index, x, y, z, neighbor-1-node_index, neighbor-2-node-index, ...
    """
    # swc's are directed

    if origin is None or spacing is None:
        origin, spacing = load_transform(transform_path)

    # parse file
    with filename.open() as o_f:
        for line in tqdm(o_f.read().splitlines()):
            row = line.strip().split()
            if len(row) <= 4:
                raise ValueError("Skeleton file has a malformed line: {}".format(line))

            node_id, x, y, z, *neighbors = row
            location = voxel_to_micron_coords(
                np.array([float(x), float(y), float(z)]) - 1, origin, spacing
            )
            x, y, z = location.tolist()
            yield (node_id, x, y, z, list(neighbors))
