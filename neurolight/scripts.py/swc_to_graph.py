import networkx as nx
import numpy as np

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# read in an swc. Put it into a networkx graph. Pickle it.


def swc_to_pickle(filename: Path, output_location: Path = None):
    name_parts = filename.name.split(".")
    name = ".".join(name_parts[:-1])
    file_ext = name_parts[-1]

    graph = parse_swc(filename)

    if output_location is not None:
        nx.write_gpickle(graph, output_location / f"{name}.obj")
    else:
        nx.write_gpickle(graph, f"{name}.obj")


def parse_swc(filename: Path) -> nx.DiGraph:
    # swc's are directed
    graph = nx.DiGraph()

    # initialize file specific variables
    header = True
    offset = np.array([0, 0, 0])
    resolution = np.array([1, 1, 1])

    # parse file
    with filename.open() as o_f:
        for line in o_f.read().splitlines():
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

            # extract data from row (point_id, type, x, y, z, radius, parent_id)
            assert int(row[0]) not in graph.nodes, "Duplicate point {} found!".format(
                int(row[0])
            )
            node_id, node_type, x, y, z, radius, parent_id = row
            graph.add_node(
                int(node_id),
                location=(np.array([float(x) for x in row[2:5]]) + offset) * resolution,
                radius=float(row[5]),
                node_type=int(row[1]),
            )
            v, u = int(row[0]), int(row[6])

            assert (u, v) not in graph.edges, "Duplicate edge {} found!".format((u, v))
            if u != v and u is not None and v is not None and u >= 0 and v >= 0:
                graph.add_edge(u, v)
        return graph


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
