from gunpowder.graph import Graph, Node, GraphKey
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.points_spec import PointsSpec
from gunpowder.batch import Batch
from gunpowder.profiling import Timing

import numpy as np
from scipy.spatial.ckdtree import cKDTree, cKDTreeNode

from pathlib import Path
from typing import List, Dict, Tuple
import logging
import copy
import networkx as nx
import itertools

logger = logging.getLogger(__name__)


class GraphSource(BatchProvider):
    """Read a graph in a pickled networkx graph file or directory of files

    Args:

        filename:

            The file or directory to read from.

        points:

            The key of the points set to create.

        points_spec:

            An optional :class:`PointsSpec`. Default behavior is to provide
            an infinite roi with unit voxel_size

        scale:

            An option to allow scaling of points. This should be the voxel size of
            the associated image data.

        transpose:

            Point locations are stored in xyz coordinates. A transpose of [2,1,0]
            will flip them into zyx ordering.
    """

    def __init__(
        self,
        filename: Path,
        points: List[GraphKey],
        points_spec: List[PointsSpec] = None,
        scale: Coordinate = Coordinate([1, 1, 1]),
        transpose: Tuple[int] = (0, 1, 2),
        add_small_edges: bool = True,
    ):
        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.connected_component_label = 0
        self._graph = nx.Graph()
        self.transpose = transpose

    @property
    def g(self):
        return self._graph

    def setup(self):
        logger.debug("Initializing graph source")
        self._read_points()
        logger.debug("Graph source initialized")

        if self.points_spec is not None:

            assert len(self.points) == len(self.points_spec)
            for point, point_spec in zip(self.points, self.points_spec):
                self.provides(point, point_spec)
        else:
            logger.debug("No point spec provided!")

            roi = Roi(Coordinate((None, None, None)), Coordinate((None, None, None)))

            for point in self.points:
                self.provides(point, PointsSpec(roi=roi))

    def provide(self, request: BatchRequest) -> Batch:

        timing = Timing(self, "provide")
        timing.start()

        batch = Batch()

        for points_key in self.points:

            if points_key not in request:
                continue

            # Retrieve all points in the requested region using a kdtree for speed
            point_ids = self._query_kdtree(
                self.data.tree,
                (
                    np.array(request[points_key].roi.get_begin()),
                    np.array(request[points_key].roi.get_end()),
                ),
            )

            # To account for boundary crossings we must retrieve neighbors of all points
            # in the graph. This is too slow for large queries and less important
            points_subgraph = self._subgraph_points(
                point_ids, with_neighbors=len(point_ids) < len(self._graph.nodes) // 2
            )

            # Handle boundary cases
            points_subgraph = points_subgraph.crop(request[points_key].roi)

            batch = Batch()
            batch.points[points_key] = Node._from_graph(
                points_subgraph, request[points_key]
            )

            logger.debug(
                "Graph points source provided {} points for roi: {}".format(
                    len(batch.points[points_key].data), request[points_key].roi
                )
            )

            logger.debug(
                f"Providing {len(points_subgraph.nodes)} nodes to {points_key}"
            )

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _read_points(self) -> None:
        logger.info("starting to read points")
        filepath = Path(self.filename)
        # handle missing file case
        if not filepath.exists():
            raise FileNotFoundError(
                "file {} does not exist at {}".format(
                    filepath.name, filepath.absolute()
                )
            )

        if filepath.is_file():
            # read from single file
            self._read_graph(filepath)
        elif filepath.is_dir():
            # read from directory
            for graph_file in filepath.iterdir():
                if graph_file.name.endswith(".obj"):
                    self._read_graph(graph_file)

        logger.info("finished reading points, now processing")
        self._process_locations()
        self._graph_to_kdtree()
        logger.info("finished processing points")

    def _process_locations(self):
        """
        We want coordinates to be stored in world coordinates, i.e. microns,
        however the origin and spacing terms for converting between microns
        and voxel coordinates in the LargeVolumeViewer (LVV) are not
        Integers, but rather floats. This means we can not simply read and provide
        the locations since they will not fall in their expected rois.

        Thus each graph has a spacing and an origin attribute which can be used
        to calcualte the voxel coordinate, which can then be multiplied by
        the approximate voxel size to get psuedo world coordinates for each point.
        """
        origin = self._graph.graph.get("origin")
        if origin is None:
            origin = np.array([0, 0, 0])
            logger.warning(f"Origin not provided by graph! Using: {origin}")
        spacing = self._graph.graph.get("spacing")
        if spacing is None:
            spacing = np.array([1, 1, 1])
            logger.warning(f"Spacing not provided by graph! Using: {spacing}")
        for node, attrs in self._graph.nodes.items():
            attrs["location"] = (
                ((attrs["location"] - origin) / spacing).take(self.transpose)
                * self.scale
            ).astype(np.float32)

    def _graph_to_kdtree(self) -> None:
        # add node_ids to coordinates to support overlapping nodes in cKDTree
        ids, attrs = zip(*self.g.nodes.items())
        assert all(
            [a == b for a, b in zip(ids, range(len(ids)))]
        ), "ids are not properly sorted"
        data = [point_attrs["location"].tolist() for point_attrs in attrs]
        logger.debug("placing {} nodes in the kdtree".format(len(data)))
        self.data = cKDTree(np.array(list(data)))
        logger.debug("kdtree initialized".format(len(data)))

    def _query_kdtree(
        self, node: cKDTreeNode, bb: Tuple[np.ndarray, np.ndarray]
    ) -> List[np.ndarray]:
        def substitute_dim(bound: np.ndarray, sub_dim: int, sub: float):
            # replace bound[sub_dim] with sub
            return np.array([bound[i] if i != sub_dim else sub for i in range(3)])

        if node is None:
            return []

        if node.split_dim != -1:
            # if the split location is above the roi, no need to split (cannot be above None)
            if bb[1][node.split_dim] is not None and node.split > bb[1][node.split_dim]:
                return self._query_kdtree(node.lesser, bb)
            elif (
                bb[0][node.split_dim] is not None and node.split < bb[0][node.split_dim]
            ):
                return self._query_kdtree(node.greater, bb)
            else:
                return self._query_kdtree(node.greater, bb) + self._query_kdtree(
                    node.lesser, bb
                )
        else:
            # handle leaf node
            bbox = Roi(
                Coordinate(bb[0]),
                Coordinate(
                    tuple(
                        y - x if x is not None and y is not None else y
                        for x, y in zip(*bb)
                    )
                ),
            )
            points = [
                ind
                for ind, point in zip(node.indices, node.data_points)
                if bbox.contains(np.round(point))
            ]
            return points

    def _read_graph(self, filename: Path):
        graph = nx.read_gpickle(filename)
        if isinstance(graph, nx.Graph):
            graph = nx.to_directed(graph)
        graph = nx.convert_node_labels_to_integers(
            graph, first_label=len(self._graph.nodes)
        )
        self._graph = nx.union(self._graph, graph)

    def _search_swc_header(
        self, line: str, key: str, default: np.ndarray
    ) -> np.ndarray:
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

    def _subgraph_points(self, nodes: List[int], with_neighbors=False) -> nx.Graph:
        """
        Creates a subgraph of `graph` that contains the points in `nodes`.
        If `with_neighbors` is True, the subgraph contains all neighbors
        of all points in `nodes` as well.
        """
        sub_g = nx.Graph()
        subgraph_nodes = set(nodes)
        if with_neighbors:
            for n in nodes:
                for node in itertools.chain(
                    self.g.successors(n), self.g.predecessors(n)
                ):
                    subgraph_nodes.add(node)
        sub_g.add_nodes_from((n, self.g.nodes[n]) for n in subgraph_nodes)
        sub_g.add_edges_from(
            (n, nbr, d)
            for n in subgraph_nodes
            for nbr, d in self.g.adj[n].items()
            if nbr in subgraph_nodes
        )
        return sub_g

