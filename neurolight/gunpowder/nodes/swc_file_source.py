from gunpowder.points import PointsKey
from gunpowder.graph_points import GraphPoint, GraphPoints, SpatialGraph
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

logger = logging.getLogger(__name__)


class SwcFileSource(BatchProvider):
    """Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`PointsKey`):

            The key of the points set to create.

        points_spec (:class:`PointsSpec`, optional):

            An optional :class:`PointsSpec` to overwrite the points specs
            automatically determined from the CSV file. This is useful to set
            the :class:`Roi` manually.

        scale (scalar or array-like):

            An optional scaling to apply to the coordinates of the points read
            from the CSV file. This is useful if the points refer to voxel
            positions to convert them to world units.

        keep_ids (boolean):

            In the case of having to generate a new node at the intersection
            of a parent-child edge and the bounding box, should the generated
            node be given the id of the outside node? Default behavior is to
            always relabel nodes for each request so that the node id's lie
            in [0,n) if the request contains n nodes.

        transpose:

            The swc's store coordinates in xyz order as defined by the spec: 
            "http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html"
            This transpose term allows you to transpose the coordinates upon
            parsing the swc.

    """

    def __init__(
        self,
        filename: Path,
        points: List[PointsKey],
        points_spec: List[PointsSpec] = None,
        scale: Coordinate = Coordinate([1, 1, 1]),
        keep_ids: bool = False,
        transpose: Tuple[int] = (0, 1, 2),
        radius: Coordinate = Coordinate((1000, 1000, 1000)),
        directed: bool = True,
    ):
        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.connected_component_label = 0
        self.keep_ids = keep_ids
        self._graph = SpatialGraph()
        self.transpose = transpose
        self.radius = radius
        self.directed = directed

    @property
    def g(self):
        return self._graph

    def setup(self):
        self._read_points()

        if self.points_spec is not None:

            assert len(self.points) == len(self.points_spec)
            for point, point_spec in zip(self.points, self.points_spec):
                self.provides(point, point_spec)
        else:
            logger.debug("No point spec provided!")
            min_bb = Coordinate(self.data.mins[0:3]) - self.radius
            # cKDTree max is inclusive
            max_bb = Coordinate(self.data.maxes[0:3]) + self.radius

            roi = Roi(min_bb, max_bb - min_bb)

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
            points = self._query_kdtree(
                self.data.tree,
                (
                    np.array(request[points_key].roi.get_begin()),
                    np.array(request[points_key].roi.get_end()),
                ),
            )

            point_locations = {int(point[3]): np.array(point[0:3]) for point in points}

            # To account for boundary crossings we must retrieve neighbors of all points
            # in the graph. This is too slow for large queries and less important
            points_subgraph = self._subgraph_points(
                list(point_locations.keys()), with_neighbors=len(point_locations) < 1000
            )

            # Handle boundary cases
            points_subgraph = points_subgraph.crop(request[points_key].roi)

            batch = Batch()
            batch.points[points_key] = GraphPoints._from_graph(
                points_subgraph, request[points_key]
            )

            logger.debug(
                "Swc points source provided {} points for roi: {}".format(
                    len(batch.points[points_key].data), request[points_key].roi
                )
            )

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _read_points(self) -> None:
        filepath = Path(self.filename)
        # handle missing file case
        if not filepath.exists():
            raise FileNotFoundError(
                "file {} does not exist at {}".format(
                    filepath.name, filepath.absolute()
                )
            )

        self.num_ccs = 0
        if filepath.is_file():
            # read from single file
            self.num_ccs = self._parse_swc(filepath)
        elif filepath.is_dir():
            # read from directory
            for swc_file in filepath.iterdir():
                if swc_file.name.endswith(".swc"):
                    self.num_ccs += self._parse_swc(swc_file)

        self._graph_to_kdtree()
        assert len(list(nx.weakly_connected_components(self.g))) == self.num_ccs

    def _graph_to_kdtree(self) -> None:
        # add node_ids to coordinates to support overlapping nodes in cKDTree
        ids, attrs = zip(*self.g.nodes.items())
        inds, ids = zip(*enumerate(ids))
        self.ind_to_id_map = {ind: i for ind, i in zip(inds, ids)}
        data = [tuple(point_attrs["location"]) for point_attrs in attrs]
        logger.debug("placing {} nodes in the kdtree".format(len(data)))
        # place nodes in the kdtree
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
            if (
                node.split_dim < 3
                and bb[1][node.split_dim] is not None
                and node.split > bb[1][node.split_dim]
            ):
                return self._query_kdtree(node.lesser, bb)
            elif (
                node.split_dim < 3
                and bb[0][node.split_dim] is not None
                and node.split < bb[0][node.split_dim]
            ):
                return self._query_kdtree(node.greater, bb)
            else:
                greater_roi = (
                    substitute_dim(bb[0], node.split_dim, np.floor(node.split)),
                    bb[1],
                )
                lesser_roi = (
                    bb[0],
                    substitute_dim(bb[1], node.split_dim, np.ceil(node.split)),
                )
                return self._query_kdtree(
                    node.greater, greater_roi
                ) + self._query_kdtree(node.lesser, lesser_roi)
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
                tuple(point) + (self.ind_to_id_map[ind],)
                for ind, point in zip(node.indices, node.data_points)
                if bbox.contains(point)
            ]
            return points

    def _parse_swc(self, filename: Path):
        """Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        """
        # initialize file specific variables
        header = True
        offset = np.array([0, 0, 0])
        resolution = np.array([1, 1, 1])

        # parse file
        with filename.open() as o_f:
            points = {}
            edges = set()
            for line in o_f.read().splitlines():
                if header and line.startswith("#"):
                    # Search header comments for variables
                    offset = self._search_swc_header(line, "offset", offset)
                    resolution = self._search_swc_header(line, "resolution", resolution)
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
                assert int(row[0]) not in points, "Duplicate point {} found!".format(
                    int(row[0])
                )
                points[int(row[0])] = GraphPoint(
                    point_type=int(row[1]),
                    location=np.array(
                        (
                            (np.array([float(x) for x in row[2:5]]) + offset)
                            * resolution
                            * self.scale
                        ).take(self.transpose)
                    ),
                    radius=float(row[5]),
                )
                v, u = int(row[0]), int(row[6])
                assert (u, v) not in edges, "Duplicate edge {} found!".format((u, v))
                if u != v and u is not None and v is not None and u >= 0 and v >= 0:
                    edges.add((u, v))
            self._add_points_to_source(points, edges)

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

    def _add_points_to_source(
        self, points: Dict[int, GraphPoint], edges: List[Tuple[int, int]]
    ):

        # add points to a temporary graph
        logger.debug("adding {} nodes to graph".format(len(points)))
        temp = SpatialGraph()
        for point_id, graph_point in points.items():
            temp.add_node(point_id, **graph_point.attrs)
        for u, v in edges:
            if u in temp.nodes and v in temp.nodes:
                temp.add_edge(u, v)

        temp = nx.convert_node_labels_to_integers(temp, first_label=len(self._graph.nodes))

        self._graph = nx.union(self._graph, temp)
        logger.debug("graph has {} nodes".format(len(self._graph.nodes)))

    def _subgraph_points(self, nodes: List[int], with_neighbors=False) -> SpatialGraph:
        """
        Creates a subgraph of `graph` that contains the points in `nodes`.
        If `with_neighbors` is True, the subgraph contains all neighbors
        of all points in `nodes` as well.
        """
        sub_g = SpatialGraph()
        subgraph_nodes = set(nodes)
        if with_neighbors:
            for n in nodes:
                for successor in self.g.successors(n):
                    subgraph_nodes.add(successor)
                for predecessor in self.g.predecessors(n):
                    subgraph_nodes.add(predecessor)
        sub_g.add_nodes_from((n, self.g.nodes[n]) for n in subgraph_nodes)
        sub_g.add_edges_from(
            (n, nbr, d)
            for n in subgraph_nodes
            for nbr, d in self.g.adj[n].items()
            if nbr in subgraph_nodes
        )
        return sub_g

