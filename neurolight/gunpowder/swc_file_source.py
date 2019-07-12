from gunpowder.points import Point, Points, PointsKey
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.points_spec import PointsSpec
from gunpowder.batch import Batch
from gunpowder.profiling import Timing

import numpy as np
from scipy.spatial.ckdtree import cKDTree, cKDTreeNode
import networkx as nx

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SwcPoint(Point):
    def __init__(
        self,
        point_id: int,
        point_type: int,
        location: np.ndarray,
        radius: int,
        parent_id: int,
        label_id: int = None,
    ):

        super(SwcPoint, self).__init__(location)

        self.thaw()
        self.point_id = point_id
        self.parent_id = parent_id
        self.label_id = label_id
        self.radius = radius
        self.point_type = point_type
        self.freeze()

    def copy(self):
        return SwcPoint(
            point_id=self.point_id,
            point_type=self.point_type,
            location=self.location,
            radius=self.radius,
            parent_id=self.parent_id,
            label_id=self.label_id,
        )

    def __repr__(self):
        return "({}, {})".format(self.parent_id, self.location)


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
        points_spec: PointsSpec = None,
        scale: Coordinate = Coordinate([1, 1, 1]),
        keep_ids: bool = False,
        transpose: Tuple[int] = (0, 1, 2),
    ):

        self.filename = filename
        self.points = points
        self.points_spec = points_spec
        self.scale = scale
        self.connected_component_label = 0
        self.keep_ids = keep_ids
        self.g = nx.DiGraph()
        self.transpose = transpose

    def setup(self):

        self._read_points()

        if self.points_spec is not None:

            assert len(self.points) == len(self.points_spec)
            for point, point_spec in zip(self.points, self.points_spec):
                self.provides(point, point_spec)
        else:
            # If you don't provide a queryset, the roi will shrink to fit the
            # points in the swc(s). This may cause problems if you expect empty point
            # sets when querying the edges of your volume
            logger.debug("No point spec provided!")
            min_bb = Coordinate(self.data.mins[0:3])
            # cKDTree max is inclusive
            max_bb = Coordinate(self.data.maxes[0:3]) + Coordinate([1, 1, 1])

            roi = Roi(min_bb, max_bb - min_bb)

            for point in self.points:
                self.provides(point, PointsSpec(roi=roi))

    def provide(self, request: BatchRequest) -> Batch:

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for points_key in self.points:

            if points_key not in request:
                continue

            logger.debug(
                "Swc points source got request for %s", request[points_key].roi
            )

            # Retrieve all points in the requested region using a kdtree for speed
            points = self._query_kdtree(
                self.data.tree,
                (
                    np.array(request[points_key].roi.get_begin()),
                    np.array(request[points_key].roi.get_end()),
                ),
            )

            # Obtain subgraph that contains these points. Keep track of edges that
            # are present in the main graph, but not the subgraph
            sub_graph, predecessors, successors = self._points_to_graph(points)

            # Handle boundary cases
            self._handle_boundary_crossings(
                sub_graph, predecessors, successors, request[points_key].roi
            )

            # Convert graph into Points format
            points_data = self._graph_to_data(sub_graph)

            points_spec = PointsSpec(roi=request[points_key].roi.copy())

            batch = Batch()
            batch.points[points_key] = Points(points_data, points_spec)

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def _points_to_graph(
        self, points: List[np.ndarray]
    ) -> Tuple[nx.DiGraph, List[Tuple[int, int]], List[Tuple[int, int]]]:
        nodes = set([p[3] for p in points])

        sub_g = nx.DiGraph()
        sub_g.add_nodes_from((n, self.g.nodes[n]) for n in nodes)
        edges = []
        crossing_successors = []
        crossing_predecessors = []
        for n in nodes:
            for successor in self.g.successors(n):
                if successor in nodes:
                    edges.append((n, successor))
                else:
                    crossing_successors.append((n, successor))
            for predecessor in self.g.predecessors(n):
                if predecessor not in nodes:
                    crossing_predecessors.append((predecessor, n))

        sub_g.add_edges_from(edges)
        sub_g.graph.update(self.g.graph)

        return (sub_g, crossing_predecessors, crossing_successors)

    def _graph_to_data(self, graph: nx.DiGraph) -> Dict[int, SwcPoint]:
        graph = self._relabel_connected_components(graph, False)
        return {
            node: SwcPoint(
                point_id=node,
                parent_id=list(graph._pred[node].keys())[0]
                if len(graph._pred[node]) == 1
                else -1,
                **graph.nodes[node]
            )
            for node in graph.nodes
        }

    def _handle_boundary_crossings(
        self,
        graph: nx.DiGraph,
        predecessors: List[Tuple[int, int]],
        successors: List[Tuple[int, int]],
        roi: Roi,
    ):
        for pre, post in predecessors:
            out_parent = self.g.nodes[pre]
            current = self.g.nodes[post]
            loc = self._resample_relative(
                current["location"], out_parent["location"], roi
            )
            if loc is not None:
                graph.add_node(
                    pre,
                    point_type=out_parent["point_type"],
                    location=loc,
                    radius=out_parent["radius"],
                )
                graph.add_edge(pre, post)
        for pre, post in successors:
            current = self.g.nodes[pre]
            out_child = self.g.nodes[post]
            loc = self._resample_relative(
                current["location"], out_child["location"], roi
            )
            if loc is not None:
                graph.add_node(
                    post,
                    point_type=out_child["point_type"],
                    location=loc,
                    radius=out_child["radius"],
                )
                graph.add_edge(pre, post)

    def _resample_relative(
        self, inside: np.ndarray, outside: np.ndarray, bb: Roi
    ) -> Optional[np.ndarray]:
        offset = outside - inside
        with np.errstate(divide="ignore", invalid="ignore"):
            # bb_crossings will be 0 if inside is on the bb, 1 if outside is on the bb
            bb_x = np.asarray(
                [
                    (np.asarray(bb.get_begin()) - inside) / offset,
                    (np.asarray(bb.get_end() - Coordinate([1, 1, 1])) - inside)
                    / offset,
                ]
            )

            if np.sum(np.logical_and((bb_x > 0), (bb_x <= 1))) > 0:
                # all values of bb_x between 0, 1 represent a crossing of a bounding plane
                # the minimum of which is the (normalized) distance to the closest bounding plane
                s = np.min(bb_x[np.logical_and((bb_x > 0), (bb_x <= 1))])
                return Coordinate(np.floor(np.array(inside) + s * offset))
            else:
                logging.debug(
                    (
                        "Could not create a node on the bounding box {} "
                        + "given points (inside:{}, ouside:{})"
                    ).format(bb, inside, outside)
                )
                return None

    def _read_points(self) -> None:
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
            self._parse_swc(filepath)
        elif filepath.is_dir():
            # read from directory
            for swc_file in filepath.iterdir():
                if swc_file.name.endswith(".swc"):
                    self._parse_swc(swc_file)

        self._graph_to_kdtree()

    def _graph_to_kdtree(self) -> None:
        # add node_ids to coordinates to support overlapping nodes in cKDTree
        data = [
            tuple(node["location"]) + (node_id,)
            for node_id, node in self.g.nodes.items()
        ]
        # place nodes in the kdtree
        self.data = cKDTree(np.array(list(data)))

    def _query_kdtree(
        self, node: cKDTreeNode, bb: Tuple[np.ndarray, np.ndarray]
    ) -> List[np.ndarray]:
        def substitute_dim(bound: np.ndarray, sub_dim: int, sub: float):
            # replace bound[sub_dim] with sub
            return np.array([bound[i] if i != sub_dim else sub for i in range(3)])

        if node is None:
            return []

        if node.split_dim != -1:
            # recursive handling of child nodes
            greater_roi = (substitute_dim(bb[0], node.split_dim, node.split), bb[1])
            lesser_roi = (bb[0], substitute_dim(bb[1], node.split_dim, node.split))
            return self._query_kdtree(node.greater, greater_roi) + self._query_kdtree(
                node.lesser, lesser_roi
            )
        else:
            # handle leaf node
            # TODO: handle bounding box properly. bb[0], and bb[1] may not be integers.
            bbox = Roi(Coordinate(bb[0]), Coordinate(bb[1] - bb[0]))
            points = [point for point in node.data_points if bbox.contains(point)]
            return points

    def _parse_swc(self, filename: Path):
        """Read one point per line. If ``ndims`` is 0, all values in one line
        are considered as the location of the point. If positive, only the
        first ``ndims`` are used. If negative, all but the last ``-ndims`` are
        used.
        """
        # initialize file specific variables
        points = []
        header = True
        offset = np.array([0, 0, 0])
        resolution = np.array([1, 1, 1])

        # parse file
        with filename.open() as o_f:
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
                points.append(
                    {
                        "point_id": int(row[0]),
                        "parent_id": int(row[6]),
                        "point_type": int(row[1]),
                        "location": (
                            (np.array([float(x) for x in row[2:5]]) + offset)
                            * resolution
                        ).take(self.transpose),
                        "radius": float(row[5]),
                    }
                )
            self._add_points_to_source(points)

    def _search_swc_header(
        self, line: str, key: str, default: np.ndarray
    ) -> np.ndarray:
        # assumes header variables are seperated by spaces
        if key in line.lower():
            value = np.array(
                [float(x) for x in line.lower().split(key)[1].strip().split(" ")]
            )
            return value
        else:
            return default

    def _add_points_to_source(self, points: List[Dict]):
        # add points to a temporary graph
        temp_graph = nx.DiGraph()
        for point in points:
            temp_graph.add_node(
                point["point_id"],
                point_type=point["point_type"],
                location=point["location"],
                radius=point["radius"],
            )
            if point["parent_id"] != -1 and point["parent_id"] != point["point_id"]:
                # new connected component
                temp_graph.add_edge(point["parent_id"], point["point_id"])

        # check if the temporary graph is tree like
        if not nx.is_directed_acyclic_graph(temp_graph):
            raise ValueError("SWC skeleton is malformed: it contains a cycle.")

        # assign unique label id's to each connected component
        temp_graph = self._relabel_connected_components(temp_graph)

        # merge with the main graph
        self.g = nx.disjoint_union(self.g, temp_graph)

    def _relabel_connected_components(self, graph: nx.DiGraph, local: bool = False):
        # define i in case there are no connected components
        i = -1
        for i, connected_component in enumerate(nx.weakly_connected_components(graph)):
            label = i + self.connected_component_label if not local else i
            for node in connected_component:
                graph.nodes[node]["label_id"] = label

        if not local:
            self.connected_component_label += i + 1

        if not self.keep_ids:
            graph = nx.convert_node_labels_to_integers(graph)

        return graph

