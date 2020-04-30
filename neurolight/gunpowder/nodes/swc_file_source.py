from gunpowder.points import GraphKey
from gunpowder.graph import Node, Graph, Edge
from gunpowder.graph_spec import GraphSpec
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.batch_request import BatchRequest
from gunpowder.coordinate import Coordinate
from gunpowder.roi import Roi
from gunpowder.graph_spec import GraphSpec
from gunpowder.batch import Batch
from gunpowder.profiling import Timing
import numpy as np
from scipy.spatial.ckdtree import cKDTree, cKDTreeNode

from neurolight.transforms.swc_to_graph import parse_swc_no_transform

from pathlib import Path
from typing import List, Dict, Tuple
import logging
import networkx as nx

logger = logging.getLogger(__name__)


class SwcFileSource(BatchProvider):
    """Read a set of points from a comma-separated-values text file. Each line
    in the file represents one point.

    Args:

        filename (``string``):

            The file to read from.

        points (:class:`GraphKey`):

            The key of the points set to create.

        points_spec (:class:`GraphSpec`, optional):

            An optional :class:`GraphSpec` to overwrite the points specs
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
        points: List[GraphKey],
        points_spec: List[GraphSpec] = None,
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
        self._graph = nx.DiGraph()
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
                assert (
                    point_spec.directed is None or point_spec.directed == self.directed
                )
                point_spec.directed = self.directed
                self.provides(point, point_spec)
        else:
            logger.debug("No point spec provided!")
            min_bb = Coordinate(self.data.mins[0:3]) - self.radius
            # cKDTree max is inclusive
            max_bb = Coordinate(self.data.maxes[0:3]) + self.radius

            roi = Roi(min_bb, max_bb - min_bb)

            for point in self.points:
                self.provides(point, GraphSpec(roi=roi, directed=self.directed))

        for i, wcc in enumerate(nx.weakly_connected_components(self._graph)):
            for node in wcc:
                self._graph.nodes[node]["component"] = i

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
            nodes = [
                Node(id=node, location=attrs["location"], attrs=attrs)
                for node, attrs in points_subgraph.nodes.items()
            ]
            edges = [Edge(u, v) for u, v in points_subgraph.edges]
            return_graph = Graph(
                nodes,
                edges,
                GraphSpec(roi=request[points_key].roi, directed=self.directed),
            )
            return_graph = return_graph.crop(request[points_key].roi)

            batch = Batch()
            batch.graphs[points_key] = return_graph

            logger.debug(
                "Swc points source provided {} points for roi: {}".format(
                    len(list(batch.graphs[points_key].nodes)), request[points_key].roi
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
        logger.debug("kdtree initialized")

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
                return self._query_kdtree(node.greater, bb) + self._query_kdtree(
                    node.lesser, bb
                )
        else:
            # handle leaf node
            bbox = Roi(Coordinate(bb[0]), Coordinate(bb[1]))
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
        tree = parse_swc_no_transform(
            filename,
            resolution=[self.scale[i] for i in self.transpose],
            transpose=self.transpose,
        )

        assert len(list(nx.weakly_connected_components(tree))) == 1

        points = []

        for node, attrs in tree.nodes.items():
            attrs["id"] = node
            points.append(Node(id=attrs["id"], location=attrs["location"], attrs=attrs))
        self._add_points_to_source(points, set(Edge(u, v) for u, v in tree.edges))

        return len(list(nx.weakly_connected_components(tree)))

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

    def _add_points_to_source(self, points: List[Node], edges: List[Edge]):

        # add points to a temporary graph
        logger.debug("adding {} nodes to graph".format(len(points)))
        temp = nx.Graph()
        for node in points:
            temp.add_node(node.id, **node.attrs)
        for e in edges:
            temp.add_edge(e.u, e.v)

        temp = nx.convert_node_labels_to_integers(
            temp, first_label=len(self._graph.nodes)
        )

        self._graph = nx.disjoint_union(self._graph, temp)
        logger.debug("graph has {} nodes".format(len(self._graph.nodes)))

    def _subgraph_points(self, nodes: List[int], with_neighbors=False) -> nx.Graph:
        """
        Creates a subgraph of `graph` that contains the points in `nodes`.
        If `with_neighbors` is True, the subgraph contains all neighbors
        of all points in `nodes` as well.
        """
        sub_g = nx.DiGraph()
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
