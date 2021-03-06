import numpy as np

from daisy.persistence import MongoDbGraphProvider

from gunpowder.batch import Batch
from gunpowder.nodes.batch_provider import BatchProvider
from gunpowder.graph import GraphKey, Graph
from gunpowder.graph_spec import GraphSpec
from gunpowder.roi import Roi
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing

import logging
from typing import Tuple, List, Optional, Union, Dict, Any
import random

logger = logging.getLogger(__file__)

unbounded = Roi(Coordinate([None, None, None]), Coordinate([None, None, None]))


class FilteredDaisyGraphProvider(BatchProvider):
    """
    See documentation for mongo graph provider at
    https://github.com/funkelab/daisy/blob/0.3-dev/daisy/persistence/mongodb_graph_provider.py#L17
    """

    def __init__(
        self,
        dbname: str,
        url: str,
        points: List[GraphKey],
        graph_specs: Optional[Union[GraphSpec, List[GraphSpec]]] = None,
        directed: bool = False,
        total_roi: Roi = None,
        nodes_collection: str = "nodes",
        edges_collection: str = "edges",
        meta_collection: str = "meta",
        endpoint_names: Tuple[str, str] = ("u", "v"),
        position_attribute: str = "position",
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        nodes_filter: Optional[Dict[str, Any]] = None,
        edges_filter: Optional[Dict[str, Any]] = None,
        num_nodes=100000,
        dist_attribute=None,
        min_dist=29000,
    ):
        self.points = points
        graph_specs = (
            graph_specs
            if graph_specs is not None
            else GraphSpec(
                Roi(Coordinate([None] * 3), Coordinate([None] * 3)), directed=False
            )
        )
        specs = (
            graph_specs
            if isinstance(graph_specs, list) and len(graph_specs) == len(points)
            else [graph_specs] * len(points)
        )
        self.specs = {key: spec for key, spec in zip(points, specs)}

        self.position_attribute = position_attribute
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs
        self.nodes_filter = nodes_filter
        self.edges_filter = edges_filter
        self.dist_attribute = dist_attribute
        self.min_dist = min_dist

        self.num_nodes = num_nodes

        self.graph_provider = MongoDbGraphProvider(
            dbname,
            url,
            mode="r+",
            directed=directed,
            total_roi=None,
            nodes_collection=nodes_collection,
            edges_collection=edges_collection,
            meta_collection=meta_collection,
            endpoint_names=endpoint_names,
            position_attribute=position_attribute,
        )

    def setup(self):
        for key, spec in self.specs.items():
            self.provides(key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for key, spec in request.items():
            logger.debug(f"fetching {key} in roi {spec.roi}")
            requested_graph = self.graph_provider.get_graph(
                spec.roi,
                edge_inclusion="either",
                node_inclusion="dangling",
                node_attrs=self.node_attrs,
                edge_attrs=self.edge_attrs,
                nodes_filter=self.nodes_filter,
                edges_filter=self.edges_filter,
            )
            logger.debug(
                f"got {len(requested_graph.nodes)} nodes and {len(requested_graph.edges)} edges"
            )
            for node, attrs in list(requested_graph.nodes.items()):
                if self.dist_attribute in attrs:
                    if attrs[self.dist_attribute] < self.min_dist:
                        requested_graph.remove_node(node)

            logger.debug(
                f"{len(requested_graph.nodes)} nodes remaining after filtering by distance"
            )
            
            if len(requested_graph.nodes) > self.num_nodes:
                nodes = list(requested_graph.nodes)
                nodes_to_keep = set(random.sample(nodes, self.num_nodes))

                for node in list(requested_graph.nodes()):
                    if node not in nodes_to_keep:
                        requested_graph.remove_node(node)

            for node, attrs in requested_graph.nodes.items():
                attrs["location"] = np.array(
                    attrs[self.position_attribute], dtype=np.float32
                )
                attrs["id"] = node

            if spec.directed:
                requested_graph = requested_graph.to_directed()
            else:
                requested_graph = requested_graph.to_undirected()

            logger.debug(
                f"providing {key} with {len(requested_graph.nodes)} nodes and {len(requested_graph.edges)} edges"
            )

            points = Graph.from_nx_graph(requested_graph, spec)
            points.crop(spec.roi)
            batch[key] = points

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
