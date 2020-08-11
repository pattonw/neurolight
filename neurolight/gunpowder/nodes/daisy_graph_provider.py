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

logger = logging.getLogger(__file__)

unbounded = Roi(Coordinate([None, None, None]), Coordinate([None, None, None]))


class DaisyGraphProvider(BatchProvider):
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
        edge_inclusion: str = "either",
        node_inclusion: str = "dangling",
        fail_on_inconsistent_node: bool = False,
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

        self.edge_inclusion = edge_inclusion
        self.node_inclusion = node_inclusion

        self.dbname = dbname
        self.nodes_collection = nodes_collection

        self.fail_on_inconsistent_node = fail_on_inconsistent_node
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
                edge_inclusion=self.edge_inclusion,
                node_inclusion=self.node_inclusion,
                node_attrs=self.node_attrs,
                edge_attrs=self.edge_attrs,
                nodes_filter=self.nodes_filter,
                edges_filter=self.edges_filter,
            )
            logger.debug(
                f"got {len(requested_graph.nodes)} nodes and {len(requested_graph.edges)} edges"
            )

            failed_nodes = []

            for node, attrs in requested_graph.nodes.items():
                try:
                    attrs["location"] = np.array(
                        attrs[self.position_attribute], dtype=np.float32
                    )
                except KeyError:
                    logger.warning(
                        f"node: {node} was written (probably part of an edge), but never given coordinates!"
                    )
                    failed_nodes.append(node)
                attrs["id"] = node

            for node in failed_nodes:
                if self.fail_on_inconsistent_node:
                    raise ValueError(
                        f"Mongodb contains node {node} without location! "
                        f"It was probably written as part of an edge"
                    )
                requested_graph.remove_node(node)

            if spec.directed:
                requested_graph = requested_graph.to_directed()
            else:
                requested_graph = requested_graph.to_undirected()

            points = Graph.from_nx_graph(requested_graph, spec)
            points.relabel_connected_components()
            points.crop(spec.roi)
            batch[key] = points

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch
