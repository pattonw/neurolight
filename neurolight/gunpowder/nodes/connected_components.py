from gunpowder import BatchFilter, BatchRequest, Batch, Graph
from daisy.persistence import MongoDbGraphProvider
from daisy.roi import Roi

import numpy as np
import networkx as nx

import logging

logger = logging.getLogger(__name__)


class ConnectedComponents(BatchFilter):
    def __init__(self, graph, size_attr, component_attr, read_size, write_size):
        self.graph = graph
        self.size_attr = size_attr
        self.component_attr = component_attr
        self.read_size = read_size
        self.write_size = write_size

        self.context = (read_size - write_size) / 2

        assert self.read_size == self.write_size + self.context * 2

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        assert (
            request[self.graph].roi.get_shape() == self.read_size
        ), f"Got wrong size graph in request"

        return deps

    def process(self, batch, request):
        g = batch[self.graph].to_nx_graph()
        assert batch[self.graph].spec.roi.get_shape() == self.read_size

        logger.debug(
            f"{self.name()} got graph with {g.number_of_nodes()} nodes, and "
            f"{g.number_of_edges()} edges!"
        )

        write_roi = batch[self.graph].spec.roi.grow(-self.context, -self.context)

        cc_func = (
            nx.connected_components
            if not g.is_directed()
            else nx.weakly_connected_components
        )

        for cc in cc_func(g):
            contained_nodes = [
                n for n in cc if write_roi.contains(g.nodes[n]["location"])
            ]
            if len(contained_nodes) == 0:
                continue
            else:
                cc_id = min(contained_nodes)
                cc_subgraph = g.subgraph(cc)

                # total edge length of this connected component in this write_roi
                total_edge_len = 0

                for u, v in cc_subgraph.edges:
                    u_loc = cc_subgraph.nodes[u]["location"]
                    v_loc = cc_subgraph.nodes[v]["location"]
                    edge_len = np.linalg.norm(u_loc - v_loc)
                    if write_roi.contains(u_loc) and write_roi.contains(v_loc):
                        total_edge_len += edge_len
                    elif write_roi.contains(u_loc) or write_roi.contains(v_loc):
                        total_edge_len += edge_len / 2

                for u in contained_nodes:
                    attrs = cc_subgraph.nodes[u]
                    attrs[self.component_attr] = int(cc_id)
                    attrs[self.size_attr] = float(total_edge_len)

        count = 0
        for node, attrs in g.nodes.items():
            if write_roi.contains(attrs["location"]):
                assert self.component_attr in attrs
                count += 1

        logger.debug(
            f"{self.name()} updated component id of {count} nodes in write_roi"
        )

        outputs = Batch()
        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())

        return outputs


class GlobalConnectedComponents(BatchFilter):
    def __init__(
        self,
        db_name,
        db_host,
        collection,
        graph,
        size_attr,
        component_attr,
        read_size,
        write_size,
        mode="r+",
    ):
        self.db_name = db_name
        self.db_host = db_host
        self.collection = collection
        self.component_collection = f"{self.collection}_components"
        self.component_edge_collection = f"{self.collection}_component_edges"
        self.client = None
        self.mode = mode

        self.graph = graph
        self.size_attr = size_attr
        self.component_attr = component_attr
        self.read_size = read_size
        self.write_size = write_size

        self.context = (read_size - write_size) / 2

        assert self.read_size == self.write_size + self.context * 2

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

        if self.client is None:
            self.client = MongoDbGraphProvider(
                self.db_name,
                self.db_host,
                mode=self.mode,
                directed=False,
                nodes_collection=self.component_collection,
                edges_collection=self.component_edge_collection,
            )

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        assert (
            request[self.graph].roi.get_shape() == self.read_size
        ), f"Got wrong size graph in request"

        return deps

    def process(self, batch, request):
        g = batch[self.graph].to_nx_graph()

        logger.debug(
            f"{self.name()} got graph with {g.number_of_nodes()} nodes, and "
            f"{g.number_of_edges()} edges!"
        )

        write_roi = batch[self.graph].spec.roi.grow(-self.context, -self.context)

        contained_nodes = [
            node
            for node, attr in g.nodes.items()
            if write_roi.contains(attr["location"])
        ]
        contained_components = set(
            g.nodes[n][self.component_attr] for n in contained_nodes
        )

        logger.debug(
            f"Graph contains {len(contained_nodes)} nodes with "
            f"{len(contained_components)} components in write_roi"
        )

        component_graph = self.client.get_graph(
            roi=write_roi, node_inclusion="dangling", edge_inclusion="either"
        )

        for node in contained_nodes:
            attrs = g.nodes[node]
            block_component_id = attrs[self.component_attr]
            global_component_id = component_graph.nodes[block_component_id][
                self.component_attr
            ]
            attrs[self.component_attr] = global_component_id
            attrs[self.size_attr] = component_graph.nodes[block_component_id][
                self.size_attr
            ]

        logger.debug(
            f"Graph contains {len(contained_nodes)} nodes with "
            f"{len(contained_components)} components in write_roi"
        )

        outputs = Batch()
        outputs[self.graph] = Graph.from_nx_graph(g, batch[self.graph].spec.copy())

        return outputs


class MongoWriteComponents(BatchFilter):
    def __init__(
        self,
        graph,
        db_host,
        db_name,
        read_size,
        write_size,
        component_attr,
        node_attrs=None,
        collection="",
        mode="r+",
    ):
        self.graph = graph
        self.db_host = db_host
        self.db_name = db_name
        self.read_size = read_size
        self.write_size = write_size
        self.context = (read_size - write_size) / 2
        self.mode = mode
        self.component_attr = component_attr

        assert self.write_size + (self.context * 2) == self.read_size

        self.collection = collection
        self.component_collection = f"{self.collection}_components"
        self.component_edge_collection = f"{self.collection}_component_edges"

        if node_attrs is None:
            self.node_attrs = []
        else:
            self.node_attrs = node_attrs

        self.client = None

    def setup(self):
        self.updates(self.graph, self.spec[self.graph].copy())

        if self.client is None:
            self.client = MongoDbGraphProvider(
                self.db_name,
                self.db_host,
                mode=self.mode,
                directed=False,
                nodes_collection=self.component_collection,
                edges_collection=self.component_edge_collection,
            )

    def prepare(self, request):
        deps = BatchRequest()
        deps[self.graph] = request[self.graph].copy()
        assert (
            request[self.graph].roi.get_shape() == self.read_size
        ), f"Got wrong size graph in request"

        return deps

    def process(self, batch, request):

        graph = batch[self.graph].to_nx_graph()

        logger.debug(
            f"{self.name()} got graph with {graph.number_of_nodes()} nodes and "
            f"{graph.number_of_edges()} edges!"
        )

        write_roi = batch[self.graph].spec.roi.grow(-self.context, -self.context)

        contained_nodes = [
            node
            for node, attr in graph.nodes.items()
            if write_roi.contains(attr["location"])
        ]
        contained_components = set(
            graph.nodes[n][self.component_attr] for n in contained_nodes
        )

        logger.debug(
            f"Graph contains {len(contained_nodes)} nodes with "
            f"{len(contained_components)} components in write_roi"
        )

        mongo_graph = self.client.get_graph(
            write_roi,
            node_attrs=[],
            edge_attrs=[],
            node_inclusion="dangling",
            edge_inclusion="either",
        )

        num_new_components = 0
        num_new_component_edges = 0
        for node, attrs in graph.nodes.items():
            if write_roi.contains(attrs["location"]):
                cc_id = attrs[self.component_attr]
                node_attrs = {}
                for attr in self.node_attrs:
                    node_attrs[attr] = attrs[attr]
                if cc_id not in mongo_graph.nodes:
                    num_new_components += 1
                    mongo_graph.add_node(
                        cc_id, position=write_roi.get_center(), **node_attrs
                    )
                else:
                    for k, v in node_attrs.items():
                        assert mongo_graph.nodes[cc_id][k] == v

        # Always write crossing edges if neither component id is None. Other end
        # point wont have a component id unless it has already been processed,
        # thus it is the duty of the second pass to write the edge, regardless
        # of whether the lower or upper end point is contained.
        for u, v in graph.edges:
            u_loc = graph.nodes[u]["location"]
            v_loc = graph.nodes[v]["location"]
            if write_roi.contains(u_loc) or write_roi.contains(v_loc):
                u_cc_id = graph.nodes[u].get(self.component_attr)
                v_cc_id = graph.nodes[v].get(self.component_attr)

                if u_cc_id is None or v_cc_id is None:
                    continue
                elif u_cc_id == v_cc_id:
                    continue
                elif (u_cc_id, v_cc_id) not in mongo_graph.edges:
                    num_new_component_edges += 1
                    mongo_graph.add_edge(u_cc_id, v_cc_id)

        logger.debug(
            f"{self.name()} writing {num_new_components} new components and "
            f"{num_new_component_edges} new component edges!"
        )

        mongo_graph.write_nodes()
        mongo_graph.write_edges(ignore_v=False)

