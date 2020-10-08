from funlib import math
import gunpowder as gp
import numpy as np

from daisy.persistence import MongoDbGraphProvider

import logging
from typing import List
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class MongoRemoveDifference(gp.BatchFilter):
    """
    Node to remove difference between mongodb graph and the current graph.
    Usage: read a mongo graph, filter out undesired nodes/edges, delete difference
    from db.
    """
    def __init__(
        self, graph, db_host, graph_db, block_read_shape, block_write_shape, collection
    ):
        self.graph = graph
        self.db_host = db_host
        self.graph_db = graph_db
        self.block_read_shape = block_read_shape
        self.block_write_shape = block_write_shape
        self.collection = collection


class MongoWriteGraph(gp.BatchFilter):
    def __init__(
        self,
        mst,
        db_host,
        db_name,
        read_size,
        write_size,
        voxel_size,
        directed,
        mode="r+",
        collection="",
        position_attr="position",
        node_attrs: List[str] = None,
        edge_attrs: List[str] = None,
    ):

        self.mst = mst
        self.db_host = db_host
        self.db_name = db_name
        self.client = None

        self.voxel_size = voxel_size
        self.read_size = read_size
        self.write_size = write_size
        self.context = (read_size - write_size) / 2

        assert self.write_size + (self.context * 2) == self.read_size

        self.directed = directed
        self.mode = mode
        self.position_attr = position_attr

        self.nodes_collection = f"{collection}_nodes"
        self.edges_collection = f"{collection}_edges"

        self.node_attrs = node_attrs if node_attrs is not None else []
        self.node_attrs += [position_attr]
        self.edge_attrs = edge_attrs if edge_attrs is not None else []

    def setup(self):

        # Initialize client. Doesn't the daisy mongodb graph provider handle this?
        if self.client is None:
            self.client = MongoDbGraphProvider(
                self.db_name,
                self.db_host,
                mode=self.mode,
                directed=self.directed,
                nodes_collection=self.nodes_collection,
                edges_collection=self.edges_collection,
            )

        self.updates(self.mst, self.spec[self.mst].copy())

    def prepare(self, request):
        deps = gp.BatchRequest()
        deps[self.mst] = request[self.mst].copy()
        return deps

    def process(self, batch, request):

        voxel_size = self.voxel_size

        read_roi = batch[self.mst].spec.roi
        write_roi = read_roi.grow(-self.context, -self.context)

        mst = batch[self.mst].to_nx_graph()

        # get saved graph with 'dangling' edges. Some nodes may not be in write_roi
        # Edges crossing boundary should only be saved if lower id is contained in write_roi
        mongo_graph = self.client.get_graph(
            write_roi,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            node_inclusion="dangling",
            edge_inclusion="either",
        )

        for node in list(mongo_graph.nodes()):
            if self.position_attr not in mongo_graph.nodes[node]:
                mongo_graph.remove_node(node)

        num_created_nodes = 0
        num_updated_nodes = 0
        for node, attrs in mst.nodes.items():
            loc = attrs["location"]
            pos = np.floor(loc / voxel_size)
            node_id = int(math.cantor_number(pos))

            # only update attributes of nodes in the write_roi
            if write_roi.contains(loc):
                attrs_to_save = {self.position_attr: tuple(float(v) for v in loc)}
                for attr in self.node_attrs:
                    if attr in attrs:
                        value = attrs[attr]
                        if isinstance(value, Iterable):
                            value = tuple(float(v) for v in value)
                        if isinstance(value, np.float32) or (
                            isinstance(value, Iterable)
                            and any([isinstance(v, np.float32) for v in value])
                        ):
                            raise ValueError(f"value of attr {attr} is a np.float32")
                        attrs_to_save[attr] = value

                if node_id in mongo_graph:
                    num_updated_nodes += 1
                    mongo_attrs = mongo_graph.nodes[node_id]
                    mongo_attrs.update(attrs_to_save)

                else:
                    num_created_nodes += 1
                    mongo_graph.add_node(node_id, **attrs_to_save)

        num_created_edges = 0
        num_updated_edges = 0
        for (u, v), attrs in mst.edges.items():
            u_loc = mst.nodes[u]["location"]
            u_pos = np.floor(u_loc / voxel_size)
            u_id = int(math.cantor_number(u_pos))

            v_loc = mst.nodes[v]["location"]
            v_pos = np.floor(v_loc / voxel_size)
            v_id = int(math.cantor_number(v_pos))

            # node a is the node with the lower id out of u, v
            a_loc, a_id, b_loc, b_id = (
                (u_loc, u_id, v_loc, v_id)
                if u_id < v_id
                else (v_loc, v_id, u_loc, u_id)
            )
            # only write edge if a is contained
            # may create a node without any attributes if node creation is
            # dependent on your field of view and the neighboring block
            # fails to create the same node.
            if write_roi.contains(a_loc):

                attrs_to_save = {}
                for attr in self.edge_attrs:
                    if attr in attrs:
                        value = attrs[attr]
                        if isinstance(value, Iterable):
                            value = tuple(float(v) for v in value)
                        if isinstance(value, np.float32) or (
                            isinstance(value, Iterable)
                            and any([isinstance(v, np.float32) for v in value])
                        ):
                            raise ValueError(f"value of attr {attr} is a np.float32")
                        attrs_to_save[attr] = value

                if (u_id, v_id) in mongo_graph.edges:
                    num_updated_edges += 1
                    mongo_attrs = mongo_graph.edges[(u_id, v_id)]
                    mongo_attrs.update(attrs_to_save)
                else:
                    num_created_edges += 1
                    mongo_graph.add_edge(u_id, v_id, **attrs_to_save)

        for node in mst.nodes:
            if node in mongo_graph.nodes and write_roi.contains(
                mst.nodes[node]["location"]
            ):
                assert all(
                    np.isclose(
                        mongo_graph.nodes[node][self.position_attr],
                        mst.nodes[node]["location"],
                    )
                )
                if write_roi.contains(mongo_graph.nodes[node][self.position_attr]):
                    assert (
                        mongo_graph.nodes[node]["component_id"]
                        == mst.nodes[node]["component_id"]
                    )

        if len(mongo_graph.nodes) > 0:
            mongo_graph.write_nodes(roi=write_roi, attributes=self.node_attrs)

        if len(mongo_graph.edges) > 0:
            for edge, attrs in mongo_graph.edges.items():
                for attr in self.edge_attrs:
                    assert attr in attrs
            mongo_graph.write_edges(roi=write_roi, attributes=self.edge_attrs)
