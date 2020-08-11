from gunpowder import BatchFilter, BatchRequest, Batch, Graph

import networkx as nx
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ThresholdEdges(BatchFilter):
    def __init__(
        self, msts, edge_threshold, component_threshold, distance_attr, msts_dense=None
    ):
        self.msts = msts
        self.msts_dense = msts_dense
        self.edge_threshold = edge_threshold
        self.component_threshold = component_threshold
        self.distance_attr = distance_attr

    def setup(self):
        self.provides(self.msts[1], self.spec[self.msts[0]].copy())
        if self.msts_dense is not None:
            self.provides(self.msts_dense[1], self.spec[self.msts_dense[0]].copy())

    def prepare(self, request):
        deps = BatchRequest()

        deps[self.msts[0]] = request[self.msts[1]].copy()
        if (self.msts_dense is not None) and (self.msts_dense[1] in request):
            deps[self.msts_dense[0]] = request[self.msts_dense[1]].copy()

        return deps

    def process(self, batch, request):
        mst = batch[self.msts[0]].to_nx_graph()
        
        logger.info(
            f"mst has {mst.number_of_nodes()} nodes and "
            f"{mst.number_of_edges()} edges and "
            f"{len(list(nx.connected_components(mst)))} components"
        )

        # threshold out edges
        if (self.msts_dense is not None) and (self.msts_dense[1] in request):
            dense_mst = batch[self.msts_dense[0]].to_nx_graph()

            for (u, v), chain in self.get_edge_chains(mst, dense_mst):

                distance = mst.edges[(u, v)][self.distance_attr]
                if distance > self.edge_threshold:
                    mst.remove_edge(u, v)
                    for u, v in zip(chain[:-1], chain[1:]):
                        dense_mst.remove_edge(u, v)

        else:
            for (u, v), attrs in mst.edges.items():
                distance = attrs[self.distance_attr]
                if distance > self.edge_threshold:
                    mst.remove_edge(u, v)

        # threshold out small components
        components_to_remove = []
        for component in nx.connected_components(mst):
            if np.isclose(self.component_threshold, 0):
                continue
            lower_bound = None
            upper_bound = None
            for node in component:
                loc = mst.nodes[node]["location"]
                if lower_bound is None:
                    lower_bound = loc
                if upper_bound is None:
                    upper_bound = loc
                lower_bound = np.min(np.array([lower_bound, loc]), axis=0)
                upper_bound = np.max(np.array([upper_bound, loc]), axis=0)
            if np.linalg.norm(upper_bound - lower_bound) < self.component_threshold:
                components_to_remove.append(component)

        for component in components_to_remove:
            mst_subgraph = mst.subgraph(component)

            if (self.msts_dense is not None) and (self.msts_dense[1] in request):
                for (u, v), chain in self.get_edge_chains(mst_subgraph, dense_mst):
                    mst.remove_edge(u, v)
                    for (u, v) in zip(chain[:-1], chain[1:]):
                        dense_mst.remove_edge(u, v)

            else:
                for (u, v) in mst_subgraph.edges:
                    mst.remove_edge(u, v)

        for node in list(mst.nodes):
            if mst.degree(node) == 0:
                mst.remove_node(node)

        if (self.msts_dense is not None) and (self.msts_dense[1] in request):
            for node in list(dense_mst.nodes):
                if dense_mst.degree(node) == 0:
                    dense_mst.remove_node(node)

        logger.info(
            f"mst has {mst.number_of_nodes()} nodes and "
            f"{mst.number_of_edges()} edges and "
            f"{len(list(nx.connected_components(mst)))} components"
        )

        outputs = Batch()
        outputs[self.msts[1]] = Graph.from_nx_graph(mst, batch[self.msts[0]].spec)
        outputs[self.msts[1]].relabel_connected_components()

        if (self.msts_dense is not None) and (self.msts_dense[1] in request):
            outputs[self.msts_dense[1]] = Graph.from_nx_graph(
                dense_mst, batch[self.msts_dense[0]].spec
            )
            outputs[self.msts_dense[1]].relabel_connected_components()

        return outputs

    def get_edge_chains(self, mst, dense_mst):
        location_lookup = {}
        for node, attrs in mst.nodes.items():
            location_lookup[tuple(int(x) for x in attrs["location"])] = node

        mst_nodes_to_dense = {}
        for node, attrs in dense_mst.nodes.items():
            if tuple(int(x) for x in attrs["location"]) in location_lookup:
                mst_nodes_to_dense[
                    location_lookup[tuple(int(x) for x in attrs["location"])]
                ] = node

        for u, v in mst.edges:
            yield (u, v), nx.shortest_path(
                dense_mst, mst_nodes_to_dense[u], mst_nodes_to_dense[v]
            )
